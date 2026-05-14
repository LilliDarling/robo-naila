"""Weather action handler with auto-location and utterance parsing.

Resolution per turn:
  1. Try to extract a location from the utterance ("in Tokyo", "for Boston").
  2. If extracted, geocode it (Open-Meteo geocoding API). On success, use it.
  3. Otherwise — or if the geocode came back empty — use the IP-detected
     server location. The IP location is cached in-process AND persisted to
     ``data/ip_location.json`` with a 30-day TTL, so cold starts after the
     first run never hit ip-api.com.
  4. Call Open-Meteo's forecast endpoint and format the reply.

If the IP lookup network call fails AND we have a cached value (even an
expired one), we fall back to the cache — stale location is always better
than no weather.

Per doc §3.4 the handler catches its own errors and always returns a string.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import httpx

from agents.actions import registry
from config import weather as cfg
from utils import get_logger


logger = get_logger(__name__)


_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
# ip-api.com offers 45 req/min with no daily cap. HTTP-only on the free tier;
# the data is non-sensitive lat/lon/city, MITM risk = wrong city in weather.
_IP_GEO_URL = "http://ip-api.com/json/"

_FRIENDLY_ERROR = "I couldn't reach the weather service right now. Please try again in a moment."

# WMO weather codes → short human descriptions.
# https://open-meteo.com/en/docs#weathervariables
_WEATHER_CODES = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


# Process-level caches.
_IP_LOCATION: Optional[Tuple[float, float, str]] = None
_GEOCODE_CACHE: dict = {}

# Persistent cache for the IP geo lookup. Path is anchored to the ai-server
# package dir so it stays put regardless of cwd at launch time, matching
# config/memory.py's pattern.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
_CACHE_PATH = str(_PACKAGE_DIR / "data" / "ip_location.json")
_CACHE_TTL_SECONDS = 30 * 86400  # 30 days


# Match "in/for/at <rest of sentence>". Trailing punctuation and stopwords
# are stripped after the match. The geocoder is the real filter — bad parses
# just fail to geocode and fall back to IP.
_LOCATION_PATTERN = re.compile(
    r"\b(?:in|for|at)\s+(.+?)(?:[?.!,]|$)",
    re.IGNORECASE,
)
_STOPWORD_TAIL = re.compile(
    r"\s+(?:please|now|today|tonight|tomorrow|right now)\s*$",
    re.IGNORECASE,
)


def _describe(code: int) -> str:
    return _WEATHER_CODES.get(code, "cloudy")


def _unit_symbol(units: str) -> str:
    return "°C" if units == "celsius" else "°F"


def _format_response(temp: float, code: int, units: str, location: str) -> str:
    description = _describe(code)
    symbol = _unit_symbol(units)
    location_clause = f" in {location}" if location else ""
    return f"It's {round(temp)}{symbol} and {description}{location_clause}."


def _extract_location(utterance: str) -> Optional[str]:
    if not utterance:
        return None
    match = _LOCATION_PATTERN.search(utterance)
    if not match:
        return None
    candidate = _STOPWORD_TAIL.sub("", match.group(1)).strip()
    return candidate or None


def _read_disk_cache() -> Optional[dict]:
    """Return the cached IP-location record, or None if missing/corrupted.
    The TTL check is the caller's job — this just reads."""
    try:
        with open(_CACHE_PATH) as f:
            data = json.load(f)
        if not all(k in data for k in ("lat", "lon", "city", "fetched_at")):
            return None
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        if not isinstance(e, FileNotFoundError):
            logger.warning("ip_cache_read_failed", error=str(e), error_type=type(e).__name__)
        return None


def _write_disk_cache(lat: float, lon: float, city: str) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(
                {"lat": lat, "lon": lon, "city": city, "fetched_at": time.time()},
                f,
            )
    except OSError as e:
        logger.warning("ip_cache_write_failed", error=str(e), error_type=type(e).__name__)


def _is_fresh(cache_record: dict) -> bool:
    age = time.time() - float(cache_record.get("fetched_at", 0))
    return age < _CACHE_TTL_SECONDS


async def _fetch_ip_location(client: httpx.AsyncClient) -> Optional[Tuple[float, float, str]]:
    """Hit ip-api.com. Returns None on any failure (caller decides fallback)."""
    response = await client.get(_IP_GEO_URL)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "success":
        return None
    lat = data.get("lat")
    lon = data.get("lon")
    city = data.get("city") or ""
    if lat is None or lon is None:
        return None
    return (float(lat), float(lon), str(city))


async def _ip_lookup(client: httpx.AsyncClient) -> Optional[Tuple[float, float, str]]:
    """Resolve the server's location via cache → network → stale-cache-fallback."""
    global _IP_LOCATION
    if _IP_LOCATION is not None:
        return _IP_LOCATION

    cached = _read_disk_cache()
    if cached is not None and _is_fresh(cached):
        _IP_LOCATION = (cached["lat"], cached["lon"], cached["city"])
        return _IP_LOCATION

    # Cache missing or expired — try the network.
    try:
        fresh = await _fetch_ip_location(client)
    except Exception as e:
        # Network failure: if we have ANY cached value (even stale), use it.
        # Stale location is always better than failing the turn.
        logger.warning("ip_lookup_failed", error=str(e), error_type=type(e).__name__)
        if cached is not None:
            _IP_LOCATION = (cached["lat"], cached["lon"], cached["city"])
            return _IP_LOCATION
        return None

    if fresh is None:
        # ip-api.com replied but the data wasn't usable — fall back to stale.
        if cached is not None:
            _IP_LOCATION = (cached["lat"], cached["lon"], cached["city"])
            return _IP_LOCATION
        return None

    lat, lon, city = fresh
    _write_disk_cache(lat, lon, city)
    _IP_LOCATION = (lat, lon, city)
    return _IP_LOCATION


async def _geocode(client: httpx.AsyncClient, query: str) -> Optional[Tuple[float, float, str]]:
    key = query.lower().strip()
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    response = await client.get(_GEOCODE_URL, params={"name": query, "count": 1})
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        _GEOCODE_CACHE[key] = None
        return None
    top = results[0]
    coords = (float(top["latitude"]), float(top["longitude"]), top.get("name") or query)
    _GEOCODE_CACHE[key] = coords
    return coords


async def _forecast(
    client: httpx.AsyncClient, latitude: float, longitude: float, units: str
) -> dict:
    response = await client.get(
        _FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": units,
        },
    )
    response.raise_for_status()
    return response.json()


async def handle(utterance: str, context: dict) -> str:
    try:
        async with httpx.AsyncClient(timeout=cfg.TIMEOUT_SECONDS) as client:
            extracted = _extract_location(utterance)
            location: Optional[Tuple[float, float, str]] = None
            if extracted:
                try:
                    location = await _geocode(client, extracted)
                except Exception as e:
                    logger.warning(
                        "weather_geocode_failed",
                        query=extracted,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
            if location is None:
                location = await _ip_lookup(client)
            if location is None:
                return _FRIENDLY_ERROR

            lat, lon, name = location
            data = await _forecast(client, lat, lon, cfg.UNITS)

        current = data["current"]
        return _format_response(
            temp=current["temperature_2m"],
            code=current["weather_code"],
            units=cfg.UNITS,
            location=name,
        )
    except httpx.TimeoutException as e:
        logger.warning("weather_timeout", error=str(e))
        return _FRIENDLY_ERROR
    except httpx.HTTPStatusError as e:
        logger.warning("weather_http_error", status_code=e.response.status_code, error=str(e))
        return _FRIENDLY_ERROR
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(
            "weather_response_malformed", error=str(e), error_type=type(e).__name__
        )
        return _FRIENDLY_ERROR
    except Exception as e:
        logger.error("weather_unexpected_error", error=str(e), error_type=type(e).__name__)
        return _FRIENDLY_ERROR


registry.register("weather_query", handle)
