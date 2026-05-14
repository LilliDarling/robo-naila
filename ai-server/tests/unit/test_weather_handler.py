"""Unit tests for the weather_query action handler.

Resolution flow per turn:
  1. Try to extract a location from the utterance ("in Tokyo", "for Boston")
  2. If a location string was extracted, geocode it; on success use those coords
  3. Otherwise (no location extracted, or geocode failed), use the IP-detected
     server location — first checking the on-disk cache, then falling back to
     a network call to ip-api.com if the cache is missing or stale
  4. Call Open-Meteo forecast and format the reply

Each network call (IP geo, geocoding, forecast) is mocked. The handler always
takes ownership of the turn — errors surface as friendly strings, never raise.
"""

import inspect
import json
import os
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from agents.actions import weather_handler
from config import weather as cfg


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch, tmp_path):
    """Each test starts with empty in-memory caches and a fresh disk cache
    location, so prior runs don't leak fixture data across tests."""
    monkeypatch.setattr(weather_handler, "_IP_LOCATION", None)
    monkeypatch.setattr(weather_handler, "_GEOCODE_CACHE", {})
    # Point the on-disk cache at a per-test tmp file so we never touch the
    # real ``data/ip_location.json``.
    monkeypatch.setattr(
        weather_handler, "_CACHE_PATH", str(tmp_path / "ip_location.json")
    )


def _config(monkeypatch, **overrides):
    defaults = {"UNITS": "fahrenheit", "TIMEOUT_SECONDS": 5.0}
    defaults.update(overrides)
    for key, value in defaults.items():
        monkeypatch.setattr(cfg, key, value)


def _httpx_response(json_payload, status_code=200):
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = Mock(return_value=json_payload)
    resp.raise_for_status = Mock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=Mock(), response=resp
        )
    return resp


def _ip_api_payload(lat=42.36, lon=-71.06, city="Boston"):
    """The shape ip-api.com /json/ returns on success."""
    return {"status": "success", "lat": lat, "lon": lon, "city": city}


def _mock_client_with_routes(routes: dict):
    """Build a mock httpx AsyncClient whose .get returns the response paired
    with the matching URL substring (first match wins)."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    async def _get(url, *args, **kwargs):
        for substring, response in routes.items():
            if substring in url:
                return response
        raise AssertionError(
            f"unexpected URL {url!r} — routed substrings were {list(routes)}"
        )

    mock_client.get = AsyncMock(side_effect=_get)
    return mock_client


def _write_cache(path: str, lat=42.36, lon=-71.06, city="Boston", age_seconds=0):
    """Pre-populate the on-disk cache with a record N seconds old."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "lat": lat,
        "lon": lon,
        "city": city,
        "fetched_at": time.time() - age_seconds,
    }
    with open(path, "w") as f:
        json.dump(data, f)


class TestSignature:
    def test_handler_is_a_coroutine_function(self):
        assert inspect.iscoroutinefunction(weather_handler.handle)

    def test_handler_accepts_utterance_and_context(self):
        params = list(inspect.signature(weather_handler.handle).parameters)
        assert params[:2] == ["utterance", "context"]


class TestRegistration:
    def test_module_registers_on_import(self):
        from agents.actions import registry
        import importlib
        from agents.actions import weather_handler as wh

        importlib.reload(wh)
        assert registry.get("weather_query") is not None


class TestIpApiProvider:
    """The IP geo provider is ip-api.com (45 req/min, no daily cap, HTTP-only).
    Pin URL + response shape so we don't accidentally regress to ipapi.co or
    miss the slightly different field names ('lat' not 'latitude')."""

    @pytest.mark.asyncio
    async def test_ip_lookup_calls_ip_api_dot_com(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response(_ip_api_payload()),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 60.0, "weather_code": 2}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            await weather_handler.handle("weather", {})

        ip_calls = [
            c for c in mock_client.get.call_args_list if "ip-api.com" in c.args[0]
        ]
        assert len(ip_calls) == 1
        # Must NOT have called the old provider.
        legacy_calls = [
            c for c in mock_client.get.call_args_list if "ipapi.co" in c.args[0]
        ]
        assert legacy_calls == []

    @pytest.mark.asyncio
    async def test_ip_lookup_handles_status_field(self, monkeypatch):
        """ip-api.com signals failure via ``status: 'fail'`` not a 4xx HTTP
        status code. Treat ``fail`` as no-data."""
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response({"status": "fail", "message": "private range"}),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        # No location → friendly error rather than crashing on missing 'lat'.
        assert "couldn't" in result.lower() or "could not" in result.lower() \
            or "unable" in result.lower()


class TestUtterancePassThrough:
    @pytest.mark.asyncio
    async def test_uses_ip_location_when_no_place_in_utterance(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response(_ip_api_payload(city="Boston")),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 60.0, "weather_code": 2}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("what's the weather", {})

        assert "Boston" in result
        assert "60" in result

    @pytest.mark.asyncio
    async def test_ip_location_cached_in_memory_across_calls(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response(_ip_api_payload()),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 60.0, "weather_code": 2}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            await weather_handler.handle("weather", {})
            await weather_handler.handle("weather", {})
            await weather_handler.handle("weather", {})

        ip_calls = [
            c for c in mock_client.get.call_args_list if "ip-api.com" in c.args[0]
        ]
        assert len(ip_calls) == 1


class TestOnDiskCache:
    """The on-disk cache makes the handler robust to ip-api.com being down
    or rate-limited. It also means a fresh process start doesn't have to
    re-look-up the IP (which is essentially static for a home server)."""

    @pytest.mark.asyncio
    async def test_disk_cache_skips_network_when_fresh(self, monkeypatch):
        _config(monkeypatch)
        _write_cache(weather_handler._CACHE_PATH, lat=10.0, lon=20.0,
                     city="CachedCity", age_seconds=60)

        mock_client = _mock_client_with_routes({
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 50.0, "weather_code": 0}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        # No ip-api.com call — handler used the cache.
        ip_calls = [
            c for c in mock_client.get.call_args_list if "ip-api.com" in c.args[0]
        ]
        assert ip_calls == []
        assert "CachedCity" in result

    @pytest.mark.asyncio
    async def test_disk_cache_written_after_successful_lookup(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response(_ip_api_payload(
                lat=39.54, lon=-104.97, city="Highlands Ranch"
            )),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 60.0, "weather_code": 2}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            await weather_handler.handle("weather", {})

        assert os.path.exists(weather_handler._CACHE_PATH)
        with open(weather_handler._CACHE_PATH) as f:
            cached = json.load(f)
        assert cached["lat"] == pytest.approx(39.54)
        assert cached["lon"] == pytest.approx(-104.97)
        assert cached["city"] == "Highlands Ranch"
        assert "fetched_at" in cached

    @pytest.mark.asyncio
    async def test_disk_cache_expired_triggers_fresh_lookup(self, monkeypatch):
        _config(monkeypatch)
        # Write a cache file that's 31 days old (TTL is 30 days).
        _write_cache(
            weather_handler._CACHE_PATH,
            lat=10.0, lon=20.0, city="StaleCity",
            age_seconds=31 * 86400,
        )

        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response(_ip_api_payload(
                lat=39.54, lon=-104.97, city="FreshCity"
            )),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 60.0, "weather_code": 2}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        assert "FreshCity" in result
        # The fresh lookup must have rewritten the cache.
        with open(weather_handler._CACHE_PATH) as f:
            cached = json.load(f)
        assert cached["city"] == "FreshCity"

    @pytest.mark.asyncio
    async def test_disk_cache_falls_back_to_stale_on_network_failure(
        self, monkeypatch
    ):
        """Even with an expired cache, if the refresh attempt fails (429,
        timeout, no internet), use the stale value rather than failing the
        whole turn. Stale weather location is always better than no weather."""
        _config(monkeypatch)
        _write_cache(
            weather_handler._CACHE_PATH,
            lat=10.0, lon=20.0, city="StaleCity",
            age_seconds=31 * 86400,
        )

        # ip-api.com fails; forecast succeeds. We should still produce a reply
        # using the stale cached coords.
        async def _get(url, *args, **kwargs):
            if "ip-api.com" in url:
                raise httpx.TimeoutException("rate-limited or down")
            return _httpx_response({
                "current": {"temperature_2m": 50.0, "weather_code": 0}
            })

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        assert "StaleCity" in result

    @pytest.mark.asyncio
    async def test_corrupted_cache_triggers_fresh_lookup(self, monkeypatch):
        """A garbled cache file shouldn't crash the handler."""
        _config(monkeypatch)
        os.makedirs(os.path.dirname(weather_handler._CACHE_PATH), exist_ok=True)
        with open(weather_handler._CACHE_PATH, "w") as f:
            f.write("{ this is not json")

        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response(_ip_api_payload(city="FreshCity")),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 60.0, "weather_code": 2}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        assert "FreshCity" in result


class TestUtteranceLocationExtraction:
    @pytest.mark.asyncio
    async def test_extracts_in_phrase_and_geocodes(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "geocoding-api.open-meteo.com": _httpx_response({
                "results": [{"latitude": 35.68, "longitude": 139.69, "name": "Tokyo"}]
            }),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 70.0, "weather_code": 0}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("what's the weather in Tokyo", {})

        # Utterance had a location → IP lookup must be skipped.
        ip_calls = [
            c for c in mock_client.get.call_args_list if "ip-api.com" in c.args[0]
        ]
        assert ip_calls == []
        assert "Tokyo" in result

    @pytest.mark.asyncio
    async def test_extracts_for_phrase(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "geocoding-api.open-meteo.com": _httpx_response({
                "results": [{"latitude": 30.27, "longitude": -97.74, "name": "Austin"}]
            }),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 80.0, "weather_code": 0}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather for Austin please", {})

        geo_call = next(
            c for c in mock_client.get.call_args_list
            if "geocoding-api.open-meteo.com" in c.args[0]
        )
        params = geo_call.kwargs.get("params", {})
        assert "Austin" in params.get("name", "")
        assert "Austin" in result

    @pytest.mark.asyncio
    async def test_geocode_failure_falls_back_to_ip(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "geocoding-api.open-meteo.com": _httpx_response({"results": []}),
            "ip-api.com": _httpx_response(_ip_api_payload(city="Boston")),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 50.0, "weather_code": 2}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather in qwertyasdf", {})

        assert "Boston" in result

    @pytest.mark.asyncio
    async def test_geocode_results_cached_in_memory(self, monkeypatch):
        _config(monkeypatch)
        mock_client = _mock_client_with_routes({
            "geocoding-api.open-meteo.com": _httpx_response({
                "results": [{"latitude": 35.68, "longitude": 139.69, "name": "Tokyo"}]
            }),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 70.0, "weather_code": 0}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            await weather_handler.handle("weather in Tokyo", {})
            await weather_handler.handle("weather in Tokyo", {})

        geo_calls = [
            c for c in mock_client.get.call_args_list
            if "geocoding-api.open-meteo.com" in c.args[0]
        ]
        assert len(geo_calls) == 1


class TestForecastFormatting:
    @pytest.mark.asyncio
    async def test_celsius_units_passed_through(self, monkeypatch):
        _config(monkeypatch, UNITS="celsius")
        mock_client = _mock_client_with_routes({
            "ip-api.com": _httpx_response(_ip_api_payload()),
            "api.open-meteo.com/v1/forecast": _httpx_response({
                "current": {"temperature_2m": 17.0, "weather_code": 0}
            }),
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        forecast_call = next(
            c for c in mock_client.get.call_args_list
            if "api.open-meteo.com/v1/forecast" in c.args[0]
        )
        assert forecast_call.kwargs.get("params", {}).get("temperature_unit") == "celsius"
        assert "°C" in result

    @pytest.mark.asyncio
    async def test_weather_codes_map_to_human_descriptions(self, monkeypatch):
        _config(monkeypatch)
        cases = [
            (0, "clear"),
            (3, "overcast"),
            (61, "rain"),
            (71, "snow"),
            (95, "thunder"),
        ]
        for code, expected in cases:
            weather_handler._IP_LOCATION = None
            mock_client = _mock_client_with_routes({
                "ip-api.com": _httpx_response(_ip_api_payload()),
                "api.open-meteo.com/v1/forecast": _httpx_response({
                    "current": {"temperature_2m": 50.0, "weather_code": code}
                }),
            })
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await weather_handler.handle("weather", {})
            assert expected in result.lower(), (
                f"code {code} should mention {expected!r}, got {result!r}"
            )


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_ip_lookup_failure_with_no_cache_returns_friendly_error(
        self, monkeypatch
    ):
        _config(monkeypatch)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("no internet"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        assert "couldn't" in result.lower() or "could not" in result.lower() \
            or "unable" in result.lower()

    @pytest.mark.asyncio
    async def test_forecast_timeout_returns_friendly_error(self, monkeypatch):
        _config(monkeypatch)

        async def _get(url, *args, **kwargs):
            if "ip-api.com" in url:
                return _httpx_response(_ip_api_payload())
            raise httpx.TimeoutException("forecast slow")

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        assert "couldn't" in result.lower() or "could not" in result.lower() \
            or "unable" in result.lower()

    @pytest.mark.asyncio
    async def test_handler_never_raises(self, monkeypatch):
        _config(monkeypatch)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await weather_handler.handle("weather", {})

        assert isinstance(result, str)
        assert len(result) > 0
