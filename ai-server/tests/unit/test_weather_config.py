"""Tests for weather configuration defaults and env overrides.

Lat/lon/location are auto-detected at runtime — no config knobs for those.
The only config left is presentation (units) and reliability (timeout).
"""

import importlib

import pytest


@pytest.fixture
def reload_weather_config(monkeypatch):
    def _reload(env: dict | None = None):
        for key in ("NAILA_WEATHER_UNITS", "NAILA_WEATHER_TIMEOUT_SECONDS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in (env or {}).items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        import config.weather as mod

        return importlib.reload(mod)

    return _reload


class TestWeatherConfigDefaults:
    def test_units_default_is_fahrenheit(self, reload_weather_config):
        mod = reload_weather_config()
        assert mod.UNITS == "fahrenheit"

    def test_timeout_default_is_5_seconds(self, reload_weather_config):
        mod = reload_weather_config()
        assert mod.TIMEOUT_SECONDS == 5.0


class TestWeatherConfigOverrides:
    def test_units_pass_through(self, reload_weather_config):
        mod = reload_weather_config({"NAILA_WEATHER_UNITS": "celsius"})
        assert mod.UNITS == "celsius"

    def test_timeout_parsed_as_float(self, reload_weather_config):
        mod = reload_weather_config({"NAILA_WEATHER_TIMEOUT_SECONDS": "2.5"})
        assert mod.TIMEOUT_SECONDS == pytest.approx(2.5)


class TestWeatherConfigDoesNotExposeLocation:
    """Location knobs are gone — the handler auto-detects via IP and parses
    utterances. Pinning this so they don't sneak back in."""

    def test_no_latitude_attribute(self, reload_weather_config):
        mod = reload_weather_config()
        assert not hasattr(mod, "LATITUDE")

    def test_no_longitude_attribute(self, reload_weather_config):
        mod = reload_weather_config()
        assert not hasattr(mod, "LONGITUDE")

    def test_no_location_name_attribute(self, reload_weather_config):
        mod = reload_weather_config()
        assert not hasattr(mod, "LOCATION_NAME")
