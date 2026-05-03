"""Weather action handler configuration.

Location is auto-detected at runtime — IP geolocation by default, overridden
per turn when the user says "in <place>" / "for <place>" / "at <place>". Only
presentation (units) and reliability (timeout) are configurable.
"""

import os


UNITS = os.getenv("NAILA_WEATHER_UNITS", "fahrenheit")
TIMEOUT_SECONDS = float(os.getenv("NAILA_WEATHER_TIMEOUT_SECONDS", "5.0"))
