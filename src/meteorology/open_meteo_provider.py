#!/usr/bin/env python3
"""Read-only Open-Meteo rainfall provider for the S6 weather adapter."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class WeatherProviderError(RuntimeError):
    """Raised when a live weather response cannot satisfy the S6 contract."""


def _finite_non_negative(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WeatherProviderError(f"{field} is not numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise WeatherProviderError(f"{field} must be finite and non-negative")
    return number


def _coordinate(value: Any, field: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WeatherProviderError(f"{field} is not numeric") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise WeatherProviderError(f"{field} must be in [{minimum}, {maximum}]")
    return number


def build_open_meteo_url(weather: dict[str, Any]) -> str:
    location = weather.get("location", {})
    api = weather.get("api", {})
    latitude = _coordinate(location.get("latitude"), "latitude", -90.0, 90.0)
    longitude = _coordinate(location.get("longitude"), "longitude", -180.0, 180.0)
    endpoint = str(api.get("endpoint", "https://api.open-meteo.com/v1/forecast"))
    if not endpoint.startswith("https://"):
        raise WeatherProviderError("Open-Meteo endpoint must use HTTPS")
    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "precipitation,rain,showers",
            "minutely_15": "precipitation",
            "forecast_minutely_15": 4,
            "precipitation_unit": "mm",
            "timezone": "auto",
        }
    )
    return f"{endpoint}?{query}"


def _parse_open_meteo_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("error"):
        raise WeatherProviderError(f"Open-Meteo error: {payload.get('reason', 'unknown')}")
    current = payload.get("current")
    minutely = payload.get("minutely_15")
    if not isinstance(current, dict) or not isinstance(minutely, dict):
        raise WeatherProviderError("Open-Meteo response lacks current or minutely_15 data")
    interval_seconds = _finite_non_negative(current.get("interval"), "current.interval")
    if interval_seconds <= 0.0:
        raise WeatherProviderError("current.interval must be greater than zero")
    current_precipitation_mm = _finite_non_negative(
        current.get("precipitation"), "current.precipitation"
    )
    forecast_values = minutely.get("precipitation")
    forecast_times = minutely.get("time")
    if not isinstance(forecast_values, list) or len(forecast_values) < 4:
        raise WeatherProviderError("minutely_15.precipitation requires at least four values")
    if not isinstance(forecast_times, list) or len(forecast_times) < 4:
        raise WeatherProviderError("minutely_15.time requires at least four values")
    increments = [
        _finite_non_negative(value, f"minutely_15.precipitation[{index}]")
        for index, value in enumerate(forecast_values[:4])
    ]
    rainfall_intensity = current_precipitation_mm * 3600.0 / interval_seconds
    canonical_payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "current": {
            "rainfall_intensity_mm_h": rainfall_intensity,
            "observation_time": current.get("time"),
            "source_interval_seconds": interval_seconds,
            "source_precipitation_mm": current_precipitation_mm,
        },
        "forecast": {
            "rainfall_15min_mm": sum(increments[:1]),
            "rainfall_30min_mm": sum(increments[:2]),
            "rainfall_60min_mm": sum(increments[:4]),
            "forecast_times": forecast_times[:4],
            "increments_15min_mm": increments,
        },
        "provider_metadata": {
            "returned_latitude": payload.get("latitude"),
            "returned_longitude": payload.get("longitude"),
            "elevation_m": payload.get("elevation"),
            "timezone": payload.get("timezone"),
            "utc_offset_seconds": payload.get("utc_offset_seconds"),
            "response_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        },
    }


def fetch_open_meteo_weather(
    weather: dict[str, Any],
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    api = weather.get("api", {})
    safety = weather.get("safety", {})
    if api.get("enabled") is not True:
        raise WeatherProviderError("live weather API is disabled")
    if safety.get("allow_live_weather_api") is not True:
        raise WeatherProviderError("safety.allow_live_weather_api must be true")
    if safety.get("allow_external_notification") is not False:
        raise WeatherProviderError("weather provider must not enable external notifications")
    if safety.get("allow_real_warning") is not False:
        raise WeatherProviderError("weather provider must not enable real warnings")
    timeout = _finite_non_negative(api.get("timeout_seconds", 8.0), "timeout_seconds")
    if timeout <= 0.0 or timeout > 30.0:
        raise WeatherProviderError("timeout_seconds must be in (0, 30]")
    url = build_open_meteo_url(weather)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "water-agent-system/phase2d-c14-weather-sandbox",
        },
        method="GET",
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except WeatherProviderError:
        raise
    except Exception as exc:
        raise WeatherProviderError(f"Open-Meteo request failed: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise WeatherProviderError("Open-Meteo response must be a JSON object")
    result = _parse_open_meteo_payload(payload)
    result["provider_metadata"]["request_url"] = url
    result["provider_metadata"]["timeout_seconds"] = timeout
    return result
