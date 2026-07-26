from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.meteorology.compute_weather_correction import compute_weather_correction
from src.meteorology.open_meteo_provider import (
    WeatherProviderError,
    build_open_meteo_url,
    fetch_open_meteo_weather,
)


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def weather_config() -> dict:
    return {
        "provider": "open_meteo",
        "data_mode": "live_api_sandbox",
        "location_name": "test",
        "location": {"latitude": 31.2, "longitude": 121.5},
        "api": {
            "enabled": True,
            "endpoint": "https://api.open-meteo.com/v1/forecast",
            "timeout_seconds": 5,
            "fallback_on_error": True,
        },
        "safety": {
            "allow_live_weather_api": True,
            "allow_external_notification": False,
            "allow_real_warning": False,
        },
        "fallback_current": {
            "rainfall_intensity_mm_h": 1.0,
            "observation_time": "fallback",
        },
        "fallback_forecast": {
            "rainfall_15min_mm": 0.1,
            "rainfall_30min_mm": 0.2,
            "rainfall_60min_mm": 0.4,
        },
    }


def api_payload() -> dict:
    return {
        "latitude": 31.2,
        "longitude": 121.5,
        "elevation": 4.0,
        "timezone": "Asia/Shanghai",
        "utc_offset_seconds": 28800,
        "current": {
            "time": "2026-07-26T15:00",
            "interval": 900,
            "precipitation": 0.5,
            "rain": 0.5,
            "showers": 0.0,
        },
        "minutely_15": {
            "time": [
                "2026-07-26T15:00",
                "2026-07-26T15:15",
                "2026-07-26T15:30",
                "2026-07-26T15:45",
            ],
            "precipitation": [0.2, 0.3, 0.4, 0.5],
        },
    }


def test_open_meteo_url_is_read_only_and_requests_required_fields() -> None:
    url = build_open_meteo_url(weather_config())
    assert url.startswith("https://api.open-meteo.com/v1/forecast?")
    assert "current=precipitation%2Crain%2Cshowers" in url
    assert "minutely_15=precipitation" in url
    assert "forecast_minutely_15=4" in url
    assert "apikey" not in url


def test_provider_converts_interval_precipitation_and_accumulates_forecast() -> None:
    result = fetch_open_meteo_weather(
        weather_config(), opener=lambda request, timeout: FakeResponse(api_payload())
    )
    assert result["current"]["rainfall_intensity_mm_h"] == pytest.approx(2.0)
    assert result["forecast"]["rainfall_15min_mm"] == pytest.approx(0.2)
    assert result["forecast"]["rainfall_30min_mm"] == pytest.approx(0.5)
    assert result["forecast"]["rainfall_60min_mm"] == pytest.approx(1.4)
    assert len(result["provider_metadata"]["response_sha256"]) == 64


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("api", "enabled", False),
        ("safety", "allow_live_weather_api", False),
        ("safety", "allow_external_notification", True),
        ("safety", "allow_real_warning", True),
    ],
)
def test_unsafe_or_disabled_live_requests_are_rejected(
    section: str, key: str, value: object
) -> None:
    config = weather_config()
    config[section][key] = value
    with pytest.raises(WeatherProviderError):
        fetch_open_meteo_weather(
            config, opener=lambda request, timeout: FakeResponse(api_payload())
        )


def test_invalid_coordinate_is_rejected_before_network() -> None:
    config = weather_config()
    config["location"]["latitude"] = 100.0
    with pytest.raises(WeatherProviderError, match="latitude"):
        build_open_meteo_url(config)


def test_compute_weather_uses_live_values_without_notifications(
    tmp_path: Path,
) -> None:
    config = weather_config()
    config_path = tmp_path / "weather.yaml"
    config_path.write_text(
        yaml.safe_dump({"weather": config}, sort_keys=False), encoding="utf-8"
    )
    live = fetch_open_meteo_weather(
        config, opener=lambda request, timeout: FakeResponse(api_payload())
    )
    result = compute_weather_correction(
        config_path, tmp_path / "runtime", weather_fetcher=lambda weather: live
    )
    assert result["api_status"] == "success"
    assert result["current_rainfall_intensity_mm_h"] == pytest.approx(2.0)
    assert result["weather_correction_factor"] == pytest.approx(1.0)
    assert result["external_notification_allowed"] is False
    assert result["real_warning_allowed"] is False


def test_compute_weather_falls_back_without_claiming_live_success(
    tmp_path: Path,
) -> None:
    config = weather_config()
    config_path = tmp_path / "weather.yaml"
    config_path.write_text(
        yaml.safe_dump({"weather": config}, sort_keys=False), encoding="utf-8"
    )

    def fail(_weather):
        raise WeatherProviderError("network unavailable")

    result = compute_weather_correction(
        config_path, tmp_path / "runtime", weather_fetcher=fail
    )
    assert result["api_status"] == "fallback"
    assert result["current_rainfall_intensity_mm_h"] == pytest.approx(1.0)
    assert result["api_error"] == "network unavailable"
    assert "fallback" in result["weather_data_note"]


def test_fallback_can_be_disabled(tmp_path: Path) -> None:
    config = deepcopy(weather_config())
    config["api"]["fallback_on_error"] = False
    config_path = tmp_path / "weather.yaml"
    config_path.write_text(
        yaml.safe_dump({"weather": config}, sort_keys=False), encoding="utf-8"
    )

    def fail(_weather):
        raise WeatherProviderError("network unavailable")

    with pytest.raises(WeatherProviderError):
        compute_weather_correction(
            config_path, tmp_path / "runtime", weather_fetcher=fail
        )
