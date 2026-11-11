"""Input mistakes that used to pass without a word.

- An unknown name in ``outputs`` produced folders of all-NaN GeoTIFFs.
- A weather series reused with a corrected ``Location`` kept the sun
  positions computed for the first one.
- Missing EPW values were replaced with defaults silently.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather, calculate
from solweig.errors import ConfigurationError
from solweig.models import weather as weather_module
from test_epw_missing_codes import _row, _write_epw

SHAPE = (12, 12)


def _surface():
    return SurfaceData(dsm=np.zeros(SHAPE, dtype=np.float32), pixel_size=1.0, svf=make_mock_svf(SHAPE))


def _weather():
    return Weather(datetime=datetime(2024, 7, 15, 9, 0), ta=25.0, rh=50.0, global_rad=500.0)


# ── outputs ────────────────────────────────────────────────────────────────


def test_unknown_output_name_is_rejected_before_anything_is_written(tmp_path):
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    with pytest.raises(ConfigurationError) as excinfo:
        calculate(_surface(), [_weather()], location, output_dir=tmp_path, outputs=["tmrt", "tmtr"])
    assert excinfo.value.parameter == "outputs"
    assert "tmtr" in str(excinfo.value)
    assert not any(tmp_path.iterdir())


def test_every_documented_output_name_is_accepted(tmp_path):
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    names = ["tmrt", "shadow", "kdown", "kup", "ldown", "lup", "utci", "pet"]
    calculate(_surface(), [_weather()], location, output_dir=tmp_path, outputs=names, use_anisotropic_sky=False)
    for name in names:
        assert list((tmp_path / name).glob("*.tif")), name


# ── derived weather follows the location ───────────────────────────────────


def test_derived_values_are_recomputed_for_a_different_location():
    weather = _weather()
    utc0 = Location(latitude=57.7, longitude=12.0, utc_offset=0)
    utc2 = Location(latitude=57.7, longitude=12.0, utc_offset=2)

    assert weather.needs_derived(utc0)
    weather.compute_derived(utc0)
    altitude_utc0 = weather.sun_altitude
    assert not weather.needs_derived(utc0)

    assert weather.needs_derived(utc2)
    weather.compute_derived(utc2)
    # 09:00 at UTC+2 is two hours earlier in solar time than 09:00 at UTC+0.
    assert weather.sun_altitude < altitude_utc0 - 5.0
    assert not weather.needs_derived(utc2)


def test_reusing_a_series_with_a_corrected_utc_offset_moves_the_shadows(tmp_path):
    dsm = np.zeros((30, 30), dtype=np.float32)
    dsm[12:18, 12:18] = 10.0
    surface = SurfaceData(dsm=dsm, pixel_size=1.0, svf=make_mock_svf(dsm.shape))
    series = [_weather()]

    wrong = calculate(surface, series, Location(latitude=57.7, longitude=12.0, utc_offset=0), output_dir=tmp_path / "a")
    right = calculate(surface, series, Location(latitude=57.7, longitude=12.0, utc_offset=2), output_dir=tmp_path / "b")
    fresh = calculate(
        surface, [_weather()], Location(latitude=57.7, longitude=12.0, utc_offset=2), output_dir=tmp_path / "c"
    )
    np.testing.assert_allclose(right.tmrt_mean, fresh.tmrt_mean, atol=1e-4, equal_nan=True)
    assert not np.allclose(right.tmrt_mean, wrong.tmrt_mean, atol=0.5, equal_nan=True)


# ── EPW gaps ───────────────────────────────────────────────────────────────


@pytest.fixture
def warnings(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(weather_module.logger, "warning", messages.append)
    return messages


def test_missing_epw_values_are_reported(tmp_path, warnings):
    path = _write_epw(tmp_path / "gaps.epw", [_row(1, ta="99.9"), _row(2, ws="999"), _row(3)])
    series = Weather.from_epw(str(path))
    assert series[0].ta == pytest.approx(20.0)
    assert len(warnings) == 1
    assert "air temperature: 1" in warnings[0]
    assert "wind speed: 1" in warnings[0]


def test_complete_epw_is_silent(tmp_path, warnings):
    path = _write_epw(tmp_path / "ok.epw", [_row(1), _row(2)])
    Weather.from_epw(str(path))
    assert warnings == []
