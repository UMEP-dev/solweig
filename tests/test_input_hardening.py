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


# ── materials ──────────────────────────────────────────────────────────────


def _kup_with(surface, materials, out_dir):
    from tests.raster_helpers import read_band

    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    calculate(
        surface, [_weather()], location, output_dir=out_dir, outputs=["kup"], materials=materials,
        use_anisotropic_sky=False,
    )  # fmt: skip
    (tif,) = (out_dir / "kup").glob("*.tif")
    return read_band(tif)


def test_materials_edited_in_place_are_picked_up_on_the_same_surface(tmp_path):
    """A changed material value must reach the cached land-cover property grids.

    The cache is keyed on the content of the materials (it was keyed on
    ``id(materials)``, which can be reused after a free). This test passed
    before that change as well; it guards the behaviour, not that bug.
    """
    from solweig import load_params

    def grass_surface():
        return SurfaceData(
            dsm=np.zeros(SHAPE, dtype=np.float32),
            pixel_size=1.0,
            svf=make_mock_svf(SHAPE),
            land_cover=np.full(SHAPE, 5, dtype=np.uint8),
        )

    surface = grass_surface()
    materials = load_params()
    before = _kup_with(surface, materials, tmp_path / "a")

    materials.Albedo.Effective.Value.Grass_unmanaged = 0.5
    after = _kup_with(surface, materials, tmp_path / "b")
    fresh = _kup_with(grass_surface(), materials, tmp_path / "c")

    assert after.mean() > before.mean() * 2.0
    np.testing.assert_array_equal(after, fresh)


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


# ── layers in another coordinate system ────────────────────────────────────


def test_layer_in_a_different_crs_is_reported(monkeypatch):
    """Layers are aligned by coordinates without reprojection, so a CRS mismatch misplaces them."""
    pyproj = pytest.importorskip("pyproj")
    from solweig.models import surface_loading

    monkeypatch.setattr(surface_loading, "GDAL_ENV", False)
    messages: list[str] = []
    monkeypatch.setattr(surface_loading.logger, "warning", messages.append)

    sweref99_tm = pyproj.CRS.from_epsg(3006).to_wkt()
    sweref99_1200 = pyproj.CRS.from_epsg(3007).to_wkt()
    same_crs_other_text = pyproj.CRS.from_epsg(3006).to_wkt(version="WKT1_GDAL")

    surface_loading.warn_if_crs_differs("cdsm", sweref99_tm, sweref99_tm)
    surface_loading.warn_if_crs_differs("cdsm", same_crs_other_text, sweref99_tm)
    surface_loading.warn_if_crs_differs("cdsm", None, sweref99_tm)
    assert messages == []

    surface_loading.warn_if_crs_differs("cdsm", sweref99_1200, sweref99_tm)
    assert len(messages) == 1
    assert "cdsm" in messages[0] and "different coordinate system" in messages[0]


# ── ground view factor height ──────────────────────────────────────────────


def _walled_tmrt(tmp_path, name, human):
    from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls

    dsm = np.zeros((40, 40), dtype=np.float32)
    dsm[15:25, 15:25] = 12.0
    walls = findwalls(dsm, 1.0)
    surface = SurfaceData(
        dsm=dsm,
        pixel_size=1.0,
        svf=make_mock_svf(dsm.shape),
        wall_height=walls,
        wall_aspect=filter1Goodwin_as_aspect_v3(walls, 1.0, dsm),
    )
    weather = Weather(datetime=datetime(2024, 7, 15, 13, 0), ta=25.0, rh=50.0, global_rad=700.0)
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    summary = calculate(
        surface, [weather], location, output_dir=tmp_path / name, human=human, use_anisotropic_sky=False
    )
    return summary.tmrt_mean


def test_posture_height_is_the_centre_of_gravity():
    from solweig import HumanParams

    assert HumanParams(posture="standing").posture_height == pytest.approx(1.1)
    assert HumanParams(posture="sitting").posture_height == pytest.approx(0.75)
    assert HumanParams(height=1.95).posture_height == pytest.approx(1.1)


def test_stature_does_not_change_tmrt(tmp_path):
    """The ground view factor is evaluated at the posture height, as in UMEP.

    Up to 0.1.0b99 it was given the stature, so a taller person changed Tmrt
    beside walls (first/second search heights 2 and 35 px at 1.75 m against
    1 and 22 px at the standing posture height).
    """
    from solweig import HumanParams

    short = _walled_tmrt(tmp_path, "short", HumanParams(height=1.50))
    tall = _walled_tmrt(tmp_path, "tall", HumanParams(height=2.00))
    np.testing.assert_array_equal(short, tall)


# ── anisotropic sky falling back to isotropic ──────────────────────────────


def test_default_anisotropic_without_shadow_matrices_is_recorded_as_isotropic(tmp_path, monkeypatch):
    """The sky model is anisotropic by default but needs shadow matrices."""
    import json

    from solweig import timeseries

    messages: list[str] = []
    monkeypatch.setattr(timeseries.logger, "warning", messages.append)
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)

    calculate(_surface(), [_weather()], location, output_dir=tmp_path)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["parameters"]["use_anisotropic_sky"] is False
    assert any("isotropic sky" in m for m in messages)


def test_explicit_isotropic_does_not_warn(tmp_path, monkeypatch):
    from solweig import timeseries

    messages: list[str] = []
    monkeypatch.setattr(timeseries.logger, "warning", messages.append)
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)

    calculate(_surface(), [_weather()], location, output_dir=tmp_path, use_anisotropic_sky=False)
    assert not any("isotropic sky" in m for m in messages)
