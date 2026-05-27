"""Focused unit tests for low-coverage helpers.

Targets `models/results.py`, `models/location.py`, `models/config.py`
and `components/gvf.py` — files the existing suite barely touches
because they only run end-to-end. These tests exercise the smaller
methods, error paths, and side-effect-free branches directly, without
running a full SOLWEIG calculation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# models/results.py — SolweigResult.to_geotiff / compute_utci / compute_pet
# ---------------------------------------------------------------------------
from solweig.models.results import SolweigResult


def _make_simple_result(rows: int = 8, cols: int = 8) -> SolweigResult:
    """Minimal SolweigResult with deterministic Tmrt for assertion."""
    tmrt = np.full((rows, cols), 30.0, dtype=np.float32)
    shadow = np.ones((rows, cols), dtype=np.float32)
    return SolweigResult(tmrt=tmrt, shadow=shadow)


def test_solweig_result_dataclass_defaults():
    """Optional fields default to None; tmrt is required."""
    r = _make_simple_result()
    assert r.tmrt.shape == (8, 8)
    assert r.utci is None
    assert r.pet is None
    assert r.kdown is None and r.kup is None
    assert r.ldown is None and r.lup is None
    assert r.shadow is not None


def test_to_geotiff_writes_default_tmrt(tmp_path: Path):
    """Default ``outputs=None`` writes only tmrt."""
    r = _make_simple_result()
    r.to_geotiff(tmp_path, timestamp=datetime(2026, 5, 27, 12, 0))

    tmrt_dir = tmp_path / "tmrt"
    files = list(tmrt_dir.glob("*.tif"))
    assert len(files) == 1
    assert files[0].name == "tmrt_20260527_1200.tif"


def test_to_geotiff_multiple_outputs(tmp_path: Path):
    """Each requested output gets its own subdir and dated file."""
    r = _make_simple_result()
    r.utci = np.full((8, 8), 25.0, dtype=np.float32)
    r.to_geotiff(
        tmp_path,
        timestamp=datetime(2026, 5, 27, 14, 30),
        outputs=["tmrt", "utci"],
    )
    assert (tmp_path / "tmrt" / "tmrt_20260527_1430.tif").exists()
    assert (tmp_path / "utci" / "utci_20260527_1430.tif").exists()


def test_to_geotiff_unknown_output_logs_and_skips(tmp_path: Path, caplog):
    """Unknown output name → warning, no file."""
    import logging

    caplog.set_level(logging.WARNING)
    r = _make_simple_result()
    r.to_geotiff(tmp_path, outputs=["tmrt", "nonsense"])
    assert any("Unknown output 'nonsense'" in rec.message for rec in caplog.records)
    assert not (tmp_path / "nonsense").exists()


def test_to_geotiff_skips_none_output(tmp_path: Path, caplog):
    """Output field that is None → warning, no file."""
    import logging

    caplog.set_level(logging.WARNING)
    r = _make_simple_result()  # utci is None
    r.to_geotiff(tmp_path, outputs=["tmrt", "utci"])
    assert any("'utci' is None" in rec.message for rec in caplog.records)


def test_to_geotiff_default_timestamp(tmp_path: Path):
    """Default timestamp=None falls back to current time without crashing."""
    r = _make_simple_result()
    r.to_geotiff(tmp_path)
    files = list((tmp_path / "tmrt").glob("*.tif"))
    assert len(files) == 1


def test_compute_utci_requires_rh_when_float_ta():
    """Float ``ta`` without ``rh`` raises ValueError."""
    r = _make_simple_result()
    with pytest.raises(ValueError, match="rh is required"):
        r.compute_utci(25.0)


def test_compute_pet_requires_rh_when_float_ta():
    """Float ``ta`` without ``rh`` raises ValueError."""
    r = _make_simple_result()
    with pytest.raises(ValueError, match="rh is required"):
        r.compute_pet(25.0)


# ---------------------------------------------------------------------------
# models/location.py — from_epw / __post_init__ validation
# ---------------------------------------------------------------------------
from solweig.models.location import Location  # noqa: E402


def test_location_validates_latitude_range():
    with pytest.raises(ValueError, match="Latitude"):
        Location(latitude=95.0, longitude=0.0)
    with pytest.raises(ValueError, match="Latitude"):
        Location(latitude=-91.0, longitude=0.0)


def test_location_validates_longitude_range():
    with pytest.raises(ValueError, match="Longitude"):
        Location(latitude=0.0, longitude=181.0)
    with pytest.raises(ValueError, match="Longitude"):
        Location(latitude=0.0, longitude=-181.0)


def test_location_to_sun_position_dict_has_expected_keys():
    loc = Location(latitude=37.98, longitude=23.73, altitude=175, utc_offset=2)
    d = loc.to_sun_position_dict()
    assert d == {"latitude": 37.98, "longitude": 23.73, "altitude": 175}


def test_location_from_epw_parses_athens_header():
    """Athens EPW header has known coordinates and UTC offset."""
    epw = Path("demos/data/athens/athens_2023.epw")
    if not epw.exists():
        pytest.skip(f"Athens EPW fixture missing: {epw}")
    loc = Location.from_epw(epw)
    assert loc.latitude == pytest.approx(38.0)
    assert loc.longitude == pytest.approx(23.75)
    assert loc.utc_offset == pytest.approx(2.0)
    assert loc.altitude == pytest.approx(175.0)


def test_location_from_epw_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Location.from_epw(tmp_path / "missing.epw")


def test_location_from_epw_malformed_header_raises(tmp_path: Path):
    bad = tmp_path / "bad.epw"
    bad.write_text("NOT_A_LOCATION_LINE\n")
    with pytest.raises(ValueError, match="LOCATION"):
        Location.from_epw(bad)


# ---------------------------------------------------------------------------
# models/config.py — HumanParams / ModelConfig validation + save/load
# ---------------------------------------------------------------------------
from solweig.models.config import HumanParams, ModelConfig  # noqa: E402


def test_human_params_default_construction():
    h = HumanParams()
    assert h.posture == "standing"
    assert h.abs_k == pytest.approx(0.7)
    assert h.weight == pytest.approx(75.0)


@pytest.mark.parametrize(
    "field,bad_value,err_substr",
    [
        ({"posture": "lying"}, None, "Posture"),
        ({"abs_k": 0}, None, "abs_k"),
        ({"abs_k": 1.5}, None, "abs_k"),
        ({"abs_l": 0}, None, "abs_l"),
        ({"abs_l": 2.0}, None, "abs_l"),
        ({"age": -1}, None, "age"),
        ({"age": 200}, None, "age"),
        ({"weight": 0}, None, "weight"),
        ({"weight": -5}, None, "weight"),
        ({"height": 0.3}, None, "height"),
        ({"height": 3.0}, None, "height"),
        ({"sex": 3}, None, "sex"),
        ({"activity": -1}, None, "activity"),
        ({"clothing": -0.1}, None, "clothing"),
    ],
)
def test_human_params_validation_raises(field, bad_value, err_substr):
    with pytest.raises(ValueError, match=err_substr):
        HumanParams(**field)


def test_model_config_defaults_use_anisotropic_sky():
    cfg = ModelConfig.defaults()
    assert cfg.use_anisotropic_sky is True


@pytest.mark.parametrize(
    "kwargs,err_substr",
    [
        ({"max_shadow_distance_m": 0}, "max_shadow_distance_m"),
        ({"max_shadow_distance_m": -1}, "max_shadow_distance_m"),
        ({"tile_size": 100}, "tile_size"),
        ({"tile_workers": 0}, "tile_workers"),
        ({"tile_queue_depth": -1}, "tile_queue_depth"),
    ],
)
def test_model_config_validation_raises(kwargs, err_substr):
    with pytest.raises(ValueError, match=err_substr):
        ModelConfig(**kwargs)


def test_model_config_save_then_load_roundtrip(tmp_path: Path):
    out = tmp_path / "config.json"
    cfg = ModelConfig(
        use_anisotropic_sky=False,
        max_shadow_distance_m=750.0,
        tile_size=512,
        tile_workers=2,
        outputs=["tmrt", "utci"],
        human=HumanParams(posture="sitting", weight=80.0),
    )
    cfg.save(out)

    raw = json.loads(out.read_text())
    assert raw["use_anisotropic_sky"] is False
    assert raw["max_shadow_distance_m"] == 750.0
    assert raw["human"]["posture"] == "sitting"

    loaded = ModelConfig.load(out)
    assert loaded.use_anisotropic_sky is False
    assert loaded.max_shadow_distance_m == 750.0
    assert loaded.tile_size == 512
    assert loaded.tile_workers == 2
    assert loaded.outputs == ["tmrt", "utci"]
    assert loaded.human is not None
    assert loaded.human.posture == "sitting"
    assert loaded.human.weight == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# components/gvf.py — detect_building_mask
# ---------------------------------------------------------------------------
from solweig.components.gvf import detect_building_mask  # noqa: E402
from solweig.errors import InvalidSurfaceData  # noqa: E402


def test_building_mask_missing_dsm_raises():
    with pytest.raises(InvalidSurfaceData, match="DSM"):
        detect_building_mask(np.array([]), None, None, pixel_size=1.0)


def test_building_mask_zero_pixel_size_raises():
    dsm = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(InvalidSurfaceData, match="pixel_size"):
        detect_building_mask(dsm, None, None, pixel_size=0.0)


def test_building_mask_from_land_cover_marks_id_2():
    """Land cover ID 2 = building → mask value 0; everything else = 1."""
    dsm = np.zeros((4, 4), dtype=np.float32)
    lc = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    mask = detect_building_mask(dsm, lc, None, pixel_size=1.0)
    assert mask.dtype == np.float32
    assert mask.shape == (4, 4)
    # ID 2 → 0 (building); everything else → 1 (ground)
    assert mask[0, 0] == 1.0 and mask[0, 2] == 0.0
    assert mask[2, 2] == 1.0


def test_building_mask_land_cover_shape_mismatch_raises():
    dsm = np.zeros((4, 4), dtype=np.float32)
    lc_wrong = np.zeros((5, 5), dtype=np.int32)
    with pytest.raises(InvalidSurfaceData, match="land_cover"):
        detect_building_mask(dsm, lc_wrong, None, pixel_size=1.0)


def test_building_mask_wall_height_shape_mismatch_raises():
    dsm = np.zeros((4, 4), dtype=np.float32)
    wall_wrong = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(InvalidSurfaceData, match="wall_height"):
        detect_building_mask(dsm, None, wall_wrong, pixel_size=1.0)


def test_building_mask_from_wall_height_marks_buildings():
    """Wall pixels + elevated rooftops → building (0); distant ground → 1.

    Uses pixel_size=25 m so the ~25 m dilation radius stays ≈1 pixel and the
    test grid can fit a clear "far-away" pixel that should remain ground.
    """
    dsm = np.zeros((30, 30), dtype=np.float32)
    dsm[2:5, 2:5] = 10.0  # rooftop 10 m up
    wall = np.zeros_like(dsm)
    wall[2, 2] = 10.0  # one wall pixel marks the building edge

    mask = detect_building_mask(dsm, None, wall, pixel_size=25.0)
    assert mask.dtype == np.float32
    # Rooftop pixel should be 0 (building)
    assert mask[3, 3] == 0.0
    # Pixel far outside the dilation radius and at ground level should be 1
    assert mask[29, 29] == 1.0


def test_building_mask_no_inputs_returns_all_ground():
    """When neither land_cover nor wall_height is provided → all 1.0."""
    dsm = np.zeros((6, 6), dtype=np.float32)
    mask = detect_building_mask(dsm, None, None, pixel_size=1.0)
    assert mask.dtype == np.float32
    assert np.all(mask == 1.0)


# ---------------------------------------------------------------------------
# io_epw.py — pure-Python DataFrame stand-in
# ---------------------------------------------------------------------------
from solweig.io_epw import read_epw  # noqa: E402


@pytest.fixture(scope="module")
def athens_epw_df():
    """Load Athens EPW once for the whole module of tests."""
    path = Path("demos/data/athens/athens_2023.epw")
    if not path.exists():
        pytest.skip(f"Athens EPW fixture missing: {path}")
    df, meta = read_epw(path)
    return df, meta


def test_read_epw_returns_metadata(athens_epw_df):
    _, meta = athens_epw_df
    assert meta["latitude"] == pytest.approx(38.0)
    assert meta["longitude"] == pytest.approx(23.75)
    assert meta["tz_offset"] == pytest.approx(2.0)
    assert meta["elevation"] == pytest.approx(175.0)


def test_read_epw_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_epw(tmp_path / "nope.epw")


def test_epw_dataframe_basic_shape_and_columns(athens_epw_df):
    df, _ = athens_epw_df
    assert len(df) == 8760
    cols = df.columns
    assert "temp_air" in cols
    assert "ghi" in cols
    assert "wind_speed" in cols
    assert not df.empty


def test_epw_dataframe_column_access_and_extrema(athens_epw_df):
    df, _ = athens_epw_df
    temps = df["temp_air"]
    assert len(temps) == 8760
    # Athens annual temperatures should fall within a sane range
    assert -10 < temps.min() < 40
    assert temps.max() > temps.min()


def test_epw_dataframe_iloc_returns_row_with_get(athens_epw_df):
    df, _ = athens_epw_df
    row = df.iloc[100]
    assert row["temp_air"] == row.get("temp_air")
    # Unknown key with default
    assert row.get("does_not_exist", default=-1.0) == -1.0


def test_epw_dataframe_iterrows_yields_timestamps(athens_epw_df):
    df, _ = athens_epw_df
    n = 0
    for ts, row in df.iterrows():
        assert hasattr(ts, "year")
        assert "temp_air" in row._data
        n += 1
        if n >= 5:
            break
    assert n == 5


def test_epw_dataframe_index_hour_filter(athens_epw_df):
    df, _ = athens_epw_df
    # Filter to midday (hour == 12)
    is_noon = df.index.hour == 12
    noon_df = df[is_noon]
    assert len(noon_df) == 365  # one row per day at noon
    assert not noon_df.empty


# ---------------------------------------------------------------------------
# Weather.from_epw — exercises the parse/filter/build path
# ---------------------------------------------------------------------------
from solweig.models.weather import Location as _Loc  # noqa: E402, F401
from solweig.models.weather import Weather  # noqa: E402


def test_weather_from_epw_full_year_with_explicit_range():
    epw = Path("demos/data/athens/athens_2023.epw")
    if not epw.exists():
        pytest.skip(f"Athens EPW fixture missing: {epw}")
    weathers = Weather.from_epw(epw, start="2023-01-01", end="2023-12-31")
    # EPW hour-24 of Dec 31 spills into Jan 1, so the inclusive end-of-year
    # range yields 8759, not the raw 8760 rows in the file.
    assert 8700 <= len(weathers) <= 8760
    w0 = weathers[0]
    assert w0.datetime.year == 2023
    # Sane plausibility checks for Athens
    assert -10 < w0.ta < 40
    assert 0 <= w0.rh <= 100


def test_weather_from_epw_date_range_filter():
    epw = Path("demos/data/athens/athens_2023.epw")
    if not epw.exists():
        pytest.skip(f"Athens EPW fixture missing: {epw}")
    weathers = Weather.from_epw(epw, start="2023-07-15", end="2023-07-15")
    # 24 hours on one day
    assert len(weathers) == 24
    assert all(w.datetime.month == 7 and w.datetime.day == 15 for w in weathers)


def test_weather_from_epw_tmy_format_filter():
    """Year-agnostic 'MM-DD' format works for TMY files."""
    epw = Path("demos/data/athens/athens_2023.epw")
    if not epw.exists():
        pytest.skip(f"Athens EPW fixture missing: {epw}")
    weathers = Weather.from_epw(epw, start="07-15", end="07-15")
    assert len(weathers) == 24


def test_weather_from_epw_hours_filter():
    """Restrict to a few hours per day."""
    epw = Path("demos/data/athens/athens_2023.epw")
    if not epw.exists():
        pytest.skip(f"Athens EPW fixture missing: {epw}")
    weathers = Weather.from_epw(
        epw,
        start="2023-07-15",
        end="2023-07-15",
        hours=[6, 12, 18],
    )
    assert len(weathers) == 3
    assert sorted(w.datetime.hour for w in weathers) == [6, 12, 18]


def test_weather_from_epw_out_of_range_raises():
    epw = Path("demos/data/athens/athens_2023.epw")
    if not epw.exists():
        pytest.skip(f"Athens EPW fixture missing: {epw}")
    with pytest.raises(ValueError, match="not found in EPW"):
        Weather.from_epw(epw, start="2099-01-01", end="2099-01-02")


def test_weather_from_epw_bad_date_format_raises():
    """Date with neither 2 nor 3 dash-separated parts triggers the helpful
    'Cannot parse date' branch."""
    epw = Path("demos/data/athens/athens_2023.epw")
    if not epw.exists():
        pytest.skip(f"Athens EPW fixture missing: {epw}")
    with pytest.raises(ValueError, match="Cannot parse date"):
        Weather.from_epw(epw, start="not_a_date_at_all")


def test_epw_dataframe_boolean_array_and_or():
    """Exercise the lightweight _BooleanArray operators."""
    from solweig.io_epw import _BooleanArray

    a = _BooleanArray([True, False, True, False])
    b = _BooleanArray([True, True, False, False])
    assert list(a & b) == [True, False, False, False]
    assert list(a | b) == [True, True, True, False]
    # Mixed with iterable
    assert list(a & [False, False, True, True]) == [False, False, True, False]
    assert a.any() is True
    assert a.all() is False
    assert _BooleanArray([True, True]).all() is True
