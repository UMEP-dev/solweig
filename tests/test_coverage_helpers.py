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


# ---------------------------------------------------------------------------
# Weather.from_umep_met — synthetic UMEP/SUEWS met file
# ---------------------------------------------------------------------------
_UMEP_HEADER = "%iy id it imin Q* QH QE Qs Qf wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd\n"


def _make_umep_row(
    year, doy, hour, minute, ta=20.0, rh=50.0, kdn=100.0, wind=2.0, press_kpa=101.3, kdir=80.0, kdiff=20.0
):
    """Build one UMEP met row (24 space-separated fields)."""
    return (
        f"{year} {doy} {hour} {minute} -999 -999 -999 -999 -999 "
        f"{wind} {rh} {ta} {press_kpa} 0 {kdn} 0 -999 0 0 0 0 {kdiff} {kdir} 180\n"
    )


def _write_umep_file(path: Path, rows: list[str]) -> None:
    path.write_text(_UMEP_HEADER + "".join(rows))


def test_umep_met_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Weather.from_umep_met(tmp_path / "missing.txt")


def test_umep_met_parses_hourly_rows(tmp_path):
    p = tmp_path / "met.txt"
    rows = [_make_umep_row(2024, 1, h, 0, ta=10.0 + h, rh=70.0, kdn=h * 50.0) for h in range(6)]
    _write_umep_file(p, rows)
    weathers = Weather.from_umep_met(p)
    assert len(weathers) == 6
    assert weathers[0].datetime == datetime(2024, 1, 1, 0, 0)
    assert weathers[5].ta == pytest.approx(15.0)
    # pressure: 101.3 kPa → 1013 hPa
    assert weathers[0].pressure == pytest.approx(1013.0)


def test_umep_met_resample_hourly_filters_subhour_rows(tmp_path):
    p = tmp_path / "met.txt"
    rows = []
    for h in range(2):
        for m in (0, 10, 20, 30, 40, 50):
            rows.append(_make_umep_row(2024, 1, h, m, ta=10.0))
    _write_umep_file(p, rows)
    hourly = Weather.from_umep_met(p, resample_hourly=True)
    assert len(hourly) == 2  # 2 on-the-hour rows
    sub = Weather.from_umep_met(p, resample_hourly=False)
    assert len(sub) == 12


def test_umep_met_skips_missing_data_rows(tmp_path):
    """Rows with ta=-999 / rh=-999 / kdn=-999 are filtered out."""
    p = tmp_path / "met.txt"
    rows = [
        _make_umep_row(2024, 1, 0, 0, ta=10.0, rh=50.0, kdn=100.0),
        _make_umep_row(2024, 1, 1, 0, ta=-999, rh=50.0, kdn=100.0),
        _make_umep_row(2024, 1, 2, 0, ta=10.0, rh=-999, kdn=100.0),
        _make_umep_row(2024, 1, 3, 0, ta=10.0, rh=50.0, kdn=-999),
        _make_umep_row(2024, 1, 4, 0, ta=12.0, rh=55.0, kdn=200.0),
    ]
    _write_umep_file(p, rows)
    weathers = Weather.from_umep_met(p)
    assert len(weathers) == 2
    assert weathers[0].ta == pytest.approx(10.0)
    assert weathers[1].ta == pytest.approx(12.0)


def test_umep_met_no_valid_rows_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text(_UMEP_HEADER + "# only comments\n")
    with pytest.raises(ValueError, match="No valid data rows"):
        Weather.from_umep_met(p)


def test_umep_met_concatenates_multiple_files(tmp_path):
    p1 = tmp_path / "may.txt"
    p2 = tmp_path / "jun.txt"
    _write_umep_file(p1, [_make_umep_row(2024, 121, 12, 0)])  # day 121 = May 1
    _write_umep_file(p2, [_make_umep_row(2024, 152, 12, 0)])  # day 152 = Jun 1
    weathers = Weather.from_umep_met([p1, p2])
    assert len(weathers) == 2
    # Sorted by datetime
    assert weathers[0].datetime < weathers[1].datetime


def test_umep_met_date_range_filter(tmp_path):
    p = tmp_path / "met.txt"
    rows = [_make_umep_row(2024, d, 12, 0) for d in range(1, 11)]  # 10 days
    _write_umep_file(p, rows)
    weathers = Weather.from_umep_met(p, start="2024-01-03", end="2024-01-05")
    assert len(weathers) == 3
    assert weathers[0].datetime == datetime(2024, 1, 3, 12, 0)
    assert weathers[-1].datetime == datetime(2024, 1, 5, 12, 0)


def test_umep_met_handles_negative_pressure_as_default(tmp_path):
    """press_kpa=-999 → falls back to standard 1013.25 hPa."""
    p = tmp_path / "met.txt"
    _write_umep_file(p, [_make_umep_row(2024, 1, 0, 0, press_kpa=-999)])
    weathers = Weather.from_umep_met(p)
    assert weathers[0].pressure == pytest.approx(1013.25)


def test_umep_met_detects_subhourly_timestep(tmp_path):
    """When resample_hourly=False, timestep_minutes is inferred from row gap."""
    p = tmp_path / "met.txt"
    rows = [_make_umep_row(2024, 1, 0, m) for m in (0, 10, 20)]
    _write_umep_file(p, rows)
    weathers = Weather.from_umep_met(p, resample_hourly=False)
    assert len(weathers) == 3
    assert weathers[0].timestep_minutes == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# components/svf_resolution.py — precomputed branch + missing-data error
# ---------------------------------------------------------------------------
from solweig.components.svf_resolution import (  # noqa: E402
    adjust_svfbuveg_with_psi,
    resolve_svf,
)
from solweig.errors import MissingPrecomputedData  # noqa: E402


def _make_svf_arrays(shape=(4, 4)):
    """Build a complete SvfArrays with deterministic uniform values."""
    from solweig.models.precomputed import SvfArrays

    ones = np.full(shape, 0.8, dtype=np.float32)
    return SvfArrays(
        svf=ones,
        svf_north=ones,
        svf_east=ones,
        svf_south=ones,
        svf_west=ones,
        svf_veg=ones,
        svf_veg_north=ones,
        svf_veg_east=ones,
        svf_veg_south=ones,
        svf_veg_west=ones,
        svf_aveg=ones,
        svf_aveg_north=ones,
        svf_aveg_east=ones,
        svf_aveg_south=ones,
        svf_aveg_west=ones,
    )


class _SurfaceStub:
    """Minimal SurfaceData stand-in for resolve_svf — only ``.svf`` is read."""

    def __init__(self, svf):
        self.svf = svf


class _PrecompStub:
    def __init__(self, svf):
        self.svf = svf


def test_resolve_svf_uses_surface_first():
    arrays = _make_svf_arrays()
    surface = _SurfaceStub(svf=arrays)
    bundle = resolve_svf(surface, precomputed=None)  # type: ignore[arg-type]
    assert bundle.svf is arrays.svf
    # svfalfa must be finite (arcsin domain handled safely)
    assert np.all(np.isfinite(bundle.svfalfa))


def test_resolve_svf_falls_back_to_precomputed():
    surface = _SurfaceStub(svf=None)
    arrays = _make_svf_arrays()
    precomputed = _PrecompStub(svf=arrays)
    bundle = resolve_svf(surface, precomputed=precomputed)  # type: ignore[arg-type]
    assert bundle.svf is arrays.svf


def test_resolve_svf_raises_when_nothing_available():
    surface = _SurfaceStub(svf=None)
    with pytest.raises(MissingPrecomputedData, match="Sky View Factor"):
        resolve_svf(surface, precomputed=None)  # type: ignore[arg-type]


def test_resolve_svf_raises_when_precomputed_lacks_svf():
    surface = _SurfaceStub(svf=None)
    precomputed = _PrecompStub(svf=None)
    with pytest.raises(MissingPrecomputedData):
        resolve_svf(surface, precomputed=precomputed)  # type: ignore[arg-type]


def test_adjust_svfbuveg_with_vegetation_applies_psi():
    svf = np.full((4, 4), 0.6, dtype=np.float32)
    svf_veg = np.full((4, 4), 0.7, dtype=np.float32)
    out = adjust_svfbuveg_with_psi(svf, svf_veg, psi=0.03, use_veg=True)
    expected = 0.6 - (1.0 - 0.7) * (1.0 - 0.03)
    assert np.allclose(out, expected, atol=1e-6)
    assert out.dtype == np.float32


def test_adjust_svfbuveg_without_vegetation_returns_svf():
    svf = np.full((4, 4), 0.55, dtype=np.float32)
    svf_veg = np.full((4, 4), 0.0, dtype=np.float32)
    out = adjust_svfbuveg_with_psi(svf, svf_veg, psi=0.03, use_veg=False)
    assert np.allclose(out, 0.55)


def test_adjust_svfbuveg_clamps_negative_values():
    """SVF < (1-svf_veg)*(1-psi) → clamped to 0."""
    svf = np.full((4, 4), 0.05, dtype=np.float32)
    svf_veg = np.zeros((4, 4), dtype=np.float32)
    out = adjust_svfbuveg_with_psi(svf, svf_veg, psi=0.03, use_veg=True)
    assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# io.py — bbox + path helpers, error paths
# ---------------------------------------------------------------------------
from solweig.io import (  # noqa: E402
    _assert_north_up,
    _bounds_to_tuple,
    _compute_bounds_from_transform,
    _normalise_bbox,
    _validate_bbox_within_bounds,
    check_path,
)


def test_assert_north_up_rejects_rotated_gdal_tuple():
    rotated = (0.0, 1.0, 0.5, 100.0, 0.0, -1.0)  # x_rotation != 0
    with pytest.raises(ValueError, match="north-up"):
        _assert_north_up(rotated)


def test_assert_north_up_rejects_short_tuple():
    with pytest.raises(ValueError, match="6 elements"):
        _assert_north_up((0.0, 1.0, 0.0))


def test_assert_north_up_accepts_valid_gdal_tuple():
    _assert_north_up((0.0, 1.0, 0.0, 100.0, 0.0, -1.0))


def test_compute_bounds_from_transform_returns_corner_extent():
    # 10×20 pixel raster, 2 m/px, top-left at (100, 500)
    transform = (100.0, 2.0, 0.0, 500.0, 0.0, -2.0)
    minx, miny, maxx, maxy = _compute_bounds_from_transform(transform, width=20, height=10)
    assert minx == pytest.approx(100.0)
    assert maxx == pytest.approx(140.0)
    assert maxy == pytest.approx(500.0)
    assert miny == pytest.approx(480.0)


def test_normalise_bbox_rejects_wrong_length():
    with pytest.raises(ValueError, match="four numeric"):
        _normalise_bbox([0.0, 1.0, 2.0])


def test_bounds_to_tuple_handles_object_with_attrs():
    class _Bounds:
        left, bottom, right, top = 1.0, 2.0, 3.0, 4.0

    assert _bounds_to_tuple(_Bounds()) == (1.0, 2.0, 3.0, 4.0)


def test_bounds_to_tuple_handles_plain_tuple():
    assert _bounds_to_tuple((1.0, 2.0, 3.0, 4.0)) == (1.0, 2.0, 3.0, 4.0)


def test_validate_bbox_within_bounds_passes_when_contained():
    class _B:
        left, bottom, right, top = 0.0, 0.0, 100.0, 100.0

    _validate_bbox_within_bounds((10.0, 10.0, 50.0, 50.0), _B())


def test_validate_bbox_within_bounds_raises_when_outside():
    class _B:
        left, bottom, right, top = 0.0, 0.0, 100.0, 100.0

    with pytest.raises(ValueError, match="contained"):
        _validate_bbox_within_bounds((-10.0, 10.0, 50.0, 50.0), _B())


def test_check_path_existing_dir_is_returned_as_is(tmp_path):
    out = check_path(tmp_path)
    assert out == tmp_path.absolute()


def test_check_path_creates_intermediate_dirs_when_directory_like(tmp_path):
    p = tmp_path / "a" / "b" / "newdir"
    out = check_path(p, make_dir=True)
    assert out.is_dir()


# ---------------------------------------------------------------------------
# errors.py — exception classes with structured attributes
# ---------------------------------------------------------------------------
from solweig.errors import ConfigurationError, WeatherDataError  # noqa: E402


def test_weather_data_error_without_reason():
    err = WeatherDataError(field="ta", value=999.0)
    assert err.field == "ta"
    assert err.value == 999.0
    assert err.reason is None
    assert "ta" in str(err)
    assert "999" in str(err)


def test_weather_data_error_with_reason_in_message():
    err = WeatherDataError(field="rh", value=200.0, reason="must be 0-100%")
    assert err.reason == "must be 0-100%"
    assert "must be 0-100%" in str(err)


def test_configuration_error_carries_parameter_and_reason():
    err = ConfigurationError(parameter="tile_size", reason="must be >= 256")
    assert err.parameter == "tile_size"
    assert err.reason == "must be >= 256"
    assert "tile_size" in str(err)
    assert "must be >= 256" in str(err)


# ---------------------------------------------------------------------------
# ShadowArrays — legacy NPZ formats, memmap errors, release cache
# ---------------------------------------------------------------------------
from solweig.models.shadow_arrays import (  # noqa: E402
    ShadowArrays,
    _pack_u8_to_bitpacked,
)


def _make_packed_inputs(rows: int, cols: int, patch_count: int = 153):
    """Build bitpacked uint8 inputs sized correctly for ShadowArrays."""
    n_pack = (patch_count + 7) // 8
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(rows, cols, n_pack), dtype=np.uint8)


def test_shadow_arrays_release_float32_cache_resets_lazy_unpack():
    rows, cols = 4, 4
    sa = ShadowArrays(
        _shmat_u8=_make_packed_inputs(rows, cols),
        _vegshmat_u8=_make_packed_inputs(rows, cols),
        _vbshmat_u8=_make_packed_inputs(rows, cols),
    )
    # Force unpack
    _ = sa.shmat
    _ = sa.vegshmat
    _ = sa.vbshmat
    assert sa._shmat_f32 is not None
    sa.release_float32_cache()
    assert sa._shmat_f32 is None
    assert sa._vegshmat_f32 is None
    assert sa._vbshmat_f32 is None
    # Re-access reallocates
    _ = sa.shmat
    assert sa._shmat_f32 is not None


def test_shadow_arrays_from_npz_legacy_float_format_packs(tmp_path):
    """Legacy NPZ with float32 0/1 arrays gets packed into uint8."""
    rows, cols, n_patches = 4, 4, 8
    rng = np.random.default_rng(1)
    raw = (rng.random((rows, cols, n_patches)) > 0.5).astype(np.float32)
    npz = tmp_path / "shadowmats.npz"
    np.savez(npz, shadowmat=raw, vegshadowmat=raw, vbshmat=raw)

    sa = ShadowArrays.from_npz(npz)
    assert sa.patch_count == n_patches
    # Decoded float32 view round-trips the same 0/1 pattern
    assert np.array_equal(sa.shmat[:, :, :n_patches], raw)


def test_shadow_arrays_from_npz_legacy_u8_format(tmp_path):
    """Legacy NPZ with uint8 0/255 arrays also packs correctly."""
    rows, cols, n_patches = 4, 4, 16
    raw = np.full((rows, cols, n_patches), 255, dtype=np.uint8)
    raw[:, :, 0] = 0  # first patch all-blocked
    npz = tmp_path / "shadowmats.npz"
    np.savez(npz, shadowmat=raw, vegshadowmat=raw, vbshmat=raw)

    sa = ShadowArrays.from_npz(npz)
    assert sa.patch_count == n_patches
    # All patches visible (1.0) except the first one
    assert np.all(sa.shmat[:, :, 1:] == 1.0)
    assert np.all(sa.shmat[:, :, 0] == 0.0)


def test_shadow_arrays_from_npz_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Shadow matrices"):
        ShadowArrays.from_npz(tmp_path / "missing.npz")


def test_shadow_arrays_from_memmap_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Shadow memmap"):
        ShadowArrays.from_memmap(tmp_path / "nope")


def test_shadow_arrays_from_memmap_missing_metadata_raises(tmp_path):
    d = tmp_path / "shadow"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="metadata"):
        ShadowArrays.from_memmap(d)


def test_shadow_arrays_from_memmap_invalid_shape_raises(tmp_path):
    d = tmp_path / "shadow"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"shape": [4, 4], "patch_count": 153}))
    with pytest.raises(ValueError, match="shape"):
        ShadowArrays.from_memmap(d)


def test_shadow_arrays_from_memmap_missing_data_file_raises(tmp_path):
    """metadata.json present + valid, but data files missing → FileNotFoundError."""
    d = tmp_path / "shadow"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"shape": [4, 4, 20], "patch_count": 153}))
    with pytest.raises(FileNotFoundError, match="memmap file"):
        ShadowArrays.from_memmap(d)


def test_shadow_arrays_diffsh_without_vegetation_returns_shmat():
    rows, cols = 4, 4
    sa = ShadowArrays(
        _shmat_u8=_make_packed_inputs(rows, cols),
        _vegshmat_u8=_make_packed_inputs(rows, cols),
        _vbshmat_u8=_make_packed_inputs(rows, cols),
    )
    out = sa.diffsh(use_vegetation=False)
    assert np.array_equal(out, sa.shmat)


# ---------------------------------------------------------------------------
# Location.from_dsm_crs — exercises CRS reprojection via a synthetic GeoTIFF
# ---------------------------------------------------------------------------
def _write_synthetic_geotiff_utm34n(path: Path, shape=(50, 50)) -> None:
    """Write a tiny GeoTIFF in EPSG:32634 (UTM 34N, covers Greece).

    UTM 34N central meridian 21°E, so a center near (500000 m east,
    4205000 m north) lands near Athens (37.97°N, ~23.7°E).
    """
    from solweig.io import save_raster

    data = np.zeros(shape, dtype=np.float32)
    # GDAL geotransform: top-left x, pixel_w, 0, top-left y, 0, -pixel_h
    transform = [500000.0, 30.0, 0.0, 4205000.0 + shape[0] * 30.0, 0.0, -30.0]
    # WKT for EPSG:32634
    wkt = (
        'PROJCS["WGS 84 / UTM zone 34N",GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",21],'
        'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
        'PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","32634"]]'
    )
    save_raster(str(path), data, transform, crs_wkt=wkt, use_cog=False, generate_preview=False)


def test_location_from_dsm_crs_extracts_utm_centre(tmp_path):
    """A UTM-34N raster centred near Athens → lat≈37.97, lon≈21° area."""
    tif = tmp_path / "dsm_utm.tif"
    _write_synthetic_geotiff_utm34n(tif)
    loc = Location.from_dsm_crs(tif, utc_offset=2)
    # Bounds are crude; just check sane plausibility for Greece.
    assert 35.0 < loc.latitude < 42.0
    assert 19.0 < loc.longitude < 26.0
    assert loc.utc_offset == 2


def test_location_from_dsm_crs_missing_crs_raises(tmp_path):
    """A GeoTIFF written with crs_wkt=None should fail Location.from_dsm_crs."""
    from solweig.io import save_raster

    tif = tmp_path / "no_crs.tif"
    save_raster(
        str(tif),
        np.zeros((10, 10), dtype=np.float32),
        [0.0, 1.0, 0.0, 10.0, 0.0, -1.0],
        crs_wkt=None,
        use_cog=False,
        generate_preview=False,
    )
    with pytest.raises(ValueError, match="CRS"):
        Location.from_dsm_crs(tif)


def test_shadow_arrays_diffsh_with_vegetation_combines_arrays():
    """diffsh formula: shmat - (1 - vegshmat) * (1 - psi)."""
    rows, cols, n_patches = 2, 2, 8
    shmat = np.ones((rows, cols, n_patches), dtype=np.uint8) * 255
    vegshmat = np.zeros((rows, cols, n_patches), dtype=np.uint8)  # fully blocked by veg
    sa = ShadowArrays(
        _shmat_u8=_pack_u8_to_bitpacked(shmat),
        _vegshmat_u8=_pack_u8_to_bitpacked(vegshmat),
        _vbshmat_u8=_pack_u8_to_bitpacked(vegshmat),
        _n_patches=n_patches,
    )
    out = sa.diffsh(transmissivity=0.03, use_vegetation=True)
    # shmat=1, vegshmat=0 → 1 - (1-0)*(1-0.03) = 1 - 0.97 = 0.03
    assert np.allclose(out, 0.03, atol=1e-5)
    assert out.dtype == np.float32
