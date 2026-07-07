"""Golden regression tests for the opt-in UMEP 2026a ground-surface scheme.

Gates the fused-pipeline wiring of the scheme (force-restore/OHM surface
temperature + solid-angle outgoing longwave march + Lside_veg_v2026) behind
``use_ground_scheme`` / ``use_outgoing_longwave``.

Ground truth: Rust (regression baseline). The scheme's components are
individually parity-gated against the vendored upstream reference in
``tests/spec/test_parity_2026a.py``; these fixtures pin the end-to-end
composition (ordering, state carry, flux assembly) so numerical drift in
the opt-in path is caught the same way the baseline path is.

Fixtures live in ``fixtures/ground_scheme/``. Regenerate with::

    uv run python tests/golden/test_golden_ground_scheme.py --regenerate

Only do this when intentionally updating the ground truth, and record the
scientific justification in the commit message.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ground_scheme"

# Steps captured in the fixture: pre-dawn (both branches' state carry),
# mid-morning (shadow flip damping), noon (peak fluxes), evening/night
# (water-vs-asphalt inertia).
CAPTURE_HOURS = [4, 9, 13, 21]
SERIES_START_HOUR = 3
SERIES_N_HOURS = 19  # 03:00 .. 21:00 inclusive


def _build_scene():
    """Deterministic 40x40 scene: building block, walls, mixed land cover."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from conftest import make_mock_svf
    from solweig import SurfaceData
    from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls

    shape = (40, 40)
    dsm = np.zeros(shape, dtype=np.float32)
    dsm[14:24, 14:24] = 12.0  # building block

    lc = np.zeros(shape, dtype=np.uint8)  # 0 = paved
    lc[:, 0:10] = 1  # asphalt band (west)
    lc[30:40, 24:40] = 5  # grass (south-east)
    lc[0:7, 28:40] = 7  # water (north-east)
    lc[14:24, 14:24] = 2  # buildings

    wall_ht = findwalls(dsm, 0.3)
    wall_asp = filter1Goodwin_as_aspect_v3(wall_ht, 1.0, dsm)

    surface = SurfaceData(
        dsm=dsm,
        pixel_size=1.0,
        svf=make_mock_svf(shape),
        land_cover=lc,
        wall_height=wall_ht,
        wall_aspect=wall_asp,
    )
    return surface, lc


def _weather_series():
    from solweig import Weather

    base = datetime(2024, 7, 15, SERIES_START_HOUR, 0)
    series = []
    for i in range(SERIES_N_HOURS):
        hour = SERIES_START_HOUR + i
        # Deterministic smooth diurnal forcing (Gothenburg July day)
        sun_frac = np.sin(np.pi * (hour - 5.0) / 16.0)
        series.append(
            Weather(
                datetime=base + timedelta(hours=i),
                ta=16.0 + 9.0 * max(sun_frac, 0.0),
                rh=60.0,
                global_rad=max(0.0, 720.0 * sun_frac),
            )
        )
    return series


def _run_scheme_series():
    """Run the 19-step series through the fused scheme path; return captures."""
    from solweig import Location
    from solweig.api import _calculate_single
    from solweig.components.ground_scheme import initiate_ground_scheme
    from solweig.loaders import load_params
    from solweig.models import ThermalState
    from solweig.timeseries import _precompute_weather

    surface, lc = _build_scene()
    location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
    series = _weather_series()
    _precompute_weather(series, location)

    state = ThermalState.initial(surface.shape)
    state.timestep_dec = 1.0 / 24.0
    ta_first_day = [w.ta for w in series]
    gss = initiate_ground_scheme(
        lc,
        load_params(),
        series[0].datetime.timetuple().tm_yday,
        ta_first_day,
        location.latitude,
    )

    captures: dict[str, np.ndarray] = {}
    for weather in series:
        result = _calculate_single(
            surface=surface,
            location=location,
            weather=weather,
            use_anisotropic_sky=False,
            state=state,
            ground_scheme_state=gss,
            return_state_copy=False,
            _requested_outputs={"tmrt", "shadow", "kdown", "kup", "ldown", "lup"},
        )
        state = result.state if result.state is not None else state
        hour = weather.datetime.hour
        if hour in CAPTURE_HOURS:
            captures[f"tmrt_h{hour:02d}"] = np.asarray(result.tmrt, dtype=np.float32)
            captures[f"lup_h{hour:02d}"] = np.asarray(result.lup, dtype=np.float32)
            captures[f"tg_h{hour:02d}"] = gss.tg.copy()
        if hour == 13:
            captures["kdown_h13"] = np.asarray(result.kdown, dtype=np.float32)
            captures["kup_h13"] = np.asarray(result.kup, dtype=np.float32)
            captures["ldown_h13"] = np.asarray(result.ldown, dtype=np.float32)
    return captures, lc


@pytest.fixture(scope="module")
def scheme_run():
    captures, lc = _run_scheme_series()
    return captures, lc


@pytest.fixture(scope="module")
def golden():
    path = FIXTURE_DIR / "scheme_iso_baseline.npz"
    if not path.exists():
        pytest.skip(f"Golden fixture missing: {path} (regenerate with --regenerate)")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class TestGroundSchemeGoldenRegression:
    """End-to-end regression against the pinned scheme baseline."""

    @pytest.mark.parametrize("hour", CAPTURE_HOURS)
    def test_tmrt_matches_golden(self, scheme_run, golden, hour):
        captures, _ = scheme_run
        key = f"tmrt_h{hour:02d}"
        np.testing.assert_allclose(captures[key], golden[key], rtol=1e-4, atol=0.01)

    @pytest.mark.parametrize("hour", CAPTURE_HOURS)
    def test_lup_matches_golden(self, scheme_run, golden, hour):
        captures, _ = scheme_run
        key = f"lup_h{hour:02d}"
        np.testing.assert_allclose(captures[key], golden[key], rtol=1e-4, atol=0.1)

    @pytest.mark.parametrize("hour", CAPTURE_HOURS)
    def test_tg_state_matches_golden(self, scheme_run, golden, hour):
        captures, _ = scheme_run
        key = f"tg_h{hour:02d}"
        np.testing.assert_allclose(captures[key], golden[key], rtol=1e-4, atol=0.01)

    @pytest.mark.parametrize("name", ["kdown_h13", "kup_h13", "ldown_h13"])
    def test_noon_shortwave_longwave_match_golden(self, scheme_run, golden, name):
        captures, _ = scheme_run
        np.testing.assert_allclose(captures[name], golden[name], rtol=1e-4, atol=0.1)


class TestGroundSchemePhysicalProperties:
    """Physical sanity of the scheme output (independent of the fixture)."""

    def test_outputs_finite(self, scheme_run):
        captures, _ = scheme_run
        for name, arr in captures.items():
            assert np.isfinite(arr).all(), f"{name} contains non-finite values"

    def test_lup_plausible_range(self, scheme_run):
        captures, _ = scheme_run
        for hour in CAPTURE_HOURS:
            lup = captures[f"lup_h{hour:02d}"]
            assert lup.min() > 250.0 and lup.max() < 800.0, (
                f"Lup at h{hour} outside plausible range: {lup.min():.0f}..{lup.max():.0f} W/m2"
            )

    def test_water_thermal_inertia(self, scheme_run):
        """Water heats less by day and stays warmer into the evening (slab model)."""
        captures, lc = scheme_run
        water = lc == 7
        asphalt = lc == 1
        tg_noon = captures["tg_h13"]
        rise_water = np.mean(tg_noon[water]) - np.mean(captures["tg_h04"][water])
        rise_asphalt = np.mean(tg_noon[asphalt]) - np.mean(captures["tg_h04"][asphalt])
        assert rise_water < rise_asphalt, "water warmed as fast as asphalt — slab inertia missing"

    def test_asphalt_hotter_than_grass_at_noon(self, scheme_run):
        captures, lc = scheme_run
        tg = captures["tg_h13"]
        assert np.mean(tg[lc == 1]) > np.mean(tg[lc == 5]), "asphalt not hotter than grass at noon"

    def test_scheme_diverges_from_baseline(self):
        """The flags must actually change the physics (guard against silent no-op)."""
        from solweig import Location
        from solweig.api import _calculate_single
        from solweig.models import ThermalState
        from solweig.timeseries import _precompute_weather

        surface, _ = _build_scene()
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        series = _weather_series()
        _precompute_weather(series, location)
        noon = next(w for w in series if w.datetime.hour == 13)

        state = ThermalState.initial(surface.shape)
        state.timestep_dec = 1.0 / 24.0
        baseline = _calculate_single(
            surface=surface,
            location=location,
            weather=noon,
            use_anisotropic_sky=False,
            state=state,
            return_state_copy=False,
            _requested_outputs={"tmrt"},
        )
        scheme_captures, _ = _run_scheme_series()
        assert not np.allclose(scheme_captures["tmrt_h13"], np.asarray(baseline.tmrt), atol=0.05), (
            "scheme Tmrt is indistinguishable from baseline — the opt-in flags had no effect"
        )


def _regenerate():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    captures, _ = _run_scheme_series()
    out = FIXTURE_DIR / "scheme_iso_baseline.npz"
    np.savez_compressed(out, **captures)
    print(f"Wrote {out} ({len(captures)} arrays)")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print(__doc__)
