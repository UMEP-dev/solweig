"""Parity gates against the vendored UMEP-processing 2026a reference.

The pip ``umep`` package (the usual parity reference) ships only the 2025a
generation; the 2026a Lside variant lives in UMEP-dev/UMEP-processing and is
vendored verbatim under ``tests/reference/umep_2026a/`` with provenance.

Gates here pin the Rust ``lside_veg_v2026`` port to upstream's
``Lside_veg_v2026`` across a sweep of sun positions, clearness indices, and
both longwave modes. The v2022a baseline keeps its own gate in
``test_parity_extended.py`` against the pip package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from solweig.constants import SBC
from solweig.rustalgos import vegetation

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

from umep_2026a.Lside_veg import Lside_veg_v2026  # noqa: E402

RNG = np.random.default_rng(20260706)
SHAPE = (24, 24)


def _svf_set():
    """Synthetic, mutually consistent SVF fields in the physical ranges."""
    svf = {}
    for d in ("E", "S", "W", "N"):
        base = RNG.uniform(0.35, 0.98, SHAPE).astype(np.float32)
        veg = np.clip(base + RNG.uniform(0.0, 0.4, SHAPE).astype(np.float32), 0.0, 1.0)
        aveg = np.clip(veg - RNG.uniform(0.0, 0.05, SHAPE).astype(np.float32), 0.0, 1.0)
        svf[d] = (base, veg.astype(np.float32), aveg.astype(np.float32))
    return svf


SVF = _svf_set()
LDOWN = RNG.uniform(280.0, 430.0, SHAPE).astype(np.float32)
F_SH = RNG.uniform(0.0, 1.0, SHAPE).astype(np.float32)
TA = 24.0
TW = 12.0
EWALL = 0.9
ESKY = 0.88
T_OFFSET = 0.0


def _run_upstream(azimuth, altitude, ci, aniso):
    return Lside_veg_v2026(
        SVF["S"][0].astype(np.float64),
        SVF["W"][0].astype(np.float64),
        SVF["N"][0].astype(np.float64),
        SVF["E"][0].astype(np.float64),
        SVF["E"][1].astype(np.float64),
        SVF["S"][1].astype(np.float64),
        SVF["W"][1].astype(np.float64),
        SVF["N"][1].astype(np.float64),
        SVF["E"][2].astype(np.float64),
        SVF["S"][2].astype(np.float64),
        SVF["W"][2].astype(np.float64),
        SVF["N"][2].astype(np.float64),
        azimuth,
        altitude,
        TA,
        TW,
        SBC,
        EWALL,
        LDOWN.astype(np.float64),
        ESKY,
        T_OFFSET,
        F_SH.astype(np.float64),
        ci,
        1 if aniso else 0,
    )


def _run_rust(azimuth, altitude, ci, aniso):
    result = vegetation.lside_veg_v2026(
        SVF["S"][0],
        SVF["W"][0],
        SVF["N"][0],
        SVF["E"][0],
        SVF["E"][1],
        SVF["S"][1],
        SVF["W"][1],
        SVF["N"][1],
        SVF["E"][2],
        SVF["S"][2],
        SVF["W"][2],
        SVF["N"][2],
        azimuth,
        altitude,
        TA,
        TW,
        SBC,
        EWALL,
        LDOWN,
        ESKY,
        T_OFFSET,
        F_SH,
        ci,
        aniso,
    )
    return (
        np.asarray(result.least),
        np.asarray(result.lsouth),
        np.asarray(result.lwest),
        np.asarray(result.lnorth),
    )


@pytest.mark.parametrize("azimuth", [5.0, 95.0, 185.0, 275.0, 359.0])
@pytest.mark.parametrize("altitude", [-5.0, 3.0, 35.0, 65.0])
@pytest.mark.parametrize("ci", [0.4, 0.95])
def test_lside_v2026_isotropic_matches_upstream(azimuth, altitude, ci):
    """Rust v2026 isotropic Lside matches the vendored upstream reference."""
    expected = _run_upstream(azimuth, altitude, ci, aniso=False)
    got = _run_rust(azimuth, altitude, ci, aniso=False)
    for direction, e, g in zip("ESWN", expected, got, strict=True):
        np.testing.assert_allclose(
            g,
            e,
            rtol=1e-3,
            atol=1e-2,
            err_msg=f"L{direction} mismatch at az={azimuth} alt={altitude} CI={ci}",
        )


@pytest.mark.parametrize("azimuth", [45.0, 225.0])
def test_lside_v2026_anisotropic_returns_zeros(azimuth):
    """v2026 anisotropic branch returns zeros in both implementations."""
    expected = _run_upstream(azimuth, 40.0, 0.9, aniso=True)
    got = _run_rust(azimuth, 40.0, 0.9, aniso=True)
    for e, g in zip(expected, got, strict=True):
        assert np.all(np.asarray(e) == 0.0)
        assert np.all(g == 0.0)


def test_v2026_iso_differs_from_v2022a_by_ground_and_lup_reflection():
    """The v2026/v2022a difference is exactly Lup·0.5 + Lup·viktrefl·(1−ewall)·0.5.

    Sanity check that the variant split changes only the documented terms:
    with Lup == 0, v2022a and v2026 must agree exactly.
    """
    zeros = np.zeros(SHAPE, dtype=np.float32)
    v2022 = vegetation.lside_veg(
        SVF["S"][0],
        SVF["W"][0],
        SVF["N"][0],
        SVF["E"][0],
        SVF["E"][1],
        SVF["S"][1],
        SVF["W"][1],
        SVF["N"][1],
        SVF["E"][2],
        SVF["S"][2],
        SVF["W"][2],
        SVF["N"][2],
        95.0,
        35.0,
        TA,
        TW,
        SBC,
        EWALL,
        LDOWN,
        ESKY,
        T_OFFSET,
        F_SH,
        0.9,
        zeros,
        zeros,
        zeros,
        zeros,
        False,
    )
    got = _run_rust(95.0, 35.0, 0.9, aniso=False)
    for e, g in zip((v2022.least, v2022.lsouth, v2022.lwest, v2022.lnorth), got, strict=True):
        np.testing.assert_allclose(g, np.asarray(e), rtol=0, atol=0)


# ── Ground-surface scheme: surface temperature (force-restore/OHM/RK2) ──────

matplotlib = pytest.importorskip("matplotlib")  # vendored module imports it

from solweig.rustalgos import ground as rust_ground  # noqa: E402
from umep_2026a.ground_surface import surfaceTemperature_calc  # noqa: E402


def _ground_scene():
    """Synthetic per-landcover grids, including water pixels and a shadow edge."""
    rng = np.random.default_rng(7)
    shape = SHAPE
    lc = np.zeros(shape, dtype=np.float32)
    lc[:, 8:14] = 1.0  # asphalt band
    lc[18:24, 0:6] = 7.0  # water pocket
    alb = np.where(lc == 1, 0.18, np.where(lc == 7, 0.05, 0.2)).astype(np.float32)
    emis = np.where(lc == 7, 0.98, 0.95).astype(np.float32)
    cap = np.where(lc == 1, 1.94e6, np.where(lc == 7, 4.18e6, 2.11e6)).astype(np.float32)
    diff = np.where(lc == 1, 3.8e-7, np.where(lc == 7, 1e-7, 7.2e-7)).astype(np.float32)
    a1 = np.where(lc == 1, 0.5, np.where(lc == 7, 0.1, 0.61)).astype(np.float32)
    a2 = np.where(lc == 1, 0.28, np.where(lc == 7, 0.0, 0.28)).astype(np.float32)
    a3 = np.where(lc == 1, -31.45, np.where(lc == 7, -10.0, -23.9)).astype(np.float32)
    tg0 = (24.0 + rng.uniform(-2.0, 6.0, shape)).astype(np.float32)
    tm0 = np.full(shape, 22.0, dtype=np.float32)
    shadow_a = np.ones(shape, dtype=np.float32)
    shadow_a[:, 0:6] = 0.0
    shadow_b = np.ones(shape, dtype=np.float32)
    shadow_b[0:10, :] = 0.0  # abrupt transition exercises the damping mask
    return lc, alb, emis, cap, diff, a1, a2, a3, tg0, tm0, shadow_a, shadow_b


def test_surface_temperature_matches_upstream_over_chained_steps():
    """Rust force-restore/OHM/RK2 step matches the vendored reference.

    Four chained hourly steps with varying forcing and a shadow flip (which
    exercises the ground-heat-flux damping mask) and water pixels (slab
    model). State (Tg, Rn, Rn_past, G) carries between steps in both
    implementations.
    """
    lc, alb, emis, cap, diff, a1, a2, a3, tg0, tm0, shadow_a, shadow_b = _ground_scene()
    shape = lc.shape
    timestep_s = 3600.0
    rh = 55.0

    forcing = [
        (620.0, 380.0, shadow_a, shadow_a),
        (710.0, 390.0, shadow_b, shadow_a),  # shadow flip
        (300.0, 370.0, shadow_b, shadow_b),
        (0.0, 350.0, shadow_a, shadow_b),  # evening + flip back
    ]

    # Upstream (float64, in-place mutation: pass fresh copies it may own)
    tg_py = tg0.astype(np.float64).copy()
    rn_py = np.zeros(shape, dtype=np.float64)
    rn_past_py = np.zeros(shape, dtype=np.float64)
    g_py = np.zeros(shape, dtype=np.float64)

    # Rust (float32, functional)
    tg_rs = tg0.copy()
    rn_rs = np.zeros(shape, dtype=np.float32)
    rn_past_rs = np.zeros(shape, dtype=np.float32)
    g_rs = np.zeros(shape, dtype=np.float32)

    for step, (kdown_v, ldown_v, shadow, shadow_past) in enumerate(forcing):
        kdown = np.full(shape, kdown_v)
        ldown = np.full(shape, ldown_v)

        tg_py, rn_py, rn_past_py, g_py = surfaceTemperature_calc(
            kdown.astype(np.float64),
            ldown.astype(np.float64),
            rn_py,
            rn_past_py,
            g_py,
            tg_py,
            tm0.astype(np.float64),
            alb.astype(np.float64),
            emis.astype(np.float64),
            cap.astype(np.float64),
            diff.astype(np.float64),
            lc.astype(np.float64),
            a1.astype(np.float64),
            a2.astype(np.float64),
            a3.astype(np.float64),
            timestep_s,
            rh,
            shadow.astype(np.float64),
            shadow_past.astype(np.float64),
        )

        result = rust_ground.surface_temperature_calc(
            kdown.astype(np.float32),
            ldown.astype(np.float32),
            rn_rs,
            rn_past_rs,
            g_rs,
            tg_rs,
            tm0,
            alb,
            emis,
            cap,
            diff,
            lc,
            a1,
            a2,
            a3,
            timestep_s,
            rh,
            shadow,
            shadow_past,
        )
        tg_rs = np.asarray(result.tg)
        rn_rs = np.asarray(result.rn)
        rn_past_rs = np.asarray(result.rn_past)
        g_rs = np.asarray(result.g)

        np.testing.assert_allclose(tg_rs, tg_py, rtol=1e-3, atol=0.05, err_msg=f"Tg mismatch at step {step}")
        np.testing.assert_allclose(rn_rs, rn_py, rtol=1e-3, atol=0.5, err_msg=f"Rn mismatch at step {step}")
        np.testing.assert_allclose(g_rs, g_py, rtol=1e-3, atol=0.5, err_msg=f"G mismatch at step {step}")


# ── Ground-surface scheme: outgoing longwave solid-angle march ──────────────

from umep_2026a.ground_surface import outgoingLongwave_calc  # noqa: E402

OUTGOING_FIELDS = [
    "gvf_lup",
    "gvfalbsun",
    "gvfalbtot",
    "gvf_lup_e",
    "gvfalbsun_e",
    "gvfalbtot_e",
    "gvf_lup_s",
    "gvfalbsun_s",
    "gvfalbtot_s",
    "gvf_lup_w",
    "gvfalbsun_w",
    "gvfalbtot_w",
    "gvf_lup_n",
    "gvfalbsun_n",
    "gvfalbtot_n",
    "gvf_lside_w",
    "gvf_lside_s",
    "gvf_lside_e",
    "gvf_lside_n",
]


@pytest.mark.parametrize("sizepx", [1.0, 2.0])
def test_outgoing_longwave_matches_upstream(sizepx):
    """Rust solid-angle march matches the vendored reference, all 19 outputs.

    Scene includes buildings (roof fallback), walls with a sunlit subset
    (wall emission + albedo terms), shadow structure, and mixed emissivity/
    albedo. sizepx parametrized because the march radius and the below-pixel
    view factor depend on it.
    """
    rng = np.random.default_rng(11)
    shape = (40, 40)
    buildings = np.ones(shape, dtype=np.float32)
    buildings[12:20, 12:20] = 0.0  # building/roof block
    walls = np.zeros(shape, dtype=np.float32)
    walls[11, 11:21] = 6.0  # north face
    walls[20, 11:21] = 6.0
    walls[12:20, 11] = 6.0
    walls[12:20, 20] = 6.0
    sunwall = np.zeros(shape, dtype=np.float32)
    sunwall[20, 11:21] = 6.0  # south-facing walls sunlit
    sunwall[12:20, 20] = 4.0
    shadow = np.ones(shape, dtype=np.float32)
    shadow[20:28, 8:20] = 0.0
    tg = (26.0 + rng.uniform(-3.0, 8.0, shape)).astype(np.float32)
    ldown = np.full(shape, 380.0, dtype=np.float32)
    emis = np.where(buildings > 0, 0.95, 0.92).astype(np.float32)
    alb = np.where(buildings > 0, 0.18, 0.15).astype(np.float32)
    tgwall = 6.0
    ta = 27.0
    rows, cols = shape

    expected = outgoingLongwave_calc(
        tg.astype(np.float64),
        tgwall,
        ta,
        ldown.astype(np.float64),
        emis.astype(np.float64),
        alb.astype(np.float64),
        buildings.astype(np.float64),
        shadow.astype(np.float64),
        sunwall.astype(np.float64).copy(),  # upstream mutates this in place
        walls.astype(np.float64),
        rows,
        cols,
        sizepx,
    )

    result = rust_ground.outgoing_longwave_calc(
        tg,
        tgwall,
        ta,
        ldown,
        emis,
        alb,
        buildings,
        shadow,
        sunwall,
        walls,
        sizepx,
    )

    for i, field in enumerate(OUTGOING_FIELDS):
        got = np.asarray(getattr(result, field), dtype=np.float64)
        want = np.asarray(expected[i], dtype=np.float64)
        np.testing.assert_allclose(
            got,
            want,
            rtol=1e-3,
            atol=0.5,
            err_msg=f"{field} diverges from upstream at sizepx={sizepx}",
        )


def test_outgoing_longwave_does_not_mutate_inputs():
    """The port must not reproduce upstream's in-place sunwall binarization."""
    shape = (20, 20)
    sunwall = np.zeros(shape, dtype=np.float32)
    sunwall[5, 5:10] = 7.0
    sunwall_before = sunwall.copy()
    walls = np.zeros(shape, dtype=np.float32)
    walls[5, 5:10] = 7.0
    ones = np.ones(shape, dtype=np.float32)
    rust_ground.outgoing_longwave_calc(
        np.full(shape, 25.0, dtype=np.float32),
        5.0,
        25.0,
        np.full(shape, 370.0, dtype=np.float32),
        np.full(shape, 0.95, dtype=np.float32),
        np.full(shape, 0.2, dtype=np.float32),
        ones,
        ones,
        sunwall,
        walls,
        1.0,
    )
    np.testing.assert_array_equal(sunwall, sunwall_before)


# ── Ground-surface scheme: per-run initialisation ───────────────────────────

import json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

from solweig.components.ground_scheme import initiate_ground_scheme  # noqa: E402
from solweig.loaders import load_params  # noqa: E402
from umep_2026a.ground_surface import initiate_groundScheme  # noqa: E402

_MATERIALS_JSON = _Path(__file__).resolve().parents[2] / "pysrc" / "solweig" / "data" / "default_materials.json"


@pytest.mark.parametrize("latitude", [57.7, -33.9])
@pytest.mark.parametrize("day", [40, 200])
def test_initiate_ground_scheme_matches_upstream(latitude, day):
    """Initial grids match upstream for both hemispheres and seasons."""
    lc = np.zeros((16, 16), dtype=np.float32)
    lc[:, 4:6] = 1.0
    lc[2:6, 10:14] = 2.0
    lc[8:12, 0:3] = 5.0
    lc[12:16, 12:16] = 6.0
    lc[0:2, 0:2] = 7.0
    lc[6, 6] = 101.0  # wall-material code, remapped to roofs

    ta_series = np.array([16.0, 15.5, 17.0, 21.0, 24.0, 25.5, 23.0, 19.0])

    with open(_MATERIALS_JSON) as fh:
        params_dict = json.load(fh)
    expected = initiate_groundScheme(
        lc.astype(np.float64).copy(),  # upstream mutates the grid
        params_dict,
        day,
        ta_series.astype(np.float64),
        {"latitude": latitude, "longitude": 12.0},
    )
    exp_names = ["tg", "tm", "rn", "rn_past", "g", "cap", "diff", "a1", "a2", "a3"]

    state = initiate_ground_scheme(lc, load_params(), day, ta_series, latitude)

    for name, want in zip(exp_names, expected, strict=True):
        got = np.asarray(getattr(state, name), dtype=np.float64)
        np.testing.assert_allclose(
            got, np.asarray(want, dtype=np.float64), rtol=1e-5, atol=1e-4, err_msg=f"{name} mismatch"
        )
    # Wall-material pixels were remapped to roofs
    assert state.lc_grid[6, 6] == 2.0
    # Caller's grid untouched (upstream mutates; our port must not)
    assert lc[6, 6] == 101.0


def test_initiate_ground_scheme_rejects_unparameterized_class():
    """A land-cover class without OHM coefficients raises a structured error."""
    from solweig.errors import InvalidSurfaceData

    lc = np.zeros((4, 4), dtype=np.float32)
    lc[0, 0] = 99.0  # Walls: named, but no OHM coefficients
    with pytest.raises(InvalidSurfaceData, match="OHM"):
        initiate_ground_scheme(lc, load_params(), 180, [20.0], 57.7)
