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
