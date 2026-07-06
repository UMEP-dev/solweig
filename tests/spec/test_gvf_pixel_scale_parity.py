"""GVF vs UMEP parity across pixel sizes.

The GVF ray-march covers a fixed metric source area around the person
(Smidt et al. source-area model: ``first`` and ``second`` are distances in
metres, converted to pixels inside the kernel). UMEP converts with
``pixels = round(metres * scale)`` where ``scale = 1 / pixel_size``; the
solweig kernel receives ``scale`` as metres per pixel, so the conversion
must divide. The two agree only at 1 m pixels, which is where all golden
fixtures and validation sites sit, so this parity gate pins the 0.5 m and
2.0 m cases explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
from solweig.constants import SBC
from solweig.rustalgos import gvf as gvf_module

umep_gvf = pytest.importorskip(
    "umep.functions.SOLWEIGpython.gvf_2018a",
    reason="UMEP reference implementation not installed",
)


@pytest.fixture(autouse=True, scope="module")
def _cpu_gvf(cpu_only):
    """Use the CPU path so the comparison is deterministic."""


# The domain must exceed the source-area reach in pixels at the finest pixel
# size tested (36 m / 0.5 m = 72 px), or UMEP's own slice arithmetic breaks.
SHAPE = (100, 100)
TA = 25.0
TGWALL = 2.0
EWALL = 0.90
ALBEDO_B = 0.20
FIRST_M = 2.0  # round(1.8 m person height), metres
SECOND_M = 36.0  # 1.8 m x 20 (classic UMEP source-area reach), metres


def _make_inputs():
    """Synthetic block scene; wallsun == walls so the sunwall mask is exact."""
    walls = np.zeros(SHAPE, dtype=np.float32)
    buildings = np.ones(SHAPE, dtype=np.float32)
    dirwalls = np.zeros(SHAPE, dtype=np.float32)

    walls[45:55, 45:55] = 12.0
    buildings[45:55, 45:55] = 0.0
    dirwalls[45:55, 45:55] = 180.0

    wallsun = walls.copy()  # fully sunlit walls
    shadow = np.ones(SHAPE, dtype=np.float32)
    shadow[55:65, 45:55] = 0.0  # cast shadow south of the block
    tg = np.full(SHAPE, 2.0, dtype=np.float32)
    emis_grid = np.full(SHAPE, 0.95, dtype=np.float32)
    alb_grid = np.full(SHAPE, 0.15, dtype=np.float32)
    return wallsun, walls, buildings, shadow, dirwalls, tg, emis_grid, alb_grid


def _run_solweig(pixel_size: float):
    wallsun, walls, buildings, shadow, dirwalls, tg, emis_grid, alb_grid = _make_inputs()
    params = gvf_module.GvfScalarParams(
        scale=pixel_size,  # metres per pixel
        first=FIRST_M,
        second=SECOND_M,
        tgwall=TGWALL,
        ta=TA,
        ewall=EWALL,
        sbc=SBC,
        albedo_b=ALBEDO_B,
        twater=TA,
        landcover=False,
    )
    return gvf_module.gvf_calc(wallsun, walls, buildings, shadow, dirwalls, tg, emis_grid, alb_grid, None, params)


def _run_umep(pixel_size: float):
    wallsun, walls, buildings, shadow, dirwalls, tg, emis_grid, alb_grid = _make_inputs()
    rows, cols = SHAPE
    return umep_gvf.gvf_2018a(
        wallsun.astype(np.float64),
        walls.astype(np.float64),
        buildings.astype(np.float64),
        1.0 / pixel_size,  # UMEP scale is pixels per metre
        shadow.astype(np.float64),
        FIRST_M,
        SECOND_M,
        dirwalls.astype(np.float64),
        tg.astype(np.float64),
        TGWALL,
        TA,
        emis_grid.astype(np.float64),
        EWALL,
        alb_grid.astype(np.float64),
        SBC,
        ALBEDO_B,
        rows,
        cols,
        TA,  # Twater
        None,  # lc_grid
        False,  # landcover
    )


FIELDS = [
    ("gvf_lup", 0),
    ("gvfalb", 1),
    ("gvfalbnosh", 2),
    ("gvf_lup_e", 3),
    ("gvf_lup_s", 6),
    ("gvf_lup_w", 9),
    ("gvf_lup_n", 12),
]


@pytest.mark.parametrize("pixel_size", [1.0, 0.5, 2.0])
def test_gvf_matches_umep_across_pixel_sizes(pixel_size):
    """solweig GVF must reproduce UMEP's gvf_2018a at 0.5, 1 and 2 m pixels."""
    ours = _run_solweig(pixel_size)
    theirs = _run_umep(pixel_size)

    for attr, umep_idx in FIELDS:
        got = np.asarray(getattr(ours, attr), dtype=np.float64)
        want = np.asarray(theirs[umep_idx], dtype=np.float64)
        np.testing.assert_allclose(
            got,
            want,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"{attr} diverges from UMEP at pixel_size={pixel_size}",
        )
