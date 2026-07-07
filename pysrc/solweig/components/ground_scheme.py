"""UMEP 2026a ground-surface scheme: per-run initialisation.

Builds the per-landcover parameter grids and initial state for the
force-restore/OHM surface temperature model
(:func:`solweig.rustalgos.ground.surface_temperature_calc`) and the
solid-angle outgoing longwave march. Ported from
``initiate_groundScheme`` in UMEP-processing's ``ground_surface.py``
(vendored at ``tests/reference/umep_2026a/``); parity gated by
``tests/spec/test_parity_2026a.py``.

The scheme is opt-in (``Settings.use_ground_scheme``) and requires a
land-cover grid restricted to the ground classes (0, 1, 2, 5, 6, 7);
wall-material codes (>= 100) are remapped to roofs, as upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ..errors import InvalidSurfaceData
from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)

# Seasonal phases of the initial-temperature sinusoids (upstream constants)
_PHI_TG = 1.6
_PHI_TM = 1.7


@dataclass
class GroundSchemeState:
    """Carried state + static parameter grids for the ground scheme.

    ``tg``, ``tm``, ``rn``, ``rn_past``, ``g`` evolve per timestep through
    :func:`solweig.rustalgos.ground.surface_temperature_calc`; the parameter
    grids are fixed for the run. ``shadow_past`` starts as ones (fully
    sunlit) and is replaced with the previous timestep's shadow grid by the
    caller.
    """

    tg: NDArray[np.float32]
    tm: NDArray[np.float32]
    rn: NDArray[np.float32]
    rn_past: NDArray[np.float32]
    g: NDArray[np.float32]
    cap: NDArray[np.float32]
    diff: NDArray[np.float32]
    a1: NDArray[np.float32]
    a2: NDArray[np.float32]
    a3: NDArray[np.float32]
    lc_grid: NDArray[np.float32]
    shadow_past: NDArray[np.float32]


def _block(materials: SimpleNamespace, name: str) -> dict:
    """Fetch a parameter block (names may contain spaces) as a dict."""
    ns = getattr(materials, name, None)
    if ns is None:
        ns = materials.__dict__.get(name)
    if ns is None:
        raise InvalidSurfaceData(
            f"Materials parameters are missing the '{name}' block required by the ground scheme",
            field=name,
        )
    payload = getattr(ns, "Value", None) or getattr(ns, "Values", None)
    return dict(vars(payload))


def initiate_ground_scheme(
    land_cover: NDArray,
    materials: SimpleNamespace,
    day_of_year: int,
    ta_series: NDArray | list[float],
    latitude: float,
) -> GroundSchemeState:
    """Build parameter grids and initial state for the ground scheme.

    Args:
        land_cover: Land-cover class grid (ground classes 0/1/2/5/6/7; wall
            material codes >= 100 are remapped to roofs, matching upstream).
        materials: Loaded parameter namespace (:func:`solweig.load_params`).
        day_of_year: Day of year of the first timestep.
        ta_series: Air temperatures (deg C) for the first simulated day;
            upstream uses the first value and the daily mean for the initial
            surface and deep-soil temperatures.
        latitude: Site latitude in degrees (drives the seasonal sinusoids
            and their hemisphere sign).

    Returns:
        A :class:`GroundSchemeState` ready for the first timestep.

    Raises:
        InvalidSurfaceData: If the land-cover grid contains a class the
            scheme has no parameters for.
    """
    names = _block(materials, "Names")
    heat_capacity = _block(materials, "Heat capacity")
    diffusivity = _block(materials, "Thermal_diffusivity")
    ohm = _block(materials, "OHM_coefficients")
    tg_ini = _block(materials, "Tg_ini coefficients")
    tm_ini = _block(materials, "Tm_ini coefficients")

    ta = np.asarray(ta_series, dtype=np.float64)
    ta_first = float(ta.flat[0])
    ta_mean = float(np.mean(ta))
    lat_sign = float(np.sign(latitude))
    season = 2.0 * np.pi / 365.25 * day_of_year

    # Upstream mutates the caller's grid here; we copy.
    lc = np.asarray(land_cover, dtype=np.float32).copy()
    lc[lc >= 100] = 2.0
    class_ids = np.unique(lc).astype(int)

    cap_grid = lc.copy()
    diff_grid = lc.copy()
    a1_grid = lc.copy()
    a2_grid = lc.copy()
    a3_grid = lc.copy()
    tg = lc.copy()
    tm = lc.copy()

    for i in class_ids:
        key = str(int(i))
        if key not in names:
            raise InvalidSurfaceData(
                f"Land-cover class {i} has no entry in the materials Names table",
                field="land_cover",
                got=str(i),
            )
        name = names[key]
        if name not in ohm:
            raise InvalidSurfaceData(
                f"Land-cover class {i} ({name}) has no OHM coefficients — the ground "
                "scheme supports ground classes 0/1/2/5/6/7 only",
                field="land_cover",
                got=name,
            )
        mask = lc == i

        cap_grid[mask] = heat_capacity[name]
        diff_grid[mask] = diffusivity[name]

        mean_a1, phi_a1, a2_val, a3_val = ohm[name]
        a1_grid[mask] = mean_a1 * (1.0 + 0.33 * np.sin(season + phi_a1) * lat_sign)
        a2_grid[mask] = a2_val
        a3_grid[mask] = a3_val

        offset_tg, slope_tg, ratio_tg = tg_ini[name]
        offset_tg = offset_tg + slope_tg * latitude
        ampl_tm, slope_tm, offset_tm = tm_ini[name]
        offset_tm = offset_tm + slope_tm * latitude

        tg_seasonal = ta_first + offset_tg * (1.0 + ratio_tg * np.sin(season + _PHI_TG) * lat_sign)
        tm_seasonal = ta_mean + ampl_tm * np.sin(season + _PHI_TM) * lat_sign + offset_tm

        if i in (0, 1):
            tg[mask] = tg_seasonal + 4.0
            tm[mask] = tm_seasonal + 4.0
        elif i == 2:
            tg[mask] = tg_seasonal + 4.0
            tm[mask] = ta_mean + offset_tm
        elif i == 5:
            tg[mask] = tg_seasonal
            tm[mask] = tm_seasonal
        elif i == 6:
            tg[mask] = tg_seasonal + 2.0
            tm[mask] = tm_seasonal + 2.0
        elif i == 7:
            tg[mask] = ta_first
            # Upstream leaves Tm at the raw land-cover code for water (7.0);
            # reproduced for parity, and harmless: the slab model relaxes Tg
            # toward Tm slowly relative to the radiative terms.
            tm[mask] = float(i)

    shape = lc.shape
    logger.info(f"  Ground scheme initialised: classes {sorted(int(c) for c in class_ids)}, day {day_of_year}")
    return GroundSchemeState(
        tg=tg.astype(np.float32),
        tm=tm.astype(np.float32),
        rn=np.zeros(shape, dtype=np.float32),
        rn_past=np.zeros(shape, dtype=np.float32),
        g=np.zeros(shape, dtype=np.float32),
        cap=cap_grid.astype(np.float32),
        diff=diff_grid.astype(np.float32),
        a1=a1_grid.astype(np.float32),
        a2=a2_grid.astype(np.float32),
        a3=a3_grid.astype(np.float32),
        lc_grid=lc,
        shadow_past=np.ones(shape, dtype=np.float32),
    )
