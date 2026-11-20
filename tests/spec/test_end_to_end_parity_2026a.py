"""End-to-end parity of short runs with the 2026a ground scheme against the reference.

``tests/spec/test_end_to_end_parity.py`` compares one timestep with the scheme
off. This module does the same over several timesteps with the scheme on
(``use_ground_scheme`` and ``use_outgoing_longwave``), so that it also sees the
state carried from one timestep to the next.

One synthetic scene (40 x 40 px at 1 m, one 12 m block, no vegetation,
isotropic sky, standing posture, land cover with asphalt, grass, bare soil,
water and the building) is run for two hourly series on one July day:

- ``morning``: 03:00 to 08:00. Two night steps, then four sunlit steps.
- ``evening``: 18:00 to 23:00, overcast. Four sunlit steps, then two night steps.

The package side follows the timestep loop of ``solweig.timeseries``:
``initiate_ground_scheme`` once, then ``solweig.api._calculate_single`` per
timestep with the same thermal state and scheme state.

The reference side calls ``Solweig_2026a_calc`` of umep-solweig (see
``tests/reference_package.py``) once per timestep. The reference runner
(``Solweig_run.py`` next to it) cannot be imported without GDAL, so the loop
here reproduces it: every argument is derived the way the runner derives it,
from the parameter file bundled with this package, and every value the runner
carries between iterations is taken from the tuple the function returned, as
in the runner.

Three groups of inputs are shared instead of derived twice, for the reasons
given in ``test_end_to_end_parity.py``: the sun position, the direct and
diffuse split and the daily maximum sun altitude (package ``Weather``
objects), the sky view factors (computed by the package), and the wall height
and aspect rasters.

The land cover has no class 0. With ``demforbuild = 0`` the runner builds its
building mask from the land-cover grid and leaves class 0 at zero, which is
the building value (l.401-408); the package treats class 0 as ground.

Each series lies within one day. The package starts the deep-soil temperature
from the mean air temperature of the first simulated day and the runner from
the mean of the whole met series; the two agree for one day.

Tolerances: 0.1 W/m2 on the fluxes, 0.05 degC on Tmrt and on the ground
temperature, 1e-3 on the shadow mask. The reference converts Tmrt with 273.2
and this package with 273.15; the 0.05 degC offset is removed from the
reference before comparing.

Kdown, Kup, Ldown, Lup, the shadow mask and the ground temperature agree at
every timestep of both series. Two differences remain in Tmrt, recorded as
strict xfails (measured 2026-11-20). Neither is specific to the ground scheme:
both sit in code that the path without the scheme also runs.

1. Clearness index above one. The reference function limits the clearness
   index to one in daytime (``Solweig_2026a_calc``, ``if (CI > 1) or (CI ==
   np.inf): CI = 1``) before ``Lside_veg_v2026`` weights the sky longwave with
   ``1 - CI``. The package ``Weather.clearness_index`` keeps the value that
   ``clearnessindex_2013b`` returns (1.026 and 1.037 at 07:00 and 08:00 of the
   morning series), and ``lside_veg_variant_pure`` uses it as it is. Tmrt is up
   to 0.16 degC lower in the package.
2. Wall shade fraction below 3 degrees of sun altitude. The package
   ``cylindric_wedge_pure_masked`` returns one for every pixel when the sun is
   lower than 3 degrees; the reference ``cylindric_wedge`` has no such limit
   and gives 0.75 to 0.97 on this scene at 21:00 of the evening series (sun at
   2.4 degrees). Tmrt differs by up to 0.27 degC next to the walls.

``test_rest_of_the_chain_matches_with_the_clearness_index_limited`` limits the
package clearness index to one, and
``test_rest_of_the_chain_matches_with_the_low_sun_wall_shade_fraction`` gives
the reference the package's wall shade fraction below 3 degrees. Each shows
that nothing else differs in its series.

``test_previous_shadow_start_value_is_seen`` starts the reference shadow grid
at one instead of zero and checks that the comparison then fails, so a wrong
start value on either side does not pass unnoticed.
"""

from __future__ import annotations

import functools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import solweig
from solweig import HumanParams, Location, SurfaceData, Weather
from solweig.api import _calculate_single
from solweig.components.ground_scheme import initiate_ground_scheme
from solweig.components.gvf import detect_building_mask
from solweig.io import load_raster
from solweig.loaders import load_params
from solweig.models import ThermalState
from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls
from solweig.timeseries import _precompute_weather

from tests.reference_package import reference_module

reference = reference_module("functions.SOLWEIGpython.Solweig_2026a_calc_forprocessing")
reference_ground = reference_module("functions.SOLWEIGpython.ground_surface")
reference_tgmaps = reference_module("functions.SOLWEIGpython.Tgmaps_v1")

SHAPE = (40, 40)
PIXEL_SIZE = 1.0
BUILDING = (slice(15, 25), slice(15, 25))
BUILDING_HEIGHT = 12.0
DAY = (2024, 7, 15)
LATITUDE, LONGITUDE, UTC_OFFSET = 57.7, 12.0, 1
TIMESTEP_DEC = 1.0 / 24.0  # hourly run: dectime[1] - dectime[0] in the reference runner
TIMESTEP_SECONDS = 3600  # second minus first met row in the reference runner (l.690-702)

# Hour label, air temperature (degC), relative humidity (%), global radiation (W/m2).
SERIES: dict[str, tuple[tuple[int, float, float, float], ...]] = {
    "morning": (
        (3, 14.0, 75.0, 0.0),
        (4, 14.0, 75.0, 0.0),
        (5, 14.5, 72.0, 40.0),
        (6, 16.0, 66.0, 150.0),
        (7, 17.5, 60.0, 290.0),
        (8, 19.0, 55.0, 430.0),
    ),
    "evening": (
        (18, 21.0, 55.0, 150.0),
        (19, 20.0, 58.0, 90.0),
        (20, 19.0, 62.0, 40.0),
        (21, 18.0, 66.0, 8.0),
        (22, 17.0, 70.0, 0.0),
        (23, 16.5, 72.0, 0.0),
    ),
}
N_STEPS = 6

LOW_SUN_LIMIT_DEG = 3.0  # MIN_SUN_ELEVATION_RAD in rust/src/sky.rs

FLUX_ATOL = 0.1  # W/m2
TEMPERATURE_ATOL = 0.05  # degC
SHADOW_ATOL = 1e-3
KELVIN_OFFSET_DIFFERENCE = 273.2 - 273.15  # reference minus package, specs/tmrt.md

PARAM_FILE = Path(solweig.__file__).parent / "data" / "default_materials.json"

FLUXES = ("tmrt", "kdown", "kup", "ldown", "lup", "shadow")
OUTPUTS = (*FLUXES, "tg")
ATOL = {
    "tmrt": TEMPERATURE_ATOL,
    "kdown": FLUX_ATOL,
    "kup": FLUX_ATOL,
    "ldown": FLUX_ATOL,
    "lup": FLUX_ATOL,
    "shadow": SHADOW_ATOL,
    "tg": TEMPERATURE_ATOL,
}


def _dsm() -> np.ndarray:
    dsm = np.zeros(SHAPE, dtype=np.float32)
    dsm[BUILDING] = BUILDING_HEIGHT
    return dsm


def _land_cover() -> np.ndarray:
    """Dark asphalt, with grass, bare soil and water patches and the building."""
    lc = np.ones(SHAPE, dtype=np.uint8)  # 1 = dark asphalt
    lc[:, 0:10] = 5  # grass, west strip
    lc[30:40, 24:40] = 6  # bare soil, south-east
    lc[0:7, 28:40] = 7  # water, north-east
    lc[BUILDING] = 2
    return lc


def _ground_mask() -> np.ndarray:
    """Ground pixels, without the building and without a one-pixel border."""
    mask = _dsm() == 0.0
    mask[0, :] = mask[-1, :] = False
    mask[:, 0] = mask[:, -1] = False
    return mask


@functools.cache
def _scene():
    """Surface with land cover and real SVF, and the location."""
    dsm = _dsm()
    walls = findwalls(dsm, PIXEL_SIZE)
    aspect = filter1Goodwin_as_aspect_v3(walls, PIXEL_SIZE, dsm)
    surface = SurfaceData(
        dsm=dsm, pixel_size=PIXEL_SIZE, land_cover=_land_cover(), wall_height=walls, wall_aspect=aspect
    )
    solweig.disable_gpu()
    try:
        surface.compute_svf()
    finally:
        solweig.enable_gpu()
    location = Location(latitude=LATITUDE, longitude=LONGITUDE, utc_offset=UTC_OFFSET)
    return surface, location


def _new_weather(series: str) -> list[Weather]:
    return [
        Weather(datetime=datetime(*DAY, hour, 0), ta=ta, rh=rh, global_rad=rad) for hour, ta, rh, rad in SERIES[series]
    ]


@functools.cache
def _weather(series: str, limit_clearness_index: bool = False) -> tuple[Weather, ...]:
    """Weather objects of one series, with sun position and radiation split.

    ``limit_clearness_index`` limits the clearness index to one, as the
    reference function does in daytime.
    """
    _, location = _scene()
    weather = _new_weather(series)
    _precompute_weather(weather, location)
    if limit_clearness_index:
        for w in weather:
            w.clearness_index = min(w.clearness_index, 1.0)
    return tuple(weather)


@functools.cache
def _package_side(series: str, limit_clearness_index: bool = False) -> tuple[dict[str, np.ndarray], ...]:
    """One series through the package, as the timestep loop of ``solweig.timeseries`` runs it.

    CPU is the parity path.
    """
    surface, location = _scene()
    weather = _weather(series, limit_clearness_index)
    materials = load_params()
    state = ThermalState.initial(surface.shape)
    state.timestep_dec = TIMESTEP_DEC
    first_day = weather[0].datetime.date()
    assert surface.land_cover is not None
    scheme_state = initiate_ground_scheme(
        surface.land_cover,
        materials,
        weather[0].datetime.timetuple().tm_yday,
        [w.ta for w in weather if w.datetime.date() == first_day],
        location.latitude,
    )
    steps = []
    solweig.disable_gpu()
    try:
        for w in weather:
            result = _calculate_single(
                surface=surface,
                location=location,
                weather=w,
                human=HumanParams(posture="standing"),
                use_anisotropic_sky=False,
                state=state,
                materials=materials,
                ground_scheme_state=scheme_state,
                return_state_copy=False,
                _requested_outputs=set(FLUXES),
            )
            if result.state is not None:
                state = result.state
            step = {name: np.asarray(getattr(result, name), dtype=np.float64).copy() for name in FLUXES}
            step["tg"] = np.asarray(scheme_state.tg, dtype=np.float64).copy()
            steps.append(step)
    finally:
        solweig.enable_gpu()
    return tuple(steps)


def _reference_static_inputs() -> dict:
    """Arguments of ``Solweig_2026a_calc`` that the reference runner sets before its loop.

    Line numbers refer to umep-solweig ``functions/SOLWEIGpython/Solweig_run.py``.
    """
    surface, _ = _scene()
    param = json.loads(PARAM_FILE.read_text())
    rows, cols = SHAPE
    dsm = _dsm().astype(float)
    lcgrid = _land_cover().astype(float)  # l.130-134

    # l.74-79: wall optics and absorption coefficients
    albedo_b = param["Albedo"]["Effective"]["Value"]["Walls"]
    ewall = param["Emissivity"]["Value"]["Walls"]
    absK = param["Tmrt_params"]["Value"]["absK"]
    absL = param["Tmrt_params"]["Value"]["absL"]

    # l.103, l.271: location, with the median DSM height as altitude
    location = {"longitude": LONGITUDE, "latitude": LATITUDE, "altitude": float(np.median(dsm))}

    # l.159-183: building sky view factors. Shared with the package (own parity gates).
    svfs = surface.svf
    svf = np.asarray(svfs.svf, dtype=float)
    # l.238-248: without vegetation every vegetation sky view factor is one
    ones = np.ones((rows, cols))
    # l.250-253
    tmp = svf + ones - 1.0
    tmp[tmp < 0.0] = 0.0
    svfalfa = np.arcsin(np.exp(np.log(1.0 - tmp) / 2.0))

    # l.327-339: posture factors for the standing posture
    Fside = param["Posture"]["Standing"]["Value"]["Fside"]
    Fup = param["Posture"]["Standing"]["Value"]["Fup"]
    Fcyl = param["Posture"]["Standing"]["Value"]["Fcyl"]
    height = param["Posture"]["Standing"]["Value"]["height"]
    # l.341-345: radiative source area, Schmid et al. (1990)
    first = np.round(height)
    if first == 0.0:
        first = 1.0
    second = np.round(height * 20.0)

    # l.401-408: buildings from the land-cover grid (demforbuild = 0)
    buildings = np.copy(lcgrid)
    for ground_class in (7, 6, 5, 4, 3):
        buildings[buildings == ground_class] = 1
    buildings[buildings == 2] = 0

    # l.469-480: ground and wall temperature parameters from the land-cover grid
    TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall = reference_tgmaps.Tgmaps_v1(
        lcgrid.copy(), param
    )

    return {
        "param": param,
        "lcgrid": lcgrid,
        "location": location,
        "before_altitude": {
            "dsm": dsm,
            "scale": 1.0 / PIXEL_SIZE,
            "rows": rows,
            "cols": cols,
            "svf": svf,
            "svfN": np.asarray(svfs.svf_north, dtype=float),
            "svfW": np.asarray(svfs.svf_west, dtype=float),
            "svfE": np.asarray(svfs.svf_east, dtype=float),
            "svfS": np.asarray(svfs.svf_south, dtype=float),
            "svfveg": ones,
            "svfNveg": ones,
            "svfEveg": ones,
            "svfSveg": ones,
            "svfWveg": ones,
            "svfaveg": ones,
            "svfEaveg": ones,
            "svfSaveg": ones,
            "svfWaveg": ones,
            "svfNaveg": ones,
            "vegdem": 0,  # l.126-127
            "vegdem2": 0,
            "albedo_b": albedo_b,
            "absK": absK,
            "absL": absL,
            "ewall": ewall,
            "Fside": Fside,
            "Fup": Fup,
            "Fcyl": Fcyl,
        },
        "buildings": buildings,
        # l.255-260: wall rasters, read from files by the runner. Shared with the package.
        "dirwalls": np.asarray(surface.wall_aspect, dtype=float),
        "walls": np.asarray(surface.wall_height, dtype=float),
        "ground": {
            "TgK": TgK,
            "Tstart": Tstart,
            "alb_grid": alb_grid,
            "emis_grid": emis_grid,
            "TgK_wall": TgK_wall,
            "Tstart_wall": Tstart_wall,
            "TmaxLST": TmaxLST,
            "TmaxLST_wall": TmaxLST_wall,
            "first": first,
            "second": second,
            "svfalfa": svfalfa,
            "svfbuveg": svf,  # l.388: no vegetation
        },
    }


def _wall_shade_fraction_as_package(original):
    """``cylindric_wedge`` with the package's rule below 3 degrees of sun altitude."""

    def guarded(zen, svfalfa, rows, cols):
        if np.pi / 2.0 - zen < np.deg2rad(LOW_SUN_LIMIT_DEG):
            return np.ones((rows, cols))
        return original(zen, svfalfa, rows, cols)

    return guarded


@functools.cache
def _reference_side(
    series: str, shadow_start: float = 0.0, low_sun_wall_shade_as_package: bool = False
) -> tuple[dict[str, np.ndarray], ...]:
    """One series through the reference function, with the loop of the reference runner.

    ``shadow_start`` is the fill value of the shadow grid before the first
    timestep. The runner uses zero (l.466). ``low_sun_wall_shade_as_package``
    replaces the wall shade fraction with one below 3 degrees of sun altitude,
    to test everything else.
    """
    with pytest.MonkeyPatch.context() as patch, np.errstate(divide="ignore", invalid="ignore"):
        if low_sun_wall_shade_as_package:
            patch.setattr(reference, "cylindric_wedge", _wall_shade_fraction_as_package(reference.cylindric_wedge))
        return _reference_loop(series, shadow_start)


def _reference_loop(series: str, shadow_start: float) -> tuple[dict[str, np.ndarray], ...]:
    static = _reference_static_inputs()
    weather = _weather(series)
    param = static["param"]
    lcgrid = static["lcgrid"]
    location = static["location"]
    rows, cols = SHAPE

    # Met columns as the runner reads them (l.276-285)
    doy = np.array([w.datetime.timetuple().tm_yday for w in weather], dtype=float)
    Ta = np.array([w.ta for w in weather], dtype=float)
    # dectime as in the met-file reader: day of year plus the fraction of the day
    dectime = np.array(
        [d + w.datetime.hour / 24.0 + w.datetime.minute / 1440.0 for d, w in zip(doy, weather, strict=True)]
    )

    # l.497-531: scheme state and parameter grids, no measured surface temperature
    Tg, Tm, Rn, Rn_past, G, cap_grid, diff_grid, a1_grid, a2_grid, a3_grid = reference_ground.initiate_groundScheme(
        lcgrid.copy(), param, doy[0], Ta, location
    )

    # State the runner sets before the loop and then takes from the returned tuple
    shadow = np.zeros((rows, cols)) + shadow_start  # l.466
    Tgmap1 = np.zeros((rows, cols))  # l.394-398
    Tgmap1E = np.zeros((rows, cols))
    Tgmap1S = np.zeros((rows, cols))
    Tgmap1W = np.zeros((rows, cols))
    Tgmap1N = np.zeros((rows, cols))
    timestepdec = dectime[1] - dectime[0]  # l.629
    timeadd = 0.0  # l.630
    firstdaytime = 1.0  # l.631
    CI = 1.0  # l.641
    steradians = 0  # l.465
    voxelTable = 0  # l.620
    Twater: float | list = []  # l.265

    steps = []
    for i, w in enumerate(weather):
        # l.665-667: daily water temperature, set on the first row and at midnight
        if (dectime[i] - np.floor(dectime[i])) == 0 or i == 0:
            Twater = float(np.mean(Ta[doy == np.floor(dectime[i])]))
        # l.669-687: the nocturnal clearness index is set at midnight only; no series has a midnight row
        assert (dectime[i] - np.floor(dectime[i])) != 0

        args = {
            "i": i,
            **static["before_altitude"],
            # Sun position and altmax: read from the package Weather objects (own parity tests)
            "altitude": float(w.sun_altitude),
            "azimuth": float(w.sun_azimuth),
            "zen": float(w.sun_zenith) * np.pi / 180.0,
            "jday": int(doy[i]),
            "usevegdem": 0,
            "onlyglobal": 0,  # direct and diffuse are passed in
            "buildings": static["buildings"],
            "location": location,
            "psi": 1.0,  # l.387
            "landcover": 1,
            "lc_grid": lcgrid,
            "dectime": dectime[i],
            "altmax": float(w.altmax),
            "dirwalls": static["dirwalls"],
            "walls": static["walls"],
            "cyl": 1,  # runner setting (l.73)
            "elvis": 0.0,  # l.77
            "Ta": float(Ta[i]),
            "RH": float(w.rh),
            "radG": float(w.global_rad),
            "radD": float(w.diffuse_rad),
            "radI": float(w.direct_rad),
            "P": float(w.pressure) / 10.0,  # met files carry kPa; the function multiplies by 10
            "amaxvalue": 0,  # l.390
            "bush": np.zeros((rows, cols)),  # l.389
            "Twater": Twater,
            **static["ground"],
            "firstdaytime": firstdaytime,
            "timeadd": timeadd,
            "timestepdec": timestepdec,
            "Tgmap1": Tgmap1,
            "Tgmap1E": Tgmap1E,
            "Tgmap1S": Tgmap1S,
            "Tgmap1W": Tgmap1W,
            "Tgmap1N": Tgmap1N,
            "CI": CI,
            "diffsh": None,  # l.459-465: isotropic sky
            "shmat": None,
            "vegshmat": None,
            "vbshvegshmat": None,
            "anisotropic_sky": 0,
            "asvf": None,
            "patch_option": 0,
            "voxelMaps": 0,  # l.618-623: no wall temperature scheme
            "voxelTable": voxelTable,
            "ws": float(w.ws),
            "wallScheme": 0,
            "timeStep": TIMESTEP_SECONDS,
            "steradians": steradians,
            "walls_scheme": np.ones((rows, cols)) * 10.0,
            "dirwalls_scheme": np.ones((rows, cols)) * 10.0,
            "groundScheme": 1,  # l.495
            "outgoingLW": 1,  # l.546
            "Tg": Tg,
            "Rn": Rn,
            "Rn_past": Rn_past,
            "G": G,
            "Tm": Tm,
            "cap_grid": cap_grid,
            "diff_grid": diff_grid,
            "a1_grid": a1_grid,
            "a2_grid": a2_grid,
            "a3_grid": a3_grid,
            "shadow_past": shadow,
        }
        out = reference.Solweig_2026a_calc(**args)
        # l.704-747: the returned tuple, in the order the runner unpacks it
        tmrt, kdown, kup, ldown, lup, Tg = out[0:6]
        CI, shadow, firstdaytime, timestepdec, timeadd = out[9:14]
        Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N = out[14:19]
        steradians, voxelTable, Rn, Rn_past, Tm, G = out[36:42]
        steps.append(
            {
                "tmrt": np.asarray(tmrt, dtype=np.float64) + KELVIN_OFFSET_DIFFERENCE,
                "kdown": np.array(kdown, dtype=np.float64),
                "kup": np.array(kup, dtype=np.float64),
                "ldown": np.array(ldown, dtype=np.float64),
                "lup": np.array(lup, dtype=np.float64),
                "shadow": np.array(shadow, dtype=np.float64),
                "tg": np.array(Tg, dtype=np.float64),  # the function updates Tg in place
            }
        )
    return tuple(steps)


def _difference(package: dict[str, np.ndarray], ref: dict[str, np.ndarray], name: str) -> np.ndarray:
    return np.abs(package[name] - ref[name])[_ground_mask()]


def _assert_within(package: dict[str, np.ndarray], ref: dict[str, np.ndarray], name: str, label: str) -> None:
    diff = _difference(package, ref, name)
    assert np.all(np.isfinite(diff)), f"{name}: non-finite values over ground pixels"
    assert diff.max() <= ATOL[name], (
        f"{name} ({label}): max abs diff {diff.max():.4f}, mean {diff.mean():.4f}, tolerance {ATOL[name]}"
    )


def _cases(series_names: tuple[str, ...], known: dict[tuple[str, int, str], str]) -> list:
    """All series, timestep and output triples, with a strict xfail on the known differences."""
    cases = []
    for series in series_names:
        for step, (hour, *_) in enumerate(SERIES[series]):
            for name in OUTPUTS:
                marks = ()
                if (series, hour, name) in known:
                    marks = (pytest.mark.xfail(strict=True, reason=known[(series, hour, name)]),)
                cases.append(pytest.param(series, step, name, marks=marks, id=f"{series}-{hour:02d}h-{name}"))
    return cases


CLEARNESS = "clearness index above one (package) against limited to one (reference function)"
LOW_SUN = "wall shade fraction of one below 3 degrees of sun altitude (package) against computed (reference)"

# Keyed by series, hour label and output.
AS_RUNNER = {
    ("morning", 7, "tmrt"): f"max 0.11, mean 0.08 degC: {CLEARNESS}",
    ("morning", 8, "tmrt"): f"max 0.16, mean 0.11 degC: {CLEARNESS}",
    ("evening", 21, "tmrt"): f"max 0.27, mean 0.03 degC: {LOW_SUN}",
}


@pytest.mark.parametrize("series", list(SERIES))
def test_each_series_has_night_steps_and_sunlit_steps_with_shade(series):
    """Guard against a degenerate comparison (no night, no sun, or no shadow on the ground)."""
    ref = _reference_side(series)
    ground = _ground_mask()
    night = [step for step, w in zip(ref, _weather(series), strict=True) if w.sun_altitude <= 0]
    day = [step for step, w in zip(ref, _weather(series), strict=True) if w.sun_altitude > 0]
    assert len(SERIES[series]) == N_STEPS
    assert len(night) == 2 and len(day) == 4
    for step in night:
        assert step["kdown"][ground].max() == 0.0
        assert step["shadow"][ground].max() == 0.0
    for step in day:
        assert step["kdown"][ground].min() > 0.0
        assert 0.02 < np.mean(step["shadow"][ground] < 0.5) < 0.5
    # The ground temperature is a state that moves: more than 1 degC over the series.
    mean_tg = [step["tg"][ground].mean() for step in ref]
    assert max(mean_tg) - min(mean_tg) > 1.0


def test_the_morning_series_starts_at_night_and_the_evening_series_ends_at_night():
    assert [w.sun_altitude > 0 for w in _weather("morning")] == [False, False, True, True, True, True]
    assert [w.sun_altitude > 0 for w in _weather("evening")] == [True, True, True, True, False, False]


def test_parameter_file_has_the_layout_the_reference_reads():
    """``Tg_ini`` and ``Tm_ini`` in another layout start the deep soil about 100 K off."""
    static = _reference_static_inputs()
    lcgrid = static["lcgrid"]
    ta = np.array([row[1] for row in SERIES["morning"]])
    day_of_year = float(datetime(*DAY).timetuple().tm_yday)
    tg, tm, *_ = reference_ground.initiate_groundScheme(
        lcgrid.copy(), static["param"], day_of_year, ta, static["location"]
    )
    soil = np.isin(lcgrid, (1, 5, 6))
    assert np.abs(tg[soil] - ta[0]).max() < 15.0
    assert np.abs(tm[soil] - ta.mean()).max() < 15.0


def test_building_mask_is_the_runner_mask():
    surface, _ = _scene()
    package = detect_building_mask(surface.dsm, surface.land_cover, surface.wall_height, PIXEL_SIZE)
    np.testing.assert_array_equal(package, _reference_static_inputs()["buildings"])


def test_ground_albedo_and_emissivity_are_the_runner_grids():
    """The two land-cover grids the scheme reads, from ``Tgmaps_v1`` on the reference side."""
    surface, _ = _scene()
    alb, emis, *_ = surface.get_land_cover_properties(load_params())
    runner = _reference_static_inputs()["ground"]
    np.testing.assert_allclose(alb, runner["alb_grid"], atol=1e-6)
    np.testing.assert_allclose(emis, runner["emis_grid"], atol=1e-6)


@pytest.mark.parametrize(("series", "step", "name"), _cases(tuple(SERIES), AS_RUNNER))
def test_each_timestep_matches_the_reference(series, step, name):
    """Package against the reference with every input and carried value derived as in the reference runner."""
    _assert_within(_package_side(series)[step], _reference_side(series)[step], name, f"{series}, step {step}")


@pytest.mark.parametrize(("series", "step", "name"), _cases(("morning",), {}))
def test_rest_of_the_chain_matches_with_the_clearness_index_limited(series, step, name):
    """Same comparison with the package clearness index limited to one."""
    package = _package_side(series, limit_clearness_index=True)[step]
    _assert_within(package, _reference_side(series)[step], name, f"{series}, step {step}")


@pytest.mark.parametrize(("series", "step", "name"), _cases(("evening",), {}))
def test_rest_of_the_chain_matches_with_the_low_sun_wall_shade_fraction(series, step, name):
    """Same comparison with the package's wall shade fraction below 3 degrees on both sides."""
    ref = _reference_side(series, low_sun_wall_shade_as_package=True)[step]
    _assert_within(_package_side(series)[step], ref, name, f"{series}, step {step}")


@pytest.mark.parametrize("name", ["tg", "lup", "tmrt"])
def test_previous_shadow_start_value_is_seen(name):
    """A reference shadow grid that starts at one moves the first night step out of tolerance.

    The scheme damps the ground heat flux where the shadow value changed by
    more than 0.5 since the previous timestep. The runner starts the grid at
    zero (l.466), so a run that starts at night has no change on its first
    step.
    """
    package = _package_side("morning")[0]
    assert _difference(package, _reference_side("morning")[0], name).max() <= ATOL[name]
    assert _difference(package, _reference_side("morning", shadow_start=1.0)[0], name).max() > 5 * ATOL[name]


@pytest.mark.parametrize("series", list(SERIES))
def test_public_calculate_writes_what_the_timestep_loop_returns(series, tmp_path):
    """``solweig.calculate`` with both scheme flags gives the rasters of ``_package_side``."""
    surface, location = _scene()
    solweig.disable_gpu()
    try:
        solweig.calculate(
            surface,
            _new_weather(series),
            location,
            output_dir=tmp_path,
            outputs=list(FLUXES),
            use_anisotropic_sky=False,
            use_ground_scheme=True,
            use_outgoing_longwave=True,
        )
    finally:
        solweig.enable_gpu()
    loop = _package_side(series)
    for name in FLUXES:
        rasters = sorted((tmp_path / name).glob("*.tif"))
        assert len(rasters) == N_STEPS
        for step, raster in zip(loop, rasters, strict=True):
            written, *_ = load_raster(str(raster))
            np.testing.assert_array_equal(np.asarray(written, dtype=np.float64), step[name])
