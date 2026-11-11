"""End-to-end parity of one timestep against the reference per-timestep function.

The other parity tests compare single Rust functions with their reference
Python counterparts, and the golden tests call the Rust functions with inputs
written in the test. Neither sees what ``solweig.computation`` feeds the Rust
pipeline. Two bugs passed through that gap: the ground view factor was given
the stature where the reference uses the posture height (fixed in 0.1.0b100),
and anisotropic building patches were routed to the vegetation emitter when
the surface had no vegetation (fixed in 0.1.0b97).

Here one synthetic scene (40 x 40 px at 1 m, one 12 m block, no vegetation, no
land-cover grid, isotropic sky, first daytime step of an hourly run) goes
through ``solweig.api._calculate_single`` and through ``Solweig_2025a_calc``
of the ``umep`` package. Every argument of the reference function is derived
the way the reference runner (``Solweig_run.py`` next to it) derives it, from
the parameter file bundled with this package, and not from this package's
Python layer.

Three groups of inputs are shared instead of derived twice, because each has
its own parity gate or is an input file on the reference side:

- the sun position, the direct and diffuse split and the daily maximum sun
  altitude are read from the package ``Weather`` object;
- the sky view factors are the ones the package computed for the surface;
- the wall height and aspect rasters are the ones given to the package (the
  reference runner reads both from files written by the UMEP wall tool).

Tolerances: 0.1 W/m2 on the fluxes and 0.05 degC on Tmrt (float32 pipeline
against a float64 reference), 1e-3 on the 0/1 shadow mask. The reference
converts Tmrt with 273.2 and this package with 273.15 (documented in
``specs/tmrt.md``, "Some legacy implementations used -273.2"); the 0.05 degC
offset is removed from the reference before comparing.

Findings recorded here as strict xfails (measured 2026-11-11, 0.1.0b100):

1. Ground albedo without a land-cover grid: 0.15 in
   ``SurfaceData.get_land_cover_properties``, 0.20 (``Cobble_stone_2014a``) in
   the reference runner. Kup is 25 percent lower.
2. ``Tstart`` without a land-cover grid: -3.41 in the package, +3.41 in the
   reference runner, which subtracts the parameter (``Knight - param``) on this
   path only (its land-cover path, ``Tgmaps_v1``, uses the parameter as is).
   Sunlit Lup is about 35 W/m2 lower.
3. Building mask without a land-cover grid: ``detect_building_mask`` dilates
   the wall pixels by 25 m, so every pixel within 25 m of a wall is treated as
   a building by the ground view factor (the whole of this grid). The
   reference runner uses ``dsm - dem >= 2``.
4. Sitting posture: the Rust pipeline switches to the box model
   (``cyl = human.is_standing``); in the reference runner ``cyl`` is a setting
   independent of the posture.

``test_rest_of_the_chain_matches_with_the_known_inputs_aligned`` gives the
reference the package's values for findings 1 to 3 and shows that nothing else
differs for the standing posture.
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
from solweig.components.gvf import detect_building_mask
from solweig.models import ThermalState
from solweig.physics.wallalgorithms import filter1Goodwin_as_aspect_v3, findwalls
from solweig.timeseries import _precompute_weather

reference = pytest.importorskip("umep.functions.SOLWEIGpython.Solweig_2025a_calc_forprocessing")

SHAPE = (40, 40)
PIXEL_SIZE = 1.0
BUILDING = (slice(15, 25), slice(15, 25))
BUILDING_HEIGHT = 12.0
WHEN = datetime(2024, 7, 15, 13, 0)
TA, RH, GLOBAL_RAD = 25.0, 50.0, 700.0
LATITUDE, LONGITUDE, UTC_OFFSET = 57.7, 12.0, 1
TIMESTEP_DEC = 1.0 / 24.0  # hourly run: dectime[1] - dectime[0] in the reference runner

FLUX_ATOL = 0.1  # W/m2
TMRT_ATOL = 0.05  # degC
SHADOW_ATOL = 1e-3
KELVIN_OFFSET_DIFFERENCE = 273.2 - 273.15  # reference minus package, specs/tmrt.md

PARAM_FILE = Path(solweig.__file__).parent / "data" / "default_materials.json"

OUTPUTS = ("tmrt", "kdown", "kup", "ldown", "lup", "shadow")
ATOL = {
    "tmrt": TMRT_ATOL,
    "kdown": FLUX_ATOL,
    "kup": FLUX_ATOL,
    "ldown": FLUX_ATOL,
    "lup": FLUX_ATOL,
    "shadow": SHADOW_ATOL,
}


def _dsm() -> np.ndarray:
    dsm = np.zeros(SHAPE, dtype=np.float32)
    dsm[BUILDING] = BUILDING_HEIGHT
    return dsm


def _ground_mask() -> np.ndarray:
    """Ground pixels, without the building and without a one-pixel border."""
    mask = _dsm() == 0.0
    mask[0, :] = mask[-1, :] = False
    mask[:, 0] = mask[:, -1] = False
    return mask


@functools.cache
def _scene():
    """Surface with real SVF, weather with derived values, and location."""
    dsm = _dsm()
    walls = findwalls(dsm, PIXEL_SIZE)
    aspect = filter1Goodwin_as_aspect_v3(walls, PIXEL_SIZE, dsm)
    surface = SurfaceData(dsm=dsm, pixel_size=PIXEL_SIZE, wall_height=walls, wall_aspect=aspect)
    solweig.disable_gpu()
    try:
        surface.compute_svf()
    finally:
        solweig.enable_gpu()
    weather = Weather(datetime=WHEN, ta=TA, rh=RH, global_rad=GLOBAL_RAD)
    location = Location(latitude=LATITUDE, longitude=LONGITUDE, utc_offset=UTC_OFFSET)
    _precompute_weather([weather], location)
    return surface, weather, location


@functools.cache
def _package_side(posture: str) -> dict[str, np.ndarray]:
    """One timestep through the package entry point. CPU is the parity path."""
    surface, weather, location = _scene()
    state = ThermalState.initial(surface.shape)
    state.timestep_dec = TIMESTEP_DEC
    solweig.disable_gpu()
    try:
        result = _calculate_single(
            surface=surface,
            location=location,
            weather=weather,
            human=HumanParams(posture=posture),
            use_anisotropic_sky=False,
            state=state,
            return_state_copy=False,
            _requested_outputs=set(OUTPUTS),
        )
    finally:
        solweig.enable_gpu()
    return {name: np.asarray(getattr(result, name), dtype=np.float64).copy() for name in OUTPUTS}


def _reference_inputs(posture: str) -> dict:
    """Arguments of ``Solweig_2025a_calc``, derived as in the reference ``Solweig_run.py``.

    Line numbers refer to umep-solweig ``functions/SOLWEIGpython/Solweig_run.py``.
    The dict is in the positional order of the function.
    """
    surface, weather, _ = _scene()
    param = json.loads(PARAM_FILE.read_text())
    rows, cols = SHAPE
    dsm = _dsm().astype(float)
    dem = np.zeros_like(dsm)  # flat ground; the runner needs a DEM when no land cover is given

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
    svfN = np.asarray(svfs.svf_north, dtype=float)
    svfE = np.asarray(svfs.svf_east, dtype=float)
    svfS = np.asarray(svfs.svf_south, dtype=float)
    svfW = np.asarray(svfs.svf_west, dtype=float)
    # l.238-248: without vegetation every vegetation sky view factor is one
    ones = np.ones((rows, cols))
    # l.250-253
    tmp = svf + ones - 1.0
    tmp[tmp < 0.0] = 0.0
    svfalfa = np.arcsin(np.exp(np.log(1.0 - tmp) / 2.0))

    # l.255-260: wall rasters, read from files by the runner. Shared with the package.
    walls = np.asarray(surface.wall_height, dtype=float)
    dirwalls = np.asarray(surface.wall_aspect, dtype=float)

    # l.327-339: posture factors
    key = "Standing" if posture == "standing" else "Sitting"
    Fside = param["Posture"][key]["Value"]["Fside"]
    Fup = param["Posture"][key]["Value"]["Fup"]
    Fcyl = param["Posture"][key]["Value"]["Fcyl"]
    height = param["Posture"][key]["Value"]["height"]
    # l.341-345: radiative source area, Schmid et al. (1990)
    first = np.round(height)
    if first == 0.0:
        first = 1.0
    second = np.round(height * 20.0)

    # l.386-390: no vegetation
    psi = 1.0
    svfbuveg = svf
    bush = np.zeros((rows, cols))
    amaxvalue = 0

    # l.393-398
    Knight = np.zeros((rows, cols))

    # l.409-412: buildings from the height rasters (demforbuild = 1)
    buildings = dsm - dem
    buildings[buildings < 2.0] = 1.0
    buildings[buildings >= 2.0] = 0.0

    # l.481-492: ground and wall temperature parameters without a land-cover grid
    TgK = Knight + param["Ts_deg"]["Value"]["Cobble_stone_2014a"]
    Tstart = Knight - param["Tstart"]["Value"]["Cobble_stone_2014a"]
    TmaxLST = param["TmaxLST"]["Value"]["Cobble_stone_2014a"]
    alb_grid = Knight + param["Albedo"]["Effective"]["Value"]["Cobble_stone_2014a"]
    emis_grid = Knight + param["Emissivity"]["Value"]["Cobble_stone_2014a"]
    TgK_wall = param["Ts_deg"]["Value"]["Walls"]
    Tstart_wall = param["Tstart"]["Value"]["Walls"]
    TmaxLST_wall = param["TmaxLST"]["Value"]["Walls"]

    # Sun position, radiation split and altmax: read from the package Weather
    # object (own parity tests). dectime as in the met-file reader: day of year
    # plus the fraction of the day.
    jday = WHEN.timetuple().tm_yday
    dectime = jday + WHEN.hour / 24.0 + WHEN.minute / 1440.0

    return {
        "i": 0,
        "dsm": dsm,
        "scale": 1.0 / PIXEL_SIZE,
        "rows": rows,
        "cols": cols,
        "svf": svf,
        "svfN": svfN,
        "svfW": svfW,
        "svfE": svfE,
        "svfS": svfS,
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
        "altitude": float(weather.sun_altitude),
        "azimuth": float(weather.sun_azimuth),
        "zen": float(weather.sun_zenith) * np.pi / 180.0,
        "jday": jday,
        "usevegdem": 0,
        "onlyglobal": 0,  # direct and diffuse are passed in
        "buildings": buildings,
        "location": location,
        "psi": psi,
        "landcover": 0,
        "lc_grid": np.ones_like(dsm),  # l.137
        "dectime": dectime,
        "altmax": float(weather.altmax),
        "dirwalls": dirwalls,
        "walls": walls,
        "cyl": 1,  # runner setting, independent of the posture (l.73)
        "elvis": 0.0,  # l.77
        "Ta": TA,
        "RH": RH,
        "radG": GLOBAL_RAD,
        "radD": float(weather.diffuse_rad),
        "radI": float(weather.direct_rad),
        "P": float(weather.pressure) / 10.0,  # met files carry kPa; the function multiplies by 10
        "amaxvalue": amaxvalue,
        "bush": bush,
        "Twater": [],  # l.265
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
        "svfbuveg": svfbuveg,
        "firstdaytime": 1.0,  # l.631
        "timeadd": 0.0,  # l.630
        "timestepdec": TIMESTEP_DEC,  # l.629
        "Tgmap1": np.zeros((rows, cols)),  # l.394-398
        "Tgmap1E": np.zeros((rows, cols)),
        "Tgmap1S": np.zeros((rows, cols)),
        "Tgmap1W": np.zeros((rows, cols)),
        "Tgmap1N": np.zeros((rows, cols)),
        "CI": 1.0,  # l.641; recomputed inside the function in daytime
        "TgOut1": np.zeros((rows, cols)),
        "diffsh": None,  # l.459-465: isotropic sky
        "shmat": None,
        "vegshmat": None,
        "vbshvegshmat": None,
        "anisotropic_sky": 0,
        "asvf": None,
        "patch_option": 0,
        "voxelMaps": 0,  # l.618-623: no wall temperature scheme
        "voxelTable": 0,
        "ws": float(weather.ws),
        "wallScheme": 0,
        "timeStep": 0,
        "steradians": 0,
        "walls_scheme": np.ones((rows, cols)) * 10.0,
        "dirwalls_scheme": np.ones((rows, cols)) * 10.0,
    }


def _package_ground_inputs() -> dict[str, np.ndarray]:
    """The three inputs of findings 1 to 3, as ``calculate_core_fused`` derives them."""
    surface, _, _ = _scene()
    alb, _, _, tstart, _ = surface.get_land_cover_properties(None)
    buildings = detect_building_mask(surface.dsm, None, surface.wall_height, PIXEL_SIZE)
    return {
        "alb_grid": np.asarray(alb, dtype=float),
        "Tstart": np.asarray(tstart, dtype=float),
        "buildings": np.asarray(buildings, dtype=float),
    }


@functools.cache
def _reference_side(posture: str, aligned: bool = False) -> dict[str, np.ndarray]:
    """One timestep through the reference function.

    ``aligned`` replaces the ground albedo, ``Tstart`` and the building mask
    with the package's values, to test everything else.
    """
    args = _reference_inputs(posture)
    if aligned:
        args.update(_package_ground_inputs())
    with np.errstate(divide="ignore", invalid="ignore"):  # gvf_2018a divides by zero wall heights
        out = reference.Solweig_2025a_calc(*args.values())
    tmrt, kdown, kup, ldown, lup = out[0:5]
    return {
        "tmrt": np.asarray(tmrt, dtype=np.float64) + KELVIN_OFFSET_DIFFERENCE,
        "kdown": np.asarray(kdown, dtype=np.float64),
        "kup": np.asarray(kup, dtype=np.float64),
        "ldown": np.asarray(ldown, dtype=np.float64),
        "lup": np.asarray(lup, dtype=np.float64),
        "shadow": np.asarray(out[10], dtype=np.float64),
    }


def _difference(posture: str, name: str, *, aligned: bool = False) -> np.ndarray:
    package = _package_side(posture)[name]
    ref = _reference_side(posture, aligned)[name]
    return np.abs(package - ref)[_ground_mask()]


def _assert_within(posture: str, name: str, *, aligned: bool = False) -> None:
    diff = _difference(posture, name, aligned=aligned)
    assert np.all(np.isfinite(diff)), f"{name}: non-finite values over ground pixels"
    assert diff.max() <= ATOL[name], (
        f"{name} ({posture}): max abs diff {diff.max():.4f}, mean {diff.mean():.4f}, tolerance {ATOL[name]}"
    )


def _cases(known: dict[tuple[str, str], str]) -> list:
    """All posture and output pairs, with a strict xfail on the known differences."""
    cases = []
    for posture in ("standing", "sitting"):
        for name in OUTPUTS:
            marks = ()
            if (posture, name) in known:
                marks = (pytest.mark.xfail(strict=True, reason=known[(posture, name)]),)
            cases.append(pytest.param(posture, name, marks=marks, id=f"{posture}-{name}"))
    return cases


ALBEDO = "ground albedo 0.15 (package) against 0.20 (runner)"
TSTART = "Tstart -3.41 (package) against +3.41 (runner)"
MASK = "building mask covers the whole grid (package) against the 100 block pixels (runner)"
BOX = "sitting posture runs the box model (package) against cyl=1 (runner)"

AS_RUNNER = {
    ("standing", "kup"): f"max 40.81, mean 32.21 W/m2: {ALBEDO}; {MASK}",
    ("standing", "lup"): f"max 41.64, mean 34.86 W/m2: {TSTART}; {MASK}",
    ("standing", "tmrt"): f"max 5.36, mean 3.70 degC: {ALBEDO}; {TSTART}; {MASK}",
    ("sitting", "kup"): f"max 39.22, mean 32.22 W/m2: {ALBEDO}; {MASK}",
    ("sitting", "lup"): f"max 39.96, mean 35.15 W/m2: {TSTART}; {MASK}",
    ("sitting", "tmrt"): f"max 5.05, mean 4.15 degC: {ALBEDO}; {TSTART}; {MASK}; {BOX}",
}

ALIGNED = {
    ("sitting", "tmrt"): f"max 0.56, mean 0.52 degC with the ground inputs aligned (0.0001 with cyl=0): {BOX}",
}


def test_the_scene_is_a_sunlit_daytime_step_with_shade():
    """Guard against a degenerate comparison (night, or no shadow on the ground)."""
    ref = _reference_side("standing")
    ground = _ground_mask()
    assert 0.02 < np.mean(ref["shadow"][ground] < 0.5) < 0.5
    assert ref["kdown"][ground].min() > 0.0


@pytest.mark.parametrize(("posture", "name"), _cases(AS_RUNNER))
def test_one_timestep_matches_the_reference(posture, name):
    """Package against the reference with every input derived as in the reference runner."""
    _assert_within(posture, name)


@pytest.mark.parametrize(("posture", "name"), _cases(ALIGNED))
def test_rest_of_the_chain_matches_with_the_known_inputs_aligned(posture, name):
    """Same comparison with the package's ground albedo, Tstart and building mask on both sides."""
    _assert_within(posture, name, aligned=True)


@pytest.mark.xfail(strict=True, reason="0.15 in SurfaceData.get_land_cover_properties, 0.20 in the runner")
def test_default_ground_albedo_is_the_runner_value():
    runner = _reference_inputs("standing")["alb_grid"]
    np.testing.assert_allclose(_package_ground_inputs()["alb_grid"], runner, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="-3.41 in SurfaceData.get_land_cover_properties, +3.41 in the runner")
def test_default_tstart_is_the_runner_value():
    runner = _reference_inputs("standing")["Tstart"]
    np.testing.assert_allclose(_package_ground_inputs()["Tstart"], runner, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="1500 of 1600 px differ: walls dilated by 25 m against dsm - dem >= 2")
def test_building_mask_is_the_runner_mask():
    runner = _reference_inputs("standing")["buildings"]
    np.testing.assert_array_equal(_package_ground_inputs()["buildings"], runner)
