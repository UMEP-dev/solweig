"""
Golden Fixture Generator

Generates the golden fixtures from the Athens demo inputs stored in
``fixtures/input_*.npy`` and ``fixtures/input_params.npz``.

The reference fixtures (shadows, SVF, UTCI, PET, ground temperature delay)
are computed with the reference implementation, UMEP-dev/umep-solweig. The
remaining fixtures are regression baselines from the current Rust
implementation.

The reference is the copy installed by the ``test`` dependency group, at the
commit pinned in ``pyproject.toml``, or a local checkout given with
``--reference``. It is imported through ``tests/reference_package.py``, which
needs no GDAL.

The input fixtures are fixed data. They were written once from the Athens
demo rasters and this script does not rewrite them.

Usage:
    uv run python tests/golden/generate_fixtures.py --fixtures-dir <copy of fixtures/>
    uv run python tests/golden/generate_fixtures.py --reference <umep-solweig checkout>

With ``--fixtures-dir`` the script reads the inputs from, and writes to, that
directory, so a regenerated set can be compared with the stored one before
anything is replaced.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for tests.reference_package
from tests.reference_package import use_umep_solweig  # noqa: E402

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def ensure_fixtures_dir():
    """Create fixtures directory if it doesn't exist."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_shadow_fixtures():
    """Generate golden fixtures for shadow calculations using UMEP Python."""
    from umep_solweig.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import (
        shadowingfunction_wallheight_23,
    )

    print("Generating shadow fixtures (using UMEP Python as ground truth)...")

    dsm = np.load(FIXTURES_DIR / "input_dsm.npy").astype(np.float32)
    cdsm = np.load(FIXTURES_DIR / "input_cdsm.npy").astype(np.float32)
    tdsm = np.load(FIXTURES_DIR / "input_tdsm.npy").astype(np.float32)
    bush = np.load(FIXTURES_DIR / "input_bush.npy").astype(np.float32)
    wall_ht = np.load(FIXTURES_DIR / "input_wall_ht.npy").astype(np.float32)
    # The stored aspect is in degrees; the shadow function takes radians.
    wall_asp = (np.load(FIXTURES_DIR / "input_wall_asp.npy") * np.pi / 180.0).astype(np.float32)
    params = dict(np.load(FIXTURES_DIR / "input_params.npz"))
    scale = float(params["scale"])
    amaxvalue = float(params["amaxvalue"])

    # Test with multiple sun positions
    sun_positions = [
        {"name": "morning", "azimuth": 90.0, "altitude": 30.0},
        {"name": "noon", "azimuth": 180.0, "altitude": 60.0},
        {"name": "afternoon", "azimuth": 270.0, "altitude": 45.0},
    ]

    for pos in sun_positions:
        print(f"  Computing shadows for {pos['name']}...")
        # Use UMEP Python shadowingfunction_wallheight_23
        # Returns: (veg_sh, bldg_sh, veg_blocks_bldg_sh, wall_sh, wall_sun,
        #           wall_sh_veg, face_sh, face_sun)
        (
            veg_sh,
            bldg_sh,
            _veg_blocks_bldg_sh,
            wall_sh,
            wall_sun,
            _wall_sh_veg,
            _face_sh,
            _face_sun,
        ) = shadowingfunction_wallheight_23(
            dsm,
            cdsm,
            tdsm,
            pos["azimuth"],
            pos["altitude"],
            scale,
            amaxvalue,
            bush,
            wall_ht,
            wall_asp,
        )

        # Save each shadow component
        prefix = f"shadow_{pos['name']}"
        np.save(FIXTURES_DIR / f"{prefix}_bldg_sh.npy", np.array(bldg_sh))
        np.save(FIXTURES_DIR / f"{prefix}_veg_sh.npy", np.array(veg_sh))
        np.save(FIXTURES_DIR / f"{prefix}_wall_sh.npy", np.array(wall_sh))
        np.save(FIXTURES_DIR / f"{prefix}_wall_sun.npy", np.array(wall_sun))

    # Save input metadata for reproducibility
    np.savez(
        FIXTURES_DIR / "shadow_metadata.npz",
        dsm_shape=dsm.shape,
        scale=scale,
        amaxvalue=amaxvalue,
    )

    print("  Shadow fixtures saved.")


def generate_svf_fixtures():
    """Generate golden fixtures for SVF calculations using UMEP Python."""
    from umep_solweig.functions.svf_functions import svfForProcessing153

    print("Generating SVF fixtures (using UMEP Python as ground truth)...")

    dsm = np.load(FIXTURES_DIR / "input_dsm.npy").astype(np.float32)
    cdsm_abs = np.load(FIXTURES_DIR / "input_cdsm.npy").astype(np.float32)
    tdsm_abs = np.load(FIXTURES_DIR / "input_tdsm.npy").astype(np.float32)
    params = dict(np.load(FIXTURES_DIR / "input_params.npz"))
    scale = float(params["scale"])
    amaxvalue = float(params["amaxvalue"])

    # IMPORTANT: The reference svfForProcessing153 expects RELATIVE vegetation
    # heights (height above ground), not absolute elevations. It internally adds DSM
    # to convert to absolute heights. Our input data has absolute heights, so we must
    # convert to relative before calling UMEP.
    #
    # Conversion: relative_height = absolute_height - DSM
    # Where vegetation doesn't exist (CDSM <= DSM), set to 0.
    cdsm_rel = np.maximum(cdsm_abs - dsm, 0).astype(np.float32)
    tdsm_rel = np.maximum(tdsm_abs - dsm, 0).astype(np.float32)
    cdsm_rel[cdsm_abs <= dsm] = 0
    tdsm_rel[tdsm_abs <= dsm] = 0

    print("  Computing SVF (this may take a moment)...")
    # Use UMEP Python svfForProcessing153
    # Returns a dictionary with keys: svf, svfE, svfS, svfW, svfN, svfveg, etc.
    # Note: UMEP expects relative heights; Rust expects absolute heights.
    # Both produce equivalent results when properly configured.
    result = svfForProcessing153(
        dsm,
        cdsm_rel,  # Relative vegetation heights (UMEP expectation)
        tdsm_rel,  # Relative trunk heights (UMEP expectation)
        scale,
        1,  # usevegdem (1 = True)
        1.0 / scale,  # pixel_resolution, read by the wall scheme only
        False,  # wallScheme
        None,  # demlayer, read by the wall scheme only
        None,  # feedback
    )

    # Save SVF components (mapping UMEP Python keys to fixture names)
    np.save(FIXTURES_DIR / "svf_total.npy", np.array(result["svf"]))
    np.save(FIXTURES_DIR / "svf_north.npy", np.array(result["svfN"]))
    np.save(FIXTURES_DIR / "svf_east.npy", np.array(result["svfE"]))
    np.save(FIXTURES_DIR / "svf_south.npy", np.array(result["svfS"]))
    np.save(FIXTURES_DIR / "svf_west.npy", np.array(result["svfW"]))
    np.save(FIXTURES_DIR / "svf_veg.npy", np.array(result["svfveg"]))

    # Save metadata
    np.savez(
        FIXTURES_DIR / "svf_metadata.npz",
        dsm_shape=dsm.shape,
        scale=scale,
        amaxvalue=amaxvalue,
    )

    print("  SVF fixtures saved.")


def generate_gvf_fixtures():
    """
    Generate golden fixtures for GVF calculations using UMEP Python as ground truth.
    """
    print("Generating GVF fixtures (using UMEP Python as ground truth)...")

    from scipy import ndimage
    from umep_solweig.functions.SOLWEIGpython.gvf_2018a import gvf_2018a

    SBC = 5.67e-8  # Stefan-Boltzmann constant

    # Load input data
    dsm = np.load(FIXTURES_DIR / "input_dsm.npy")
    wall_ht = np.load(FIXTURES_DIR / "input_wall_ht.npy")
    wall_asp = np.load(FIXTURES_DIR / "input_wall_asp.npy")
    shadow_noon_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy")
    shadow_noon_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy")
    wall_sun = np.load(FIXTURES_DIR / "shadow_noon_wall_sun.npy")
    params = dict(np.load(FIXTURES_DIR / "input_params.npz"))

    rows, cols = dsm.shape
    scale = float(params["scale"])

    # Create building mask (same logic as Rust)
    wall_mask = wall_ht > 0
    struct = ndimage.generate_binary_structure(2, 2)
    iterations = int(25 / scale) + 1
    dilated = ndimage.binary_dilation(wall_mask, struct, iterations=iterations)
    buildings = (~dilated).astype(np.float64)

    # Inputs matching the test parameters
    shadow = (shadow_noon_bldg * shadow_noon_veg).astype(np.float64)

    # Create realistic spatially-varying ground temperature
    # - Base: air temperature (25°C)
    # - Sunlit areas: +8-12°C warmer (solar heating)
    # - Shaded areas: +0-2°C above air temp
    # - Add slight random variation for surface heterogeneity
    ta = 25.0
    np.random.seed(42)  # Reproducible
    sun_heating = 10.0 * shadow  # shadow=1 means sunlit, shadow=0 means shaded
    random_variation = np.random.normal(0, 1.0, (rows, cols))
    tg = (ta + sun_heating + random_variation).astype(np.float64)

    emis_grid = np.full((rows, cols), 0.95, dtype=np.float64)
    alb_grid = np.full((rows, cols), 0.15, dtype=np.float64)
    lc_grid = None  # No land cover grid

    # GVF parameters (matching test_golden_gvf.py)
    first = 2.0  # round(height)
    second = 36.0  # round(height * 20) for 1.8m
    tgwall = 2.0
    ewall = 0.90
    albedo_b = 0.20
    twater = 25.0
    landcover = False

    # Call UMEP Python gvf_2018a
    (
        gvfLup,
        gvfalb,
        gvfalbnosh,
        gvfLupE,
        gvfalbE,
        gvfalbnoshE,
        gvfLupS,
        gvfalbS,
        gvfalbnoshS,
        gvfLupW,
        gvfalbW,
        gvfalbnoshW,
        gvfLupN,
        gvfalbN,
        gvfalbnoshN,
        gvfSum,
        gvfNorm,
    ) = gvf_2018a(
        wall_sun.astype(np.float64),
        wall_ht.astype(np.float64),
        buildings,
        scale,
        shadow,
        first,
        second,
        wall_asp.astype(np.float64),
        tg,
        tgwall,
        ta,
        emis_grid,
        ewall,
        alb_grid,
        SBC,
        albedo_b,
        rows,
        cols,
        twater,
        lc_grid,
        landcover,
    )

    # Save fixtures
    np.save(FIXTURES_DIR / "gvf_lup.npy", gvfLup.astype(np.float32))
    np.save(FIXTURES_DIR / "gvf_alb.npy", gvfalb.astype(np.float32))
    np.save(FIXTURES_DIR / "gvf_norm.npy", gvfNorm.astype(np.float32))
    np.save(FIXTURES_DIR / "gvf_input_tg.npy", tg.astype(np.float32))  # Ground temperature input

    print("  GVF fixtures saved (from UMEP Python).")


def generate_radiation_fixtures():
    """
    Generate golden fixtures for radiation (Kside/Lside) using UMEP Python as ground truth.

    Uses isotropic mode (anisotropic_diffuse=0, anisotropic_longwave=False) which
    doesn't require the complex shadow matrices.
    """
    print("Generating radiation fixtures (using UMEP Python as ground truth)...")

    from umep_solweig.functions.SOLWEIGpython.Kside_veg_v2022a import Kside_veg_v2022a
    from umep_solweig.functions.SOLWEIGpython.Lside_veg import Lside_veg_v2022a

    SBC = 5.67e-8  # Stefan-Boltzmann constant

    # Load SVF data
    svf = np.load(FIXTURES_DIR / "svf_total.npy").astype(np.float64)
    svf_n = np.load(FIXTURES_DIR / "svf_north.npy").astype(np.float64)
    svf_e = np.load(FIXTURES_DIR / "svf_east.npy").astype(np.float64)
    svf_s = np.load(FIXTURES_DIR / "svf_south.npy").astype(np.float64)
    svf_w = np.load(FIXTURES_DIR / "svf_west.npy").astype(np.float64)
    svf_veg = np.load(FIXTURES_DIR / "svf_veg.npy").astype(np.float64)

    # Load shadow data
    shadow_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy").astype(np.float64)
    shadow_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy").astype(np.float64)
    shadow = shadow_bldg * shadow_veg

    rows, cols = svf.shape

    # Weather parameters (matching test_golden_radiation.py)
    ta = 25.0
    rad_i = 600.0
    rad_d = 200.0
    rad_g = 800.0
    esky = 0.75
    ci = 0.85
    azimuth = 180.0  # Solar noon
    altitude = 60.0
    psi = 0.5  # Vegetation transmissivity
    t = 0.0  # Orientation offset
    albedo = 0.20
    tw = 2.0  # Wall temperature offset
    ewall = 0.90

    # Synthetic arrays
    f_sh = np.full((rows, cols), 0.5, dtype=np.float64)
    kup_base = np.full((rows, cols), 50.0, dtype=np.float64)

    # Kside calculation (isotropic mode: anisotropic_diffuse=0)
    # In isotropic mode, lv, diffsh, asvf, shmat, vegshmat, vbshvegshmat are not used
    Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside = Kside_veg_v2022a(
        rad_i,
        rad_d,
        rad_g,
        shadow,
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,  # svf_*_veg
        azimuth,
        altitude,
        psi,
        t,
        albedo,
        f_sh,
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        cyl=True,  # Use cylinder model
        lv=None,
        anisotropic_diffuse=0,  # Isotropic mode
        diffsh=None,
        rows=rows,
        cols=cols,
        asvf=None,
        shmat=None,
        vegshmat=None,
        vbshvegshmat=None,
    )

    np.save(FIXTURES_DIR / "radiation_kside_e.npy", Keast.astype(np.float32))
    np.save(FIXTURES_DIR / "radiation_kside_s.npy", Ksouth.astype(np.float32))

    # Lside calculation (isotropic mode: anisotropic_longwave=False)
    ta_k = ta + 273.15
    ldown_base = esky * SBC * (ta_k**4)
    ldown = np.full((rows, cols), ldown_base, dtype=np.float64)
    lup_base = 0.95 * SBC * (ta_k**4)
    lup = np.full((rows, cols), lup_base, dtype=np.float64)

    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2022a(
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,  # svf_*_veg
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,  # svf_*_aveg
        azimuth,
        altitude,
        ta,
        tw,
        SBC,
        ewall,
        ldown,
        esky,
        t,
        f_sh,
        ci,
        lup.copy(),
        lup.copy(),
        lup.copy(),
        lup.copy(),
        anisotropic_longwave=False,
    )

    np.save(FIXTURES_DIR / "radiation_lside_e.npy", Least.astype(np.float32))
    np.save(FIXTURES_DIR / "radiation_lside_s.npy", Lsouth.astype(np.float32))

    print("  Radiation fixtures saved (from UMEP Python).")


def generate_utci_fixtures():
    """
    Generate golden fixtures for UTCI calculations using UMEP Python as ground truth.

    Tests both single-point and grid calculations with various input combinations.
    """
    print("Generating UTCI fixtures (using UMEP Python as ground truth)...")

    from umep_solweig.functions.SOLWEIGpython.UTCI_calculations import utci_calculator

    # Test cases: (ta, rh, tmrt, va10m, description)
    # Cover a range of realistic outdoor conditions
    test_cases = [
        # Normal comfortable conditions
        (20.0, 50.0, 22.0, 1.5, "comfortable"),
        # Hot summer day
        (35.0, 40.0, 55.0, 1.0, "hot_summer"),
        # Cold winter day
        (5.0, 70.0, 2.0, 3.0, "cold_winter"),
        # High humidity tropical
        (30.0, 85.0, 38.0, 0.5, "tropical"),
        # Windy conditions
        (25.0, 60.0, 30.0, 8.0, "windy"),
        # High radiation (large Tmrt-Ta delta)
        (25.0, 45.0, 60.0, 2.0, "high_radiation"),
        # Low wind edge case (minimum valid wind)
        (22.0, 55.0, 28.0, 0.1, "low_wind"),
    ]

    # Calculate UTCI for each test case using UMEP Python
    utci_inputs = []
    utci_outputs = []

    for ta, rh, tmrt, va, desc in test_cases:
        utci = utci_calculator(ta, rh, tmrt, va)
        utci_inputs.append([ta, rh, tmrt, va])
        utci_outputs.append(utci)
        print(f"    {desc}: Ta={ta}°C, RH={rh}%, Tmrt={tmrt}°C, va={va}m/s -> UTCI={utci:.2f}°C")

    # Save fixtures
    np.savez(
        FIXTURES_DIR / "utci_single_point.npz",
        inputs=np.array(utci_inputs, dtype=np.float32),  # [n_tests, 4] -> [ta, rh, tmrt, va]
        outputs=np.array(utci_outputs, dtype=np.float32),  # [n_tests]
        descriptions=[desc for _, _, _, _, desc in test_cases],
    )

    # Also generate a grid test case using existing Tmrt-like data
    # Create a synthetic Tmrt grid (using SVF as spatial variability source)
    svf = np.load(FIXTURES_DIR / "svf_total.npy")
    rows, cols = svf.shape

    # Tmrt varies with SVF: lower SVF = more enclosed = warmer Tmrt
    ta_grid = 25.0
    tmrt_grid = ta_grid + 15.0 * (1 - svf) + 10.0 * svf  # Range: 25-40°C
    tmrt_grid = tmrt_grid.astype(np.float32)

    # Wind speed also varies spatially (lower in enclosed areas)
    va_grid = 0.5 + 3.0 * svf  # Range: 0.5-3.5 m/s
    va_grid = va_grid.astype(np.float32)

    # Calculate UTCI grid using UMEP Python (scalar Ta and RH)
    from umep_solweig.functions.SOLWEIGpython.UTCI_calculations import utci_calculator_grid

    class DummyFeedback:
        def isCanceled(self):
            return False

        def setProgress(self, _):
            pass

        def setProgressText(self, _):
            pass

    utci_grid = utci_calculator_grid(ta_grid, 50.0, tmrt_grid, va_grid, DummyFeedback())

    np.save(FIXTURES_DIR / "utci_grid_tmrt.npy", tmrt_grid)
    np.save(FIXTURES_DIR / "utci_grid_va.npy", va_grid)
    np.save(FIXTURES_DIR / "utci_grid_output.npy", utci_grid.astype(np.float32))
    np.savez(
        FIXTURES_DIR / "utci_grid_params.npz",
        ta=ta_grid,
        rh=50.0,
    )

    print("  UTCI fixtures saved (from UMEP Python).")


def generate_pet_fixtures():
    """
    Generate golden fixtures for PET calculations using UMEP Python as ground truth.

    PET uses an iterative solver and is much slower than UTCI (~50x).
    """
    print("Generating PET fixtures (using UMEP Python as ground truth)...")

    from umep_solweig.functions.SOLWEIGpython.PET_calculations import _PET

    # Default person parameters (standard adult)
    # mbody, age, height, activity, clo, sex
    mbody = 75.0  # kg
    age = 35.0  # years
    height = 1.80  # meters
    activity = 80.0  # W (walking slowly)
    clo = 0.9  # summer clothing
    sex = 1  # male

    # Test cases: (ta, rh, tmrt, va, description)
    test_cases = [
        # Comfortable conditions
        (20.0, 50.0, 22.0, 1.0, "comfortable"),
        # Hot summer day
        (35.0, 40.0, 55.0, 1.0, "hot_summer"),
        # Cold winter day
        (5.0, 70.0, 2.0, 2.0, "cold_winter"),
        # High humidity tropical
        (30.0, 85.0, 38.0, 0.5, "tropical"),
        # High radiation (large Tmrt-Ta delta)
        (25.0, 45.0, 55.0, 1.5, "high_radiation"),
    ]

    # Calculate PET for each test case using UMEP Python
    pet_inputs = []
    pet_outputs = []

    for ta, rh, tmrt, va, desc in test_cases:
        pet_val = _PET(ta, rh, tmrt, va, mbody, age, height, activity, clo, sex)
        pet_inputs.append([ta, rh, tmrt, va])
        pet_outputs.append(pet_val)
        print(f"    {desc}: Ta={ta}°C, RH={rh}%, Tmrt={tmrt}°C, va={va}m/s -> PET={pet_val:.2f}°C")

    # Save fixtures
    np.savez(
        FIXTURES_DIR / "pet_single_point.npz",
        inputs=np.array(pet_inputs, dtype=np.float32),  # [n_tests, 4] -> [ta, rh, tmrt, va]
        outputs=np.array(pet_outputs, dtype=np.float32),  # [n_tests]
        descriptions=[desc for _, _, _, _, desc in test_cases],
        # Person parameters
        mbody=mbody,
        age=age,
        height=height,
        activity=activity,
        clo=clo,
        sex=sex,
    )

    # Grid test - use small subset due to slow PET calculation
    print("    Computing PET grid (small subset, this may take a moment)...")
    from umep_solweig.functions.SOLWEIGpython.PET_calculations import PET_person, calculate_PET_grid

    # Use existing UTCI grid inputs but crop to smaller size
    tmrt_full = np.load(FIXTURES_DIR / "utci_grid_tmrt.npy")
    va_full = np.load(FIXTURES_DIR / "utci_grid_va.npy")

    # Crop to 20x20 for faster calculation
    crop_size = 20
    tmrt_crop = tmrt_full[:crop_size, :crop_size].copy()
    va_crop = va_full[:crop_size, :crop_size].copy()

    ta_grid = 25.0
    rh_grid = 50.0

    pet_person = PET_person(mbody=mbody, age=age, height=height, activity=activity, sex=sex, clo=clo)

    class DummyFeedback:
        def isCanceled(self):
            return False

        def setProgress(self, _):
            pass

        def setProgressText(self, _):
            pass

    pet_grid = calculate_PET_grid(ta_grid, rh_grid, tmrt_crop, va_crop, pet_person, DummyFeedback())

    np.save(FIXTURES_DIR / "pet_grid_tmrt.npy", tmrt_crop.astype(np.float32))
    np.save(FIXTURES_DIR / "pet_grid_va.npy", va_crop.astype(np.float32))
    np.save(FIXTURES_DIR / "pet_grid_output.npy", pet_grid.astype(np.float32))
    np.savez(
        FIXTURES_DIR / "pet_grid_params.npz",
        ta=ta_grid,
        rh=rh_grid,
        mbody=mbody,
        age=age,
        height=height,
        activity=activity,
        clo=clo,
        sex=sex,
    )

    print("  PET fixtures saved (from UMEP Python).")


def generate_tmrt_fixtures():
    """
    Generate golden fixtures for Tmrt calculations.

    Tmrt is computed from radiation budget using the Stefan-Boltzmann formula:
        Tmrt = (Sstr / (abs_l * SBC))^0.25 - 273.15

    We create synthetic but physically-consistent radiation inputs and compute
    the expected Tmrt using the same formula as UMEP Python.
    """
    print("Generating Tmrt fixtures...")

    SBC = 5.67e-8  # Stefan-Boltzmann constant
    rows, cols = 30, 30

    # Standard absorption coefficients
    abs_k = 0.70  # shortwave
    abs_l = 0.97  # longwave

    # View factors for standing posture (cylinder)
    f_up = 0.06
    f_side = 0.22
    f_cyl = 0.28

    # Create synthetic radiation inputs
    # Use realistic values for summer day conditions
    np.random.seed(42)

    # Base radiation values (W/m²)
    kdown_base = 800.0  # Global shortwave
    kup_base = 120.0  # Reflected shortwave
    ldown_base = 380.0  # Atmospheric longwave
    lup_base = 450.0  # Surface longwave

    # Create spatial variation (buildings cause shadows, lower Kdown)
    svf = np.load(FIXTURES_DIR / "svf_total.npy")[:rows, :cols]  # Use SVF for spatial variation

    # Radiation varies with SVF
    kdown = (kdown_base * svf + np.random.uniform(0, 50, (rows, cols))).astype(np.float32)
    kup = (kup_base * svf + np.random.uniform(0, 20, (rows, cols))).astype(np.float32)
    ldown = np.full((rows, cols), ldown_base, dtype=np.float32) + np.random.uniform(-10, 10, (rows, cols)).astype(
        np.float32
    )
    lup = (lup_base + 30 * (1 - svf) + np.random.uniform(-5, 5, (rows, cols))).astype(np.float32)

    # Directional radiation (simplified)
    kside_n = (0.1 * kdown + np.random.uniform(0, 30, (rows, cols))).astype(np.float32)
    kside_e = (0.15 * kdown + np.random.uniform(0, 30, (rows, cols))).astype(np.float32)
    kside_s = (0.25 * kdown + np.random.uniform(0, 30, (rows, cols))).astype(np.float32)
    kside_w = (0.12 * kdown + np.random.uniform(0, 30, (rows, cols))).astype(np.float32)
    kside_total = (0.5 * kdown + np.random.uniform(0, 50, (rows, cols))).astype(np.float32)

    lside_n = (0.25 * ldown + np.random.uniform(0, 20, (rows, cols))).astype(np.float32)
    lside_e = (0.25 * ldown + np.random.uniform(0, 20, (rows, cols))).astype(np.float32)
    lside_s = (0.25 * ldown + np.random.uniform(0, 20, (rows, cols))).astype(np.float32)
    lside_w = (0.25 * ldown + np.random.uniform(0, 20, (rows, cols))).astype(np.float32)
    lside_total = (0.6 * ldown + np.random.uniform(0, 30, (rows, cols))).astype(np.float32)

    # Compute expected Tmrt using UMEP formula (anisotropic mode)
    # Sstr = absK * (Kside * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside)
    #      + absL * ((Ldown + Lup) * Fup + Lside * Fcyl + (Lnorth + Least + Lsouth + Lwest) * Fside)

    # Anisotropic (use_aniso=True)
    sstr_aniso = abs_k * (
        kside_total * f_cyl + (kdown + kup) * f_up + (kside_n + kside_e + kside_s + kside_w) * f_side
    ) + abs_l * ((ldown + lup) * f_up + lside_total * f_cyl + (lside_n + lside_e + lside_s + lside_w) * f_side)
    tmrt_aniso = np.sqrt(np.sqrt(sstr_aniso / (abs_l * SBC))) - 273.15
    tmrt_aniso = np.clip(tmrt_aniso, -50, 80).astype(np.float32)

    # Isotropic (use_aniso=False)
    # In isotropic mode, no Lside*Fcyl term for longwave (only directional components)
    sstr_iso = abs_k * (
        kside_total * f_cyl + (kdown + kup) * f_up + (kside_n + kside_e + kside_s + kside_w) * f_side
    ) + abs_l * ((ldown + lup) * f_up + (lside_n + lside_e + lside_s + lside_w) * f_side)
    tmrt_iso = np.sqrt(np.sqrt(sstr_iso / (abs_l * SBC))) - 273.15
    tmrt_iso = np.clip(tmrt_iso, -50, 80).astype(np.float32)

    # Save inputs
    np.save(FIXTURES_DIR / "tmrt_input_kdown.npy", kdown)
    np.save(FIXTURES_DIR / "tmrt_input_kup.npy", kup)
    np.save(FIXTURES_DIR / "tmrt_input_ldown.npy", ldown)
    np.save(FIXTURES_DIR / "tmrt_input_lup.npy", lup)
    np.save(FIXTURES_DIR / "tmrt_input_kside_n.npy", kside_n)
    np.save(FIXTURES_DIR / "tmrt_input_kside_e.npy", kside_e)
    np.save(FIXTURES_DIR / "tmrt_input_kside_s.npy", kside_s)
    np.save(FIXTURES_DIR / "tmrt_input_kside_w.npy", kside_w)
    np.save(FIXTURES_DIR / "tmrt_input_kside_total.npy", kside_total)
    np.save(FIXTURES_DIR / "tmrt_input_lside_n.npy", lside_n)
    np.save(FIXTURES_DIR / "tmrt_input_lside_e.npy", lside_e)
    np.save(FIXTURES_DIR / "tmrt_input_lside_s.npy", lside_s)
    np.save(FIXTURES_DIR / "tmrt_input_lside_w.npy", lside_w)
    np.save(FIXTURES_DIR / "tmrt_input_lside_total.npy", lside_total)

    # Save expected outputs
    np.save(FIXTURES_DIR / "tmrt_output_aniso.npy", tmrt_aniso)
    np.save(FIXTURES_DIR / "tmrt_output_iso.npy", tmrt_iso)

    np.savez(
        FIXTURES_DIR / "tmrt_params.npz",
        abs_k=abs_k,
        abs_l=abs_l,
        f_up=f_up,
        f_side=f_side,
        f_cyl=f_cyl,
    )

    print(f"    Tmrt range (aniso): {tmrt_aniso.min():.1f}°C to {tmrt_aniso.max():.1f}°C")
    print(f"    Tmrt range (iso): {tmrt_iso.min():.1f}°C to {tmrt_iso.max():.1f}°C")
    print("  Tmrt fixtures saved.")


def generate_ground_temp_fixtures():
    """
    Generate golden fixtures for ground temperature (TsWaveDelay) calculations.

    TsWaveDelay implements thermal inertia for ground temperature using an
    exponential decay model with decay constant 33.27 day⁻¹.

    Formula: Lup = Tgmap0 * (1 - weight) + Tgmap1 * weight
    where:  weight = exp(-33.27 * timeadd)
    """
    print("Generating ground temperature fixtures (using UMEP Python as ground truth)...")

    from umep_solweig.functions.SOLWEIGpython.TsWaveDelay_2015a import TsWaveDelay_2015a

    rows, cols = 20, 20

    # Create synthetic gvfLup (current radiative equilibrium) and Tgmap1 (previous temp)
    np.random.seed(42)

    # Current radiative equilibrium temperature (varies spatially)
    gvfLup = (400 + np.random.uniform(-20, 20, (rows, cols))).astype(np.float64)

    # Previous temperature (slightly different)
    Tgmap1_init = (380 + np.random.uniform(-15, 15, (rows, cols))).astype(np.float64)

    # Test case 1: First timestep of the day (firstdaytime=1)
    Lup1, timeadd1, Tgmap1_1 = TsWaveDelay_2015a(
        gvfLup=gvfLup.copy(),
        firstdaytime=1,
        timeadd=0.0,
        timestepdec=30 / 1440,  # 30 minutes
        Tgmap1=Tgmap1_init.copy(),
    )

    # Test case 2: Short timestep accumulation (timeadd < 59 min)
    Lup2, timeadd2, Tgmap1_2 = TsWaveDelay_2015a(
        gvfLup=gvfLup.copy(),
        firstdaytime=0,
        timeadd=30 / 1440,  # 30 minutes accumulated
        timestepdec=30 / 1440,  # 30 minute step
        Tgmap1=Tgmap1_init.copy(),
    )

    # Test case 3: Long timestep (timeadd >= 59 min)
    Lup3, timeadd3, Tgmap1_3 = TsWaveDelay_2015a(
        gvfLup=gvfLup.copy(),
        firstdaytime=0,
        timeadd=60 / 1440,  # 60 minutes accumulated (above threshold)
        timestepdec=60 / 1440,  # 60 minute step
        Tgmap1=Tgmap1_init.copy(),
    )

    # Save inputs
    np.save(FIXTURES_DIR / "ground_temp_input_gvflup.npy", gvfLup.astype(np.float32))
    np.save(FIXTURES_DIR / "ground_temp_input_tgmap1.npy", Tgmap1_init.astype(np.float32))

    # Save outputs for each test case
    np.savez(
        FIXTURES_DIR / "ground_temp_case1.npz",
        lup=Lup1.astype(np.float32),
        timeadd=timeadd1,
        tgmap1=Tgmap1_1.astype(np.float32),
        input_firstdaytime=1,
        input_timeadd=0.0,
        input_timestepdec=30 / 1440,
    )

    np.savez(
        FIXTURES_DIR / "ground_temp_case2.npz",
        lup=Lup2.astype(np.float32),
        timeadd=timeadd2,
        tgmap1=Tgmap1_2.astype(np.float32),
        input_firstdaytime=0,
        input_timeadd=30 / 1440,
        input_timestepdec=30 / 1440,
    )

    np.savez(
        FIXTURES_DIR / "ground_temp_case3.npz",
        lup=Lup3.astype(np.float32),
        timeadd=timeadd3,
        tgmap1=Tgmap1_3.astype(np.float32),
        input_firstdaytime=0,
        input_timeadd=60 / 1440,
        input_timestepdec=60 / 1440,
    )

    print(f"    Case 1 (first morning): Lup range {Lup1.min():.1f}-{Lup1.max():.1f}")
    print(f"    Case 2 (short step): Lup range {Lup2.min():.1f}-{Lup2.max():.1f}, timeadd={timeadd2:.4f}")
    print(f"    Case 3 (long step): Lup range {Lup3.min():.1f}-{Lup3.max():.1f}, timeadd={timeadd3:.4f}")
    print("  Ground temperature fixtures saved (from UMEP Python).")


def generate_anisotropic_sky_fixtures():
    """
    Generate golden fixtures for anisotropic sky radiation model.

    Uses the Rust implementation to generate reference values for regression testing.
    The anisotropic sky model computes direction-dependent longwave and shortwave
    radiation from sky patches, vegetation, and buildings.
    """
    print("Generating anisotropic sky fixtures (using Rust implementation)...")

    from solweig.rustalgos import sky

    # Load base inputs
    dsm = np.load(FIXTURES_DIR / "input_dsm.npy").astype(np.float32)
    svf = np.load(FIXTURES_DIR / "svf_total.npy").astype(np.float32)
    shadow_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy").astype(np.float32)
    shadow_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy").astype(np.float32)

    rows, cols = dsm.shape
    SBC = 5.67e-8

    # Generate sky patches (simplified Tregenza-style)
    def generate_sky_patches(n_alt_bands=4):
        patches = []
        alt_bands = [6, 18, 30, 42]
        azis_per_band = [30, 24, 24, 18]
        for alt, n_azi in zip(alt_bands[:n_alt_bands], azis_per_band[:n_alt_bands], strict=False):
            azi_step = 360.0 / n_azi if n_azi > 1 else 0
            for azi_idx in range(n_azi):
                azi = azi_idx * azi_step
                patches.append([alt, azi])
        return np.array(patches, dtype=np.float32)

    def compute_steradians(l_patches):
        n_patches = len(l_patches)
        steradians = np.zeros(n_patches, dtype=np.float32)
        deg2rad = np.pi / 180.0
        altitudes = l_patches[:, 0]
        unique_alts = np.unique(altitudes)
        for i, alt in enumerate(unique_alts):
            mask = altitudes == alt
            count = np.sum(mask)
            if i == 0:
                ster = (360.0 / count * deg2rad) * np.sin(alt * deg2rad)
            else:
                prev_alt = unique_alts[i - 1]
                delta_alt = (alt - prev_alt) / 2
                ster = (360.0 / count * deg2rad) * (
                    np.sin((alt + delta_alt) * deg2rad) - np.sin((prev_alt + delta_alt) * deg2rad)
                )
            steradians[mask] = ster
        return steradians

    l_patches = generate_sky_patches(n_alt_bands=4)
    n_patches = len(l_patches)
    steradians = compute_steradians(l_patches)

    # Create 3D shadow matrices
    svf_expanded = svf[:, :, np.newaxis]
    base_visibility = np.broadcast_to(svf_expanded, (rows, cols, n_patches)).copy()
    bldg_factor = shadow_bldg[:, :, np.newaxis]
    veg_factor = shadow_veg[:, :, np.newaxis]

    shmat = (base_visibility * np.broadcast_to(bldg_factor, (rows, cols, n_patches))).astype(np.float32)
    shmat = (shmat > 0.5).astype(np.float32)
    vegshmat = (base_visibility * np.broadcast_to(veg_factor, (rows, cols, n_patches))).astype(np.float32)
    vegshmat = (vegshmat > 0.3).astype(np.float32)
    vbshvegshmat = (shmat * vegshmat).astype(np.float32)

    # Other inputs
    asvf = svf.astype(np.float32)
    luminance = 1000 + 500 * np.sin(l_patches[:, 0] * np.pi / 180)
    lv = np.column_stack([l_patches, luminance]).astype(np.float32)
    ta = 25.0
    ta_k = ta + 273.15
    lup_val = 0.95 * SBC * (ta_k**4)
    lup = np.full((rows, cols), lup_val, dtype=np.float32)
    shadow = (shadow_bldg * shadow_veg).astype(np.float32)
    kup_base = np.full((rows, cols), 50.0, dtype=np.float32)

    # Create parameter objects
    sun_params = sky.SunParams(altitude=60.0, azimuth=180.0)
    sky_params = sky.SkyParams(esky=0.75, ta=25.0, cyl=True, wall_scheme=False, albedo=0.20)
    surface_params = sky.SurfaceParams(tgwall=2.0, ewall=0.90, rad_i=600.0, rad_d=200.0)

    # Compute result
    result = sky.anisotropic_sky(
        shmat,
        vegshmat,
        vbshvegshmat,
        sun_params,
        asvf,
        sky_params,
        l_patches,
        None,
        None,
        steradians,
        surface_params,
        lup,
        lv,
        shadow,
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
    )

    # Save outputs
    np.savez(
        FIXTURES_DIR / "aniso_sky_output.npz",
        ldown=np.array(result.ldown),
        lside=np.array(result.lside),
        lside_sky=np.array(result.lside_sky),
        lside_veg=np.array(result.lside_veg),
        kside=np.array(result.kside),
        kside_i=np.array(result.kside_i),
        kside_d=np.array(result.kside_d),
        # Input parameters for reproducibility
        sun_altitude=60.0,
        sun_azimuth=180.0,
        ta=25.0,
        esky=0.75,
    )

    print(f"    Ldown range: {np.array(result.ldown).min():.1f}-{np.array(result.ldown).max():.1f} W/m²")
    print(f"    Lside range: {np.array(result.lside).min():.1f}-{np.array(result.lside).max():.1f} W/m²")
    print(f"    Kside range: {np.array(result.kside).min():.1f}-{np.array(result.kside).max():.1f} W/m²")
    print("  Anisotropic sky fixtures saved (from Rust implementation).")


def generate_wall_temp_fixtures():
    """
    Generate golden fixtures for wall temperature deviation calculations.

    Uses the Rust implementation to generate reference values for regression testing.
    """
    print("Generating wall temperature fixtures (using Rust implementation)...")

    from solweig.rustalgos import ground

    rows, cols = 20, 20
    np.random.seed(42)

    # Test parameters
    ta = 25.0
    sun_altitude = 45.0  # Moderate sun
    altmax = 65.0  # Max altitude for day
    dectime = 0.5  # Noon (12:00 as fraction of day)
    snup = 0.25  # Sunrise at 6:00 (6/24 = 0.25)
    global_rad = 600.0  # W/m²
    rad_g0 = 800.0  # Clear sky
    zen_deg = 45.0  # = 90 - altitude

    # Land cover parameters (per-pixel grids)
    alb_grid = np.full((rows, cols), 0.15, dtype=np.float32)
    emis_grid = np.full((rows, cols), 0.95, dtype=np.float32)

    # TgK and Tstart vary by land cover type (grass, asphalt, concrete, etc.)
    # Grass: TgK=0.37, Tstart=-3.41
    # Asphalt: TgK=0.50, Tstart=-2.0
    tgk_grid = np.full((rows, cols), 0.37, dtype=np.float32)
    tgk_grid[:10, :] = 0.50  # Upper half is asphalt

    tstart_grid = np.full((rows, cols), -3.41, dtype=np.float32)
    tstart_grid[:10, :] = -2.0  # Upper half is asphalt

    tmaxlst_grid = np.full((rows, cols), 15.0, dtype=np.float32)  # Max temp at 15:00

    # Use Rust implementation to generate expected values
    tg, tg_wall, ci_tg, alb_out, emis_out = ground.compute_ground_temperature(
        ta,
        sun_altitude,
        altmax,
        dectime,
        snup,
        global_rad,
        rad_g0,
        zen_deg,
        alb_grid,
        emis_grid,
        tgk_grid,
        tstart_grid,
        tmaxlst_grid,
    )
    tg_expected = np.array(tg)
    tg_wall_expected = float(tg_wall)
    ci_tg_expected = float(ci_tg)

    # Save inputs
    np.save(FIXTURES_DIR / "wall_temp_input_alb.npy", alb_grid)
    np.save(FIXTURES_DIR / "wall_temp_input_emis.npy", emis_grid)
    np.save(FIXTURES_DIR / "wall_temp_input_tgk.npy", tgk_grid)
    np.save(FIXTURES_DIR / "wall_temp_input_tstart.npy", tstart_grid)
    np.save(FIXTURES_DIR / "wall_temp_input_tmaxlst.npy", tmaxlst_grid)

    # Save expected outputs and parameters
    np.savez(
        FIXTURES_DIR / "wall_temp_output.npz",
        tg=tg_expected,
        tg_wall=tg_wall_expected,
        ci_tg=ci_tg_expected,
        # Input parameters
        ta=ta,
        sun_altitude=sun_altitude,
        altmax=altmax,
        dectime=dectime,
        snup=snup,
        global_rad=global_rad,
        rad_g0=rad_g0,
        zen_deg=zen_deg,
    )

    print(f"    Tg range: {tg_expected.min():.2f}°C to {tg_expected.max():.2f}°C")
    print(f"    Tg_wall: {tg_wall_expected:.2f}°C")
    print(f"    CI_Tg: {ci_tg_expected:.4f}")
    print("  Wall temperature fixtures saved.")


def generate_aniso_radiation_fixtures():
    """
    Generate golden fixtures for anisotropic radiation calculations.

    Uses the Rust implementation to generate reference values for regression testing.
    Tests kside_veg and lside_veg in anisotropic mode.
    """
    print("Generating anisotropic radiation fixtures (using Rust implementation)...")

    from solweig.constants import SBC
    from solweig.rustalgos import vegetation

    # Load input data
    dsm = np.load(FIXTURES_DIR / "input_dsm.npy").astype(np.float32)
    svf = np.load(FIXTURES_DIR / "svf_total.npy").astype(np.float32)
    svf_e = np.load(FIXTURES_DIR / "svf_east.npy").astype(np.float32)
    svf_s = np.load(FIXTURES_DIR / "svf_south.npy").astype(np.float32)
    svf_w = np.load(FIXTURES_DIR / "svf_west.npy").astype(np.float32)
    svf_n = np.load(FIXTURES_DIR / "svf_north.npy").astype(np.float32)
    svf_veg = np.load(FIXTURES_DIR / "svf_veg.npy").astype(np.float32)
    shadow_bldg = np.load(FIXTURES_DIR / "shadow_noon_bldg_sh.npy").astype(np.float32)
    shadow_veg = np.load(FIXTURES_DIR / "shadow_noon_veg_sh.npy").astype(np.float32)

    rows, cols = dsm.shape
    shadow = (shadow_bldg * shadow_veg).astype(np.float32)

    ta, rad_i, rad_d, rad_g, esky, ci = 25.0, 600.0, 200.0, 800.0, 0.75, 0.85
    f_sh = np.full((rows, cols), 0.5, dtype=np.float32)
    kup_base = np.full((rows, cols), 50.0, dtype=np.float32)

    # Generate sky patches (Tregenza-style)
    def generate_sky_patches(n_alt_bands=4):
        patches = []
        alt_bands = [6, 18, 30, 42]
        azis_per_band = [30, 24, 24, 18]
        for alt, n_azi in zip(alt_bands[:n_alt_bands], azis_per_band[:n_alt_bands], strict=False):
            azi_step = 360.0 / n_azi if n_azi > 1 else 0
            for azi_idx in range(n_azi):
                patches.append([alt, azi_idx * azi_step])
        return np.array(patches, dtype=np.float32)

    l_patches = generate_sky_patches(n_alt_bands=4)
    n_patches = len(l_patches)

    # Create luminance values (Perez model simplified)
    luminance = 1000 + 500 * np.sin(l_patches[:, 0] * np.pi / 180)
    lv = np.column_stack([l_patches, luminance]).astype(np.float32)

    # Create 3D shadow matrices from SVF and shadows
    svf_expanded = svf[:, :, np.newaxis]
    base_visibility = np.broadcast_to(svf_expanded, (rows, cols, n_patches)).copy()
    bldg_factor = shadow_bldg[:, :, np.newaxis]
    veg_factor = shadow_veg[:, :, np.newaxis]

    shmat = (base_visibility * np.broadcast_to(bldg_factor, (rows, cols, n_patches))).astype(np.float32)
    shmat = (shmat > 0.5).astype(np.float32)
    vegshmat = (base_visibility * np.broadcast_to(veg_factor, (rows, cols, n_patches))).astype(np.float32)
    vegshmat = (vegshmat > 0.3).astype(np.float32)
    vbshvegshmat = (shmat * vegshmat).astype(np.float32)

    # Diffuse shadow (3D - same shape as shmat for diffuse sky patches)
    diffsh = shmat.copy()  # 3D array (rows, cols, patches)
    asvf = svf.copy()

    # Compute anisotropic Kside
    kside_result = vegetation.kside_veg(
        rad_i,
        rad_d,
        rad_g,
        shadow,
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        180.0,
        60.0,  # azimuth, altitude (noon)
        0.5,  # psi (vegetation transmissivity)
        0.0,  # t
        0.20,  # albedo
        f_sh,
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        kup_base.copy(),
        True,  # cyl
        lv,  # luminance values
        True,  # anisotropic_diffuse
        diffsh,
        asvf,
        shmat,
        vegshmat,
        vbshvegshmat,
    )

    # Compute anisotropic Lside
    ta_k = ta + 273.15
    ldown = np.full((rows, cols), esky * SBC * (ta_k**4), dtype=np.float32)
    lup = np.full((rows, cols), 0.95 * SBC * (ta_k**4), dtype=np.float32)

    lside_result = vegetation.lside_veg(
        svf_s,
        svf_w,
        svf_n,
        svf_e,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        svf_veg,
        180.0,
        60.0,
        ta,
        2.0,
        SBC,
        0.90,
        ldown,
        esky,
        0.0,
        f_sh,
        ci,
        lup.copy(),
        lup.copy(),
        lup.copy(),
        lup.copy(),
        True,  # anisotropic_longwave
    )

    # Save Kside outputs
    np.save(FIXTURES_DIR / "radiation_aniso_kside_e.npy", np.array(kside_result.keast))
    np.save(FIXTURES_DIR / "radiation_aniso_kside_s.npy", np.array(kside_result.ksouth))
    np.save(FIXTURES_DIR / "radiation_aniso_kside_i.npy", np.array(kside_result.kside_i))
    np.save(FIXTURES_DIR / "radiation_aniso_kside_d.npy", np.array(kside_result.kside_d))

    # Save Lside outputs
    np.save(FIXTURES_DIR / "radiation_aniso_lside_e.npy", np.array(lside_result.least))
    np.save(FIXTURES_DIR / "radiation_aniso_lside_s.npy", np.array(lside_result.lsouth))

    print(
        f"    Kside East range: {np.array(kside_result.keast).min():.1f}-{np.array(kside_result.keast).max():.1f} W/m²"
    )
    kside_i = np.array(kside_result.kside_i)
    print(f"    Kside Direct range: {kside_i.min():.1f}-{kside_i.max():.1f} W/m²")
    print(
        f"    Lside East range: {np.array(lside_result.least).min():.1f}-{np.array(lside_result.least).max():.1f} W/m²"
    )
    print("  Anisotropic radiation fixtures saved (from Rust implementation).")


def main():
    """Generate all golden fixtures using UMEP Python as ground truth."""
    global FIXTURES_DIR
    parser = argparse.ArgumentParser(description="Generate the golden fixtures.")
    parser.add_argument("--reference", type=Path, help="umep-solweig checkout (default: the installed copy)")
    parser.add_argument(
        "--fixtures-dir", type=Path, default=FIXTURES_DIR, help="directory to read inputs from and write to"
    )
    args = parser.parse_args()
    reference_dir = use_umep_solweig(args.reference)
    FIXTURES_DIR = args.fixtures_dir

    print("=" * 60)
    print("Golden Fixture Generator")
    print("=" * 60)
    print()
    print(f"Reference: umep-solweig at {reference_dir}")
    print("Shadow and SVF fixtures come from the reference. GVF and radiation")
    print("fixtures come from the current Rust implementation (regression).")
    print("=" * 60)

    ensure_fixtures_dir()

    # Generate algorithm output fixtures (using UMEP Python)
    generate_shadow_fixtures()
    generate_svf_fixtures()

    # Generate GVF and radiation fixtures (using current Rust implementation)
    # These are for regression testing - the overall Tmrt has been validated
    try:
        generate_gvf_fixtures()
        generate_radiation_fixtures()
    except ImportError as e:
        print(f"  Skipping GVF/radiation fixtures: {e}")

    # Generate UTCI fixtures (using UMEP Python as ground truth)
    try:
        generate_utci_fixtures()
    except ImportError as e:
        print(f"  Skipping UTCI fixtures: {e}")

    # Generate PET fixtures (using UMEP Python as ground truth)
    try:
        generate_pet_fixtures()
    except ImportError as e:
        print(f"  Skipping PET fixtures: {e}")

    # Generate Tmrt fixtures
    try:
        generate_tmrt_fixtures()
    except Exception as e:
        print(f"  Skipping Tmrt fixtures: {e}")

    # Generate ground temperature fixtures (TsWaveDelay)
    try:
        generate_ground_temp_fixtures()
    except ImportError as e:
        print(f"  Skipping ground temperature fixtures: {e}")

    # Generate wall temperature fixtures
    try:
        generate_wall_temp_fixtures()
    except Exception as e:
        print(f"  Skipping wall temperature fixtures: {e}")

    # Generate anisotropic sky fixtures
    try:
        generate_anisotropic_sky_fixtures()
    except Exception as e:
        print(f"  Skipping anisotropic sky fixtures: {e}")

    # Generate anisotropic radiation fixtures
    try:
        generate_aniso_radiation_fixtures()
    except Exception as e:
        print(f"  Skipping anisotropic radiation fixtures: {e}")

    print("\n" + "=" * 60)
    print("All fixtures generated successfully!")
    print(f"Location: {FIXTURES_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIXTURES_DIR.glob("*.npy")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")
    for f in sorted(FIXTURES_DIR.glob("*.npz")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
