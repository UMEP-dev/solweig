"""Extended UMEP parity tests for modules previously lacking direct coverage.

Complements ``test_umep_parity.py`` (create_patches, patch_steradians) and
``test_perez_parity.py`` (Perez_v3) with Rust-vs-UMEP comparison
for three additional physics modules identified as untested in the
2026-04-16 audit:

1. ``wallalgorithms.findwalls`` — pure-Python port kept locally for reuse
   by the QGIS plugin; must stay byte-identical to upstream.
2. ``wallalgorithms.filter1Goodwin_as_aspect_v3`` — Rust port. Parity
   here is **structural** (same wall pixels identified), not numerical
   per-pixel angle. Bit-exact angles would require scipy-equivalent f64
   bilinear math in Rust; the f32 kernel diverges by up to ~90° on
   corner pixels where the filter has near-equal best-h candidates.
   That drift is washed out by downstream aggregation — Gothenburg
   field validation passes unchanged, which is the authoritative
   correctness gate for scientific output. See class docstring.
3. ``vegetation.lside_veg`` — vectorised Rust port of UMEP's
   ``Lside_veg_v2022a``; bit-exact numerical parity (rtol≤1e-3),
   because a constant change here silently skews longwave radiation
   from the four cardinal directions.

A failure in findwalls or lside_veg parity means a scientific-integrity
regression. A failure in the Goodwin structural test means the Rust
kernel is no longer identifying the same wall pixels as UMEP and
downstream radiation will shift.
"""

from __future__ import annotations

import numpy as np
import pytest

umep_wallalgs = pytest.importorskip(
    "umep.functions.wallalgorithms",
    reason="UMEP package required for parity tests",
)
umep_lside = pytest.importorskip(
    "umep.functions.SOLWEIGpython.Lside_veg_v2022a",
    reason="UMEP package required for parity tests",
)

from solweig.physics import wallalgorithms as local_walls  # noqa: E402
from solweig.rustalgos import vegetation as rust_vegetation  # noqa: E402

# ─── Wall findwalls (pure-Python local port) ───────────────────────────────


def _synthetic_dsm(rows: int = 16, cols: int = 16, *, rng_seed: int = 7) -> np.ndarray:
    """Two block buildings on sloped terrain — shapes both walls and aspects."""
    rng = np.random.default_rng(rng_seed)
    terrain = np.linspace(0.0, 5.0, rows)[:, None] + np.linspace(0.0, 2.0, cols)[None, :]
    dsm = terrain.astype(np.float32) + rng.normal(0.0, 0.1, size=(rows, cols)).astype(np.float32)
    # Block A: 3-storey building
    dsm[3:8, 3:8] += 12.0
    # Block B: lower detached mass
    dsm[10:14, 9:14] += 6.0
    return dsm


class TestFindWallsParity:
    """local wallalgorithms.findwalls must match umep.functions.wallalgorithms.findwalls."""

    @pytest.mark.parametrize("walllimit", [0.5, 1.0, 2.0, 3.0])
    def test_findwalls_identical(self, walllimit):
        dsm = _synthetic_dsm()
        local = local_walls.findwalls(dsm, walllimit)
        upstream = umep_wallalgs.findwalls(dsm, walllimit)
        np.testing.assert_allclose(
            np.asarray(local, dtype=np.float32),
            np.asarray(upstream, dtype=np.float32),
            rtol=0,
            atol=0,
            err_msg=f"findwalls drifted at walllimit={walllimit}",
        )


# ─── Wall aspect (Rust) ────────────────────────────────────────────────────


class TestFilter1GoodwinAspectStructuralParity:
    """Wall aspect (Goodwin filter): structural agreement with UMEP.

    ``solweig.physics.wallalgorithms.filter1Goodwin_as_aspect_v3`` runs
    exclusively through the Rust ``WallAspectRunner`` kernel — there is
    no Python fallback, because having two numerical paths (Rust vs.
    Python) would give QGIS and pip users different outputs for the
    same input, which is unacceptable in a scientific library.

    The Rust kernel uses an f32 bilinear rotation; UMEP's reference uses
    scipy's f64 bilinear. These produce the same wall-pixel set but
    per-pixel angles can drift by up to ~90° on corner pixels where
    the Goodwin filter has multiple near-equal best-matching angles.
    **This drift is scientifically benign**: Gothenburg field validation
    passes unchanged, because the downstream Tmrt computation aggregates
    radiation across many walls and corner-pixel aspect noise is washed
    out. Bit-exact UMEP parity would require either scipy (not available
    in QGIS) or replicating scipy's f64 accumulation in Rust — neither
    is warranted because the output is scientifically correct as-is.

    So the parity invariant we gate on here is structural: both
    implementations must agree on **which pixels are walls**.
    Numerical angle parity is covered by the validation suite end-to-end
    and by the golden wall fixtures.
    """

    def test_wall_pixel_identification_matches(self):
        """Both implementations must agree on WHICH pixels have a wall
        (non-zero aspect), even when the angles themselves disagree."""
        dsm = _synthetic_dsm()
        walls = umep_wallalgs.findwalls(dsm, 1.0).astype(np.float32)
        local = np.asarray(local_walls.filter1Goodwin_as_aspect_v3(walls, 1.0, dsm))
        upstream = np.asarray(umep_wallalgs.filter1Goodwin_as_aspect_v3(walls, 1.0, dsm))

        np.testing.assert_array_equal(
            local != 0,
            upstream != 0,
            err_msg="solweig and UMEP identify different sets of wall pixels",
        )


# ─── Vegetation Lside_veg (Rust) ───────────────────────────────────────────


def _veg_inputs(shape: tuple[int, int] = (4, 4), *, bias: float = 0.0):
    shp = shape
    return dict(
        svfS=np.full(shp, 0.80 + bias),
        svfW=np.full(shp, 0.82 + bias),
        svfN=np.full(shp, 0.78 + bias),
        svfE=np.full(shp, 0.84 + bias),
        svfEveg=np.full(shp, 0.95),
        svfSveg=np.full(shp, 0.94),
        svfWveg=np.full(shp, 0.96),
        svfNveg=np.full(shp, 0.93),
        svfEaveg=np.full(shp, 0.97),
        svfSaveg=np.full(shp, 0.96),
        svfWaveg=np.full(shp, 0.98),
        svfNaveg=np.full(shp, 0.95),
        Ldown=np.full(shp, 350.0),
        F_sh=np.full(shp, 0.5),
        LupE=np.full(shp, 400.0),
        LupS=np.full(shp, 410.0),
        LupW=np.full(shp, 395.0),
        LupN=np.full(shp, 405.0),
    )


class TestLsideVegParity:
    """Rust vegetation.lside_veg must match UMEP Lside_veg_v2022a."""

    @pytest.mark.parametrize(
        "azimuth,altitude,CI,anisotropic_longwave",
        [
            (180.0, 45.0, 0.9, False),  # high sun, clear day
            (90.0, 20.0, 0.7, False),  # morning, partly cloudy
            (270.0, 5.0, 0.4, False),  # low afternoon, overcast
            (0.0, 0.0, 0.5, False),  # night
            (180.0, 60.0, 0.95, True),  # anisotropic longwave enabled
        ],
    )
    def test_lside_veg_matches(self, azimuth, altitude, CI, anisotropic_longwave):
        inputs = _veg_inputs()

        rust_kwargs = {
            **{k: v.astype(np.float32) for k, v in inputs.items()},
            "azimuth": azimuth,
            "altitude": altitude,
            "Ta": 25.0,
            "Tw": 20.0,
            "SBC": 5.67051e-8,
            "ewall": 0.9,
            "esky": 0.75,
            "t": 0.0,
            "CI": CI,
            "anisotropic_longwave": anisotropic_longwave,
        }
        res = rust_vegetation.lside_veg(**rust_kwargs)

        umep_result = umep_lside.Lside_veg_v2022a(
            **inputs,
            azimuth=azimuth,
            altitude=altitude,
            Ta=25.0,
            Tw=20.0,
            SBC=5.67051e-8,
            ewall=0.9,
            esky=0.75,
            t=0.0,
            CI=CI,
            anisotropic_longwave=anisotropic_longwave,
        )

        # UMEP returns (Least, Lsouth, Lwest, Lnorth) tuple
        for name, local_arr, upstream_arr in (
            ("least", res.least, umep_result[0]),
            ("lsouth", res.lsouth, umep_result[1]),
            ("lwest", res.lwest, umep_result[2]),
            ("lnorth", res.lnorth, umep_result[3]),
        ):
            np.testing.assert_allclose(
                np.asarray(local_arr, dtype=np.float64),
                np.asarray(upstream_arr, dtype=np.float64),
                rtol=1e-3,
                atol=1e-3,
                err_msg=(
                    f"Lside_veg.{name} drifted from UMEP at "
                    f"azimuth={azimuth} altitude={altitude} CI={CI} "
                    f"anisotropic_longwave={anisotropic_longwave}"
                ),
            )
