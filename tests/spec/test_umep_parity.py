"""Parity tests: local Python modules vs upstream UMEP equivalents.

Modules kept locally (because production code cannot depend on UMEP)
are validated here against the upstream UMEP implementations to ensure
they remain in sync.

Local modules tested:
- physics/create_patches.py  — used by precomputed.py (production)
- physics/patch_radiation.py — patch_steradians() used by precomputed.py (production)
"""

import numpy as np
import pytest

# ── Reference imports (skip entire module if umep-solweig is not installed) ──
from tests.reference_package import reference_module  # noqa: E402

umep_patches = reference_module("util.SEBESOLWEIGCommonFiles.create_patches")
umep_patch_rad = reference_module("functions.SOLWEIGpython.patch_radiation")
umep_perez = reference_module("util.SEBESOLWEIGCommonFiles.Perez_v3")

from solweig.physics.create_patches import create_patches as local_create_patches  # noqa: E402
from solweig.physics.patch_radiation import patch_steradians as local_patch_steradians  # noqa: E402


class TestCreatePatchesParity:
    """Local create_patches must exactly match UMEP create_patches."""

    @pytest.mark.parametrize("patch_option", [1, 2, 3, 4])
    def test_all_outputs_match(self, patch_option):
        """Every return value must be identical for all patch options."""
        local = local_create_patches(patch_option)
        umep = umep_patches.create_patches(patch_option)

        assert len(local) == len(umep), "Different number of return values"

        names = [
            "skyvaultalt",
            "skyvaultazi",
            "annulino",
            "skyvaultaltint",
            "patches_in_band",
            "skyvaultaziint",
            "azistart",
        ]
        for i, name in enumerate(names):
            np.testing.assert_array_equal(
                np.asarray(local[i]),
                np.asarray(umep[i]),
                err_msg=f"create_patches({patch_option}): {name} differs",
            )

    @pytest.mark.parametrize("patch_option,expected_count", [(1, 145), (2, 153)])
    def test_patch_count(self, patch_option, expected_count):
        """Number of patches matches expected for standard options."""
        alt, _, _, _, _, _, _ = local_create_patches(patch_option)
        assert alt.size == expected_count


class TestPatchSteradiansParity:
    """Local patch_steradians must match UMEP patch_steradians."""

    @pytest.mark.parametrize("patch_option", [1, 2, 3])
    def test_steradians_match(self, patch_option):
        """Steradian values must match UMEP for each patch option."""
        # Generate lv array via UMEP Perez (used as input to steradians)
        lv, _, _ = umep_perez.Perez_v3(30.0, 180.0, 200.0, 400.0, 180, patchchoice=1, patch_option=patch_option)

        local_ster, _, _ = local_patch_steradians(lv)
        umep_ster, _, _ = umep_patch_rad.patch_steradians(lv)

        np.testing.assert_allclose(
            local_ster,
            umep_ster,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"patch_steradians differs for patch_option={patch_option}",
        )

    @pytest.mark.parametrize("patch_option", [1, 2, 3])
    def test_steradians_sum_to_2pi(self, patch_option):
        """Steradians should sum to approximately 2*pi (hemisphere)."""
        lv, _, _ = umep_perez.Perez_v3(30.0, 180.0, 200.0, 400.0, 180, patchchoice=1, patch_option=patch_option)
        local_ster, _, _ = local_patch_steradians(lv)
        ster_sum = local_ster.sum()
        # Hemisphere = 2*pi steradians
        np.testing.assert_allclose(ster_sum, 2 * np.pi, rtol=0.05, err_msg="Steradians don't sum to ~2*pi")
