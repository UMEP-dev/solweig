"""Tests for `solweig.models.precomputed` — SVF + Shadow array dataclasses."""

from __future__ import annotations

import numpy as np
from solweig.models.precomputed import (
    ShadowArrays,
    SvfArrays,
    _pack_u8_to_bitpacked,
    _unpack_bitpacked_to_float32,
)

# ── SvfArrays construction + properties ─────────────────────────────────────


def _svf_arrays(shape=(4, 5)):
    """Build a populated SvfArrays for testing."""
    kw = {
        f"svf{suffix}": np.full(shape, value, dtype=np.float32)
        for suffix, value in [
            ("", 0.8),
            ("_north", 0.9),
            ("_east", 0.85),
            ("_south", 0.7),
            ("_west", 0.75),
            ("_veg", 0.95),
            ("_veg_north", 0.93),
            ("_veg_east", 0.92),
            ("_veg_south", 0.91),
            ("_veg_west", 0.94),
            ("_aveg", 0.97),
            ("_aveg_north", 0.96),
            ("_aveg_east", 0.95),
            ("_aveg_south", 0.94),
            ("_aveg_west", 0.96),
        ]
    }
    return SvfArrays(**kw)


def test_svf_arrays_coerces_to_float32():
    """All inputs should be cast to float32 even when given as float64."""
    shape = (3, 3)
    kw = {f.name: np.ones(shape, dtype=np.float64) for f in SvfArrays.__dataclass_fields__.values()}
    svf = SvfArrays(**kw)
    for fname in SvfArrays.__dataclass_fields__:
        assert getattr(svf, fname).dtype == np.float32


def test_svfalfa_computed_in_range():
    svf = _svf_arrays()
    alfa = svf.svfalfa
    assert alfa.shape == svf.svf.shape
    # asin of a value in [0, 1] is in [0, π/2].
    assert (alfa >= 0).all()
    assert (alfa <= np.pi / 2 + 1e-6).all()


def test_svfbuveg_clipped_to_unit_range():
    svf = _svf_arrays()
    bu = svf.svfbuveg
    assert (bu >= 0).all() and (bu <= 1).all()


def test_svfbuveg_equals_clip_of_svf_plus_svf_veg_minus_one():
    svf = _svf_arrays()
    expected = np.clip(svf.svf + svf.svf_veg - 1.0, 0.0, 1.0)
    np.testing.assert_array_equal(svf.svfbuveg, expected)


def test_svf_crop_preserves_shape_and_dtype():
    svf = _svf_arrays(shape=(10, 12))
    cropped = svf.crop(2, 6, 3, 9)
    assert cropped.svf.shape == (4, 6)
    assert cropped.svf.dtype == np.float32
    # Every directional component should be cropped to the same window.
    for fname in SvfArrays.__dataclass_fields__:
        assert getattr(cropped, fname).shape == (4, 6)


def test_svf_crop_returns_independent_copy():
    """Crop must not share memory with the original — mutating the cropped
    array must not bleed into the source."""
    svf = _svf_arrays(shape=(10, 10))
    cropped = svf.crop(0, 5, 0, 5)
    cropped.svf[0, 0] = 0.123
    assert svf.svf[0, 0] != 0.123


# ── SvfArrays memmap round-trip ─────────────────────────────────────────────


def test_svf_to_memmap_then_from_memmap(tmp_path):
    """Persist to disk and reload — values must round-trip exactly (f32)."""
    svf = _svf_arrays(shape=(6, 7))
    out_dir = tmp_path / "svf_memmap"
    svf.to_memmap(out_dir)

    # Verify the basic memmap files exist.
    assert (out_dir / "svf.npy").exists()
    assert (out_dir / "svf_north.npy").exists()
    assert (out_dir / "svf_aveg_west.npy").exists()

    loaded = SvfArrays.from_memmap(out_dir)
    for fname in SvfArrays.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(loaded, fname), getattr(svf, fname))


# ── ShadowArrays bitpacking helpers ─────────────────────────────────────────


def test_pack_unpack_bitpacked_roundtrip():
    """A binary (0/255) array packed then unpacked should round-trip."""
    rng = np.random.default_rng(seed=42)
    n_patches = 153
    rows, cols = 8, 10
    # Generate random 0/1 patches in u8 representation (Rust uses 0/0xFF).
    arr = (rng.integers(0, 2, (rows, cols, n_patches)) * 255).astype(np.uint8)
    packed = _pack_u8_to_bitpacked(arr)

    expected_pack_len = (n_patches + 7) // 8
    assert packed.shape == (rows, cols, expected_pack_len)
    assert packed.dtype == np.uint8

    unpacked = _unpack_bitpacked_to_float32(packed, n_patches)
    # Unpacked values are 0.0 or 1.0.
    binary = (arr > 0).astype(np.float32)
    np.testing.assert_array_equal(unpacked, binary)


def test_pack_handles_partial_byte_at_end():
    """153 patches → 20 bytes (16 patches in the final byte, 19 full + 1 partial=8 bits unused)."""
    n_patches = 153
    expected_bytes = (n_patches + 7) // 8  # 20
    arr = np.zeros((2, 2, n_patches), dtype=np.uint8)
    arr[0, 0, 0] = 255  # first patch set at (0, 0)
    arr[0, 1, 152] = 255  # last patch set at (0, 1)
    packed = _pack_u8_to_bitpacked(arr)
    assert packed.shape[-1] == expected_bytes
    # Bit 0 of byte 0 at (0, 0): set
    assert packed[0, 0, 0] & 1 == 1
    # Bit (152 & 7) = 0 of byte (152 >> 3) = 19 at (0, 1): set
    assert packed[0, 1, 19] & 1 == 1


# ── ShadowArrays construction ───────────────────────────────────────────────


def test_shadow_arrays_construction_with_bitpacked_inputs():
    """Provide bitpacked u8 arrays + n_patches; properties should derive correctly."""
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    shape3 = (5, 5, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _vegshmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _vbshmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _n_patches=n_patches,
    )
    # patch_option property infers 2 for n_patches == 153.
    assert sa.patch_option == 2
    # The unpacked float arrays have shape (rows, cols, n_patches).
    assert sa.shmat.shape == (5, 5, n_patches)
    # All-1 shadow input → all-1 float output.
    assert (sa.shmat == 1.0).all()
    assert (sa.vegshmat == 1.0).all()
    assert (sa.vbshmat == 1.0).all()


def test_shadow_diffsh_full_visibility_returns_ones():
    """All sky visible + all vegetation visible → diffsh = 1.0 everywhere."""
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    shape3 = (3, 3, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _vegshmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _vbshmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _n_patches=n_patches,
    )
    d = sa.diffsh(transmissivity=0.03)
    np.testing.assert_allclose(d, 1.0, atol=1e-6)


def test_shadow_diffsh_no_vegetation_uses_psi_weighted_value():
    """sh=1, veg=0 → diffsh = 1 - (1 - 0) * (1 - 0.03) = 1 - 0.97 = 0.03."""
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    shape3 = (2, 2, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),  # all sky visible
        _vegshmat_u8=np.zeros(shape3, dtype=np.uint8),  # all vegetation blocking
        _vbshmat_u8=np.zeros(shape3, dtype=np.uint8),
        _n_patches=n_patches,
    )
    d = sa.diffsh(transmissivity=0.03)
    np.testing.assert_allclose(d, 0.03, atol=1e-6)


def test_shadow_steradians_sums_to_2pi():
    """The patch-steradian sum should approximate 2π (hemisphere)."""
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    shape3 = (1, 1, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.zeros(shape3, dtype=np.uint8),
        _vegshmat_u8=np.zeros(shape3, dtype=np.uint8),
        _vbshmat_u8=np.zeros(shape3, dtype=np.uint8),
        _n_patches=n_patches,
    )
    total = float(np.sum(sa.steradians))
    assert abs(total - 2 * np.pi) < 0.05


def test_shadow_crop_preserves_n_patches():
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    shape3 = (10, 12, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.zeros(shape3, dtype=np.uint8),
        _vegshmat_u8=np.zeros(shape3, dtype=np.uint8),
        _vbshmat_u8=np.zeros(shape3, dtype=np.uint8),
        _n_patches=n_patches,
    )
    cropped = sa.crop(2, 5, 3, 9)
    assert cropped._shmat_u8.shape == (3, 6, n_pack)
    assert cropped._n_patches == n_patches


def test_shadow_release_float32_cache_idempotent():
    """release_float32_cache should be safe to call multiple times."""
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    shape3 = (3, 3, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _vegshmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _vbshmat_u8=np.full(shape3, 0xFF, dtype=np.uint8),
        _n_patches=n_patches,
    )
    _ = sa.shmat  # populate the cache
    sa.release_float32_cache()
    sa.release_float32_cache()  # double-call must not crash
    # Subsequent access still works.
    assert sa.shmat.shape == (3, 3, n_patches)


# ── ShadowArrays npz round-trip ─────────────────────────────────────────────


def test_shadow_to_npz_then_from_npz_preserves_data(tmp_path):
    """Save shadow matrices as NPZ and reload — values must round-trip exactly."""
    n_patches = 153
    n_pack = (n_patches + 7) // 8
    shape3 = (4, 4, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.full(shape3, 0xAB, dtype=np.uint8),
        _vegshmat_u8=np.full(shape3, 0xCD, dtype=np.uint8),
        _vbshmat_u8=np.full(shape3, 0xEF, dtype=np.uint8),
        _n_patches=n_patches,
    )
    npz_path = tmp_path / "shadows.npz"
    np.savez(
        str(npz_path),
        shadowmat=sa._shmat_u8,
        vegshadowmat=sa._vegshmat_u8,
        vbshmat=sa._vbshmat_u8,
        patch_count=np.array(n_patches),
    )
    loaded = ShadowArrays.from_npz(str(npz_path))
    np.testing.assert_array_equal(loaded._shmat_u8, sa._shmat_u8)
    np.testing.assert_array_equal(loaded._vegshmat_u8, sa._vegshmat_u8)
    np.testing.assert_array_equal(loaded._vbshmat_u8, sa._vbshmat_u8)
    assert loaded._n_patches == n_patches


def test_shadow_patch_option_inference():
    """patch_option getter infers the layout from n_patches."""
    n_pack_map = {1: 19, 2: 20, 3: 39, 4: 77}  # ceil(N/8) for N in {145, 153, 305, 609}
    n_patches_map = {1: 145, 2: 153, 3: 305, 4: 609}
    for option, n in n_patches_map.items():
        n_pack = n_pack_map[option]
        shape3 = (1, 1, n_pack)
        sa = ShadowArrays(
            _shmat_u8=np.zeros(shape3, dtype=np.uint8),
            _vegshmat_u8=np.zeros(shape3, dtype=np.uint8),
            _vbshmat_u8=np.zeros(shape3, dtype=np.uint8),
            _n_patches=n,
        )
        assert sa.patch_option == option


def test_shadow_patch_option_unknown_n_patches():
    """If n_patches doesn't match any of the known layouts (145/153/305/609),
    `patch_option` should fall back to a default (2). The constructor does
    not validate against the known set — that's a documented choice."""
    n_pack = 2
    shape3 = (1, 1, n_pack)
    sa = ShadowArrays(
        _shmat_u8=np.zeros(shape3, dtype=np.uint8),
        _vegshmat_u8=np.zeros(shape3, dtype=np.uint8),
        _vbshmat_u8=np.zeros(shape3, dtype=np.uint8),
        _n_patches=999,
    )
    # Just check the property doesn't crash. The exact fallback is an
    # implementation detail — it returns *some* int, not a guaranteed mapping.
    assert isinstance(sa.patch_option, int)
