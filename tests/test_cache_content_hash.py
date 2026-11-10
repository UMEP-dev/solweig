"""Cache validation must notice a small edit to a large raster.

The sampled hash reads 10,000 evenly spaced values, so a one-building edit to
a city-scale DSM usually leaves it unchanged and a stale SVF cache would be
reused. The content hash reads every value.
"""

from __future__ import annotations

import numpy as np
from solweig.cache import CacheMetadata, compute_array_content_hash, compute_array_hash, validate_cache

SHAPE = (1200, 1200)  # 1.44 M values: the sampled hash reads under 1 % of them


def _dsm():
    rng = np.random.default_rng(3)
    return rng.uniform(0.0, 30.0, SHAPE).astype(np.float32)


def _edit_unsampled_pixel(dsm):
    """Change one value the sampled hash does not read."""
    sampled = set(np.linspace(0, dsm.size - 1, 10000, dtype=np.int64).tolist())
    flat_index = next(i for i in range(dsm.size // 2, dsm.size) if i not in sampled)
    edited = dsm.copy()
    edited.ravel()[flat_index] += 12.0
    return edited


def test_sampled_hash_misses_a_single_pixel_edit_and_content_hash_catches_it():
    dsm = _dsm()
    edited = _edit_unsampled_pixel(dsm)
    assert compute_array_hash(dsm) == compute_array_hash(edited)
    assert compute_array_content_hash(dsm) != compute_array_content_hash(edited)


def test_content_hash_ignores_memory_layout(tmp_path):
    dsm = _dsm()
    reference = compute_array_content_hash(dsm)
    assert compute_array_content_hash(np.asfortranarray(dsm)) == reference
    mm = np.memmap(tmp_path / "dsm.dat", dtype=np.float32, mode="w+", shape=SHAPE)
    mm[:] = dsm
    assert compute_array_content_hash(mm) == reference


def test_content_hash_depends_on_shape_and_dtype():
    values = np.arange(12, dtype=np.float32)
    assert compute_array_content_hash(values.reshape(3, 4)) != compute_array_content_hash(values.reshape(4, 3))
    assert compute_array_content_hash(values) != compute_array_content_hash(values.astype(np.float64))


def test_cache_goes_stale_on_a_small_edit(tmp_path):
    dsm = _dsm()
    CacheMetadata.from_arrays(dsm, 1.0).save(tmp_path)
    assert validate_cache(tmp_path, dsm, 1.0)
    assert not validate_cache(tmp_path, _edit_unsampled_pixel(dsm), 1.0)


def test_cache_goes_stale_on_a_small_canopy_edit(tmp_path):
    dsm, cdsm = _dsm(), _dsm()
    CacheMetadata.from_arrays(dsm, 1.0, cdsm, cdsm * 0.25).save(tmp_path)
    assert validate_cache(tmp_path, dsm, 1.0, cdsm, cdsm * 0.25)
    assert not validate_cache(tmp_path, dsm, 1.0, _edit_unsampled_pixel(cdsm), cdsm * 0.25)


def test_metadata_without_content_hash_still_validates(tmp_path):
    """Caches written before the content hash existed are not invalidated."""
    dsm = _dsm()
    current = CacheMetadata.from_arrays(dsm, 1.0)
    legacy = CacheMetadata(dsm_hash=current.dsm_hash, dsm_shape=current.dsm_shape, pixel_size=1.0)
    legacy.save(tmp_path)
    assert validate_cache(tmp_path, dsm, 1.0)
    loaded = CacheMetadata.load(tmp_path)
    assert loaded is not None and loaded.dsm_content_hash is None
