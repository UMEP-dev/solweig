"""Tests for SVF memmap caching functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from solweig.models.precomputed import SvfArrays


class TestSvfMemmap:
    """Tests for memory-mapped SVF storage."""

    @pytest.fixture
    def sample_svf_data(self):
        """Create sample SVF arrays for testing."""
        np.random.seed(42)
        size = 100

        return {
            "svf": np.random.rand(size, size).astype(np.float32),
            "svf_north": np.random.rand(size, size).astype(np.float32),
            "svf_east": np.random.rand(size, size).astype(np.float32),
            "svf_south": np.random.rand(size, size).astype(np.float32),
            "svf_west": np.random.rand(size, size).astype(np.float32),
            "svf_veg": np.random.rand(size, size).astype(np.float32),
            "svf_veg_north": np.random.rand(size, size).astype(np.float32),
            "svf_veg_east": np.random.rand(size, size).astype(np.float32),
            "svf_veg_south": np.random.rand(size, size).astype(np.float32),
            "svf_veg_west": np.random.rand(size, size).astype(np.float32),
            "svf_aveg": np.random.rand(size, size).astype(np.float32),
            "svf_aveg_north": np.random.rand(size, size).astype(np.float32),
            "svf_aveg_east": np.random.rand(size, size).astype(np.float32),
            "svf_aveg_south": np.random.rand(size, size).astype(np.float32),
            "svf_aveg_west": np.random.rand(size, size).astype(np.float32),
        }

    def test_save_and_load_memmap(self, sample_svf_data):
        """Test saving and loading SVF arrays as memmap."""
        svf = SvfArrays(**sample_svf_data)

        # ignore_cleanup_errors: the loaded arrays are still mapped when the
        # block ends, and Windows will not delete a mapped file.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            cache_dir = Path(tmpdir) / "svf_cache"

            # Save to memmap
            result_dir = svf.to_memmap(cache_dir)
            assert result_dir.exists()

            # Verify files were created
            assert (cache_dir / "svf.npy").exists()
            assert (cache_dir / "svf_north.npy").exists()

            # Load from memmap
            svf_loaded = SvfArrays.from_memmap(cache_dir)

            # Verify data matches
            assert np.allclose(svf.svf, svf_loaded.svf)
            assert np.allclose(svf.svf_north, svf_loaded.svf_north)
            assert np.allclose(svf.svf_veg, svf_loaded.svf_veg)

    def test_memmap_preserves_dtype(self, sample_svf_data):
        """Verify memmap arrays maintain float32 dtype."""
        svf = SvfArrays(**sample_svf_data)

        # ignore_cleanup_errors: the loaded arrays are still mapped when the
        # block ends, and Windows will not delete a mapped file.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            cache_dir = Path(tmpdir) / "svf_cache"
            svf.to_memmap(cache_dir)
            svf_loaded = SvfArrays.from_memmap(cache_dir)

            # Check dtype is preserved
            assert svf_loaded.svf.dtype == np.float32
            assert svf_loaded.svf_veg.dtype == np.float32

    def test_memmap_is_actually_memmap(self, sample_svf_data):
        """Verify loaded arrays are actually memory-mapped."""
        svf = SvfArrays(**sample_svf_data)

        # ignore_cleanup_errors: the loaded arrays are still mapped when the
        # block ends, and Windows will not delete a mapped file.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            cache_dir = Path(tmpdir) / "svf_cache"
            svf.to_memmap(cache_dir)
            svf_loaded = SvfArrays.from_memmap(cache_dir)

            # Verify it's a memmap
            assert isinstance(svf_loaded.svf, np.memmap)
            assert isinstance(svf_loaded.svf_north, np.memmap)

    def test_memmap_slicing_works(self, sample_svf_data):
        """Test that slicing memmap arrays works correctly."""
        svf = SvfArrays(**sample_svf_data)

        # ignore_cleanup_errors: the loaded arrays are still mapped when the
        # block ends, and Windows will not delete a mapped file.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            cache_dir = Path(tmpdir) / "svf_cache"
            svf.to_memmap(cache_dir)
            svf_loaded = SvfArrays.from_memmap(cache_dir)

            # Test slicing (simulates tiled access)
            tile = svf_loaded.svf[20:40, 30:50]
            assert tile.shape == (20, 20)
            assert np.allclose(tile, svf.svf[20:40, 30:50])

    def test_memmap_computed_properties_work(self, sample_svf_data):
        """Test that computed properties (svfalfa, svfbuveg) work with memmap."""
        svf = SvfArrays(**sample_svf_data)

        # ignore_cleanup_errors: the loaded arrays are still mapped when the
        # block ends, and Windows will not delete a mapped file.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            cache_dir = Path(tmpdir) / "svf_cache"
            svf.to_memmap(cache_dir)
            svf_loaded = SvfArrays.from_memmap(cache_dir)

            # Computed properties should work
            svfalfa = svf_loaded.svfalfa
            svfbuveg = svf_loaded.svfbuveg

            assert svfalfa.shape == svf_loaded.svf.shape
            assert svfbuveg.shape == svf_loaded.svf.shape

            # Values should match
            assert np.allclose(svfalfa, svf.svfalfa)
            assert np.allclose(svfbuveg, svf.svfbuveg)

    def test_from_memmap_nonexistent_raises(self):
        """Test that loading from nonexistent directory raises error."""
        with pytest.raises(FileNotFoundError):
            SvfArrays.from_memmap("/nonexistent/path")

    def test_from_memmap_missing_file_raises(self, sample_svf_data):
        """Test that missing files raise error."""
        svf = SvfArrays(**sample_svf_data)

        # ignore_cleanup_errors: the loaded arrays are still mapped when the
        # block ends, and Windows will not delete a mapped file.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            cache_dir = Path(tmpdir) / "svf_cache"
            svf.to_memmap(cache_dir)

            # Delete one file
            (cache_dir / "svf.npy").unlink()

            with pytest.raises(FileNotFoundError):
                SvfArrays.from_memmap(cache_dir)
