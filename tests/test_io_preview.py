"""Preview PNGs written next to float rasters."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from solweig.io_preview import _generate_preview_png

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _gradient(side: int) -> np.ndarray:
    return np.linspace(0.0, 80.0, side * side, dtype=np.float32).reshape(side, side)


def test_preview_memory_does_not_scale_with_the_raster(tmp_path):
    """A 4096 px raster must not be colour-mapped at full resolution.

    Mapping costs about 83 bytes per pixel mapped (measured). At full
    resolution that is 1.4 GB here; thinned to 1024 px a side it is 87 MB.
    """
    data = _gradient(4096)
    out = tmp_path / "tmrt_big.tif"
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        _generate_preview_png(data, out)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak - base < 8 * data.size  # 134 MB: ten times below the full-resolution cost
    with Image.open(out.with_suffix(".preview.png")) as img:
        assert img.size == (512, 512)


def test_thinned_preview_matches_the_full_resolution_one(tmp_path):
    """Thinning changes a smooth field's thumbnail by a few grey levels at most."""
    data = _gradient(4096)
    thinned = tmp_path / "tmrt_a.tif"
    _generate_preview_png(data, thinned)
    # max_size large enough that no thinning happens, then the same final size
    full = tmp_path / "tmrt_b.tif"
    _generate_preview_png(data, full, max_size=4096)
    with Image.open(full.with_suffix(".preview.png")) as img:
        reference = np.asarray(img.resize((512, 512), Image.Resampling.LANCZOS), dtype=np.int16)
    with Image.open(thinned.with_suffix(".preview.png")) as img:
        got = np.asarray(img, dtype=np.int16)
    assert np.abs(got - reference).max() <= 8


def test_small_raster_is_not_thinned(tmp_path):
    """Below 4 x max_size the code path is the one it always was."""
    data = _gradient(2047)
    out = tmp_path / "tmrt_small.tif"
    _generate_preview_png(data, out)
    with Image.open(out.with_suffix(".preview.png")) as img:
        assert img.size == (512, 512)
        got = np.asarray(img).copy()

    import matplotlib.pyplot as plt

    colored = plt.get_cmap("turbo")(np.clip(data / 80.0, 0, 1))
    expected = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8), mode="RGB")
    expected = expected.resize((512, 512), Image.Resampling.LANCZOS)
    np.testing.assert_array_equal(got, np.asarray(expected))
