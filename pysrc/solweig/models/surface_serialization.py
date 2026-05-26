"""Disk-export helpers for SVF / shadow-matrix caches.

Extracted from ``models/surface.py`` to keep the SurfaceData module focused
on data semantics rather than file I/O. These helpers are stateless — they
take a populated :class:`SvfArrays` or shadow-matrix result and write it
to a ``working_dir`` cache directory in the format expected by
:class:`PrecomputedData.prepare`.

All exporters here:

- Honour the ``SOLWEIG_COMPRESS_MAX_PIXELS`` and
  ``SOLWEIG_SHADOW_NPZ_MAX_PIXELS`` env-var thresholds (compression is
  skipped for very large rasters to avoid a long single-threaded tail).
- Are silent / log-only on failure; the memmap cache is the fallback when
  the zip/npz export is unavailable.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .. import io
from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from .precomputed import SvfArrays

logger = get_logger(__name__)


def should_compress_svf_exports(n_pixels: int) -> bool:
    """Return True when SVF / shadow exports should use compression.

    Large rasters spend a long single-threaded tail in compression after
    the GPU work completes. Override the default threshold via
    ``SOLWEIG_COMPRESS_MAX_PIXELS``.
    """
    try:
        limit = int(os.getenv("SOLWEIG_COMPRESS_MAX_PIXELS", "50000000"))
    except ValueError:
        limit = 50_000_000
    return n_pixels <= max(0, limit)


def should_export_shadow_npz(n_pixels: int) -> bool:
    """Return True when ``shadowmats.npz`` should be written.

    For very large grids, serializing 3 bitpacked matrices into one NPZ can
    dominate runtime after GPU work completes. For those cases we keep the
    memmap cache and skip NPZ export by default. Override with
    ``SOLWEIG_FORCE_SHADOW_NPZ=1`` or
    ``SOLWEIG_SHADOW_NPZ_MAX_PIXELS=<n>``.
    """
    force = os.getenv("SOLWEIG_FORCE_SHADOW_NPZ", "").strip().lower() in ("1", "true")
    if force:
        return True
    try:
        limit = int(os.getenv("SOLWEIG_SHADOW_NPZ_MAX_PIXELS", "50000000"))
    except ValueError:
        limit = 50_000_000
    return n_pixels <= max(0, limit)


def save_svfs_zip(
    svf_data: SvfArrays,
    svf_cache_dir: Path,
    aligned_rasters: dict,
    *,
    compress: bool = True,
) -> None:
    """Save SVF arrays as ``svfs.zip`` for :meth:`PrecomputedData.prepare`.

    Skips silently when no geotransform is available (the memmap cache is
    used in that case).
    """
    geotransform = aligned_rasters.get("dsm_transform")
    crs_wkt = aligned_rasters.get("dsm_crs")

    if geotransform is None:
        logger.debug("  Skipping svfs.zip (no geotransform available)")
        return

    svf_files = {
        "svf.tif": svf_data.svf,
        "svfN.tif": svf_data.svf_north,
        "svfE.tif": svf_data.svf_east,
        "svfS.tif": svf_data.svf_south,
        "svfW.tif": svf_data.svf_west,
        "svfveg.tif": svf_data.svf_veg,
        "svfNveg.tif": svf_data.svf_veg_north,
        "svfEveg.tif": svf_data.svf_veg_east,
        "svfSveg.tif": svf_data.svf_veg_south,
        "svfWveg.tif": svf_data.svf_veg_west,
        "svfaveg.tif": svf_data.svf_aveg,
        "svfNaveg.tif": svf_data.svf_aveg_north,
        "svfEaveg.tif": svf_data.svf_aveg_east,
        "svfSaveg.tif": svf_data.svf_aveg_south,
        "svfWaveg.tif": svf_data.svf_aveg_west,
    }

    if hasattr(geotransform, "to_gdal"):
        geotransform = list(geotransform.to_gdal())

    svf_zip_path = svf_cache_dir / "svfs.zip"
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, arr in svf_files.items():
            if arr is not None:
                tif_path = str(Path(tmpdir) / filename)
                # Intermediate export for zip packaging: skip COG/preview overhead.
                io.save_raster(
                    tif_path,
                    arr,
                    geotransform,
                    crs_wkt,
                    use_cog=False,
                    generate_preview=False,
                )
        compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        with zipfile.ZipFile(str(svf_zip_path), "w", compression=compression) as zf:
            for filename in svf_files:
                tif_file = Path(tmpdir) / filename
                if tif_file.exists():
                    zf.write(str(tif_file), filename)

    mode = "compressed" if compress else "stored (uncompressed)"
    logger.info(f"  ✓ SVF saved as {svf_zip_path} ({mode})")


def save_shadow_matrices(
    svf_result,
    svf_cache_dir: Path,
    patch_count: int = 153,
    *,
    compress: bool = True,
) -> None:
    """Save shadow matrices as ``shadowmats.npz`` for the anisotropic sky model."""
    shadow_path = svf_cache_dir / "shadowmats.npz"
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        str(shadow_path),
        shadowmat=np.array(svf_result.bldg_sh_matrix),
        vegshadowmat=np.array(svf_result.veg_sh_matrix),
        vbshmat=np.array(svf_result.veg_blocks_bldg_sh_matrix),
        patch_count=np.array(patch_count),
    )
    mode = "compressed" if compress else "uncompressed"
    logger.info(f"  ✓ Shadow matrices saved as {shadow_path} ({mode})")
