"""Wall and SVF computation + on-disk caching for :meth:`SurfaceData.prepare`.

Extracted from ``models/surface.py`` (the largest hot file). These
helpers were previously ``@staticmethod`` on :class:`SurfaceData` but
took ``surface_data`` as the first explicit argument — they had no real
need for class scoping.

The three functions:

- :func:`compute_and_cache_walls` — call the Goodwin filter on the
  aligned DSM, write wall_hts.tif + wall_aspects.tif to the
  pixel-size-keyed cache subdirectory, attach the loaded arrays to
  ``surface_data``.
- :func:`compute_and_cache_svf` — run the (possibly tiled) SVF kernel
  on the aligned DSM/CDSM/TDSM, persist the result as both memmap and
  zip, write shadow matrices as npz when the raster is small enough,
  attach :class:`SvfArrays` + :class:`ShadowArrays` to ``surface_data``.
- :func:`_compute_svf_tiled` — internal helper used by
  ``compute_and_cache_svf`` for large grids that exceed GPU memory.

All three deferred-import :class:`SurfaceData` / :class:`ShadowArrays`
to avoid a circular dependency on ``surface.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import io
from .. import walls as walls_module
from ..cache import CacheMetadata, pixel_size_tag
from ..rustalgos import skyview
from ..solweig_logging import get_logger
from .surface_serialization import (
    save_shadow_matrices as _save_shadow_matrices,
)
from .surface_serialization import (
    save_svfs_zip as _save_svfs_zip,
)
from .surface_serialization import (
    should_compress_svf_exports as _should_compress_svf_exports,
)
from .surface_serialization import (
    should_export_shadow_npz as _should_export_shadow_npz,
)

if TYPE_CHECKING:
    from .surface import SurfaceData

logger = get_logger(__name__)


def _max_shadow_height(dsm: np.ndarray, cdsm: np.ndarray | None = None, use_veg: bool = False) -> float:
    """Estimate maximum casting height above local ground.

    Mirrors `surface._max_shadow_height` (kept private there for use by
    other surface helpers); duplicated here so this module is self-contained.
    """
    dsm_max = float(np.nanmax(dsm))
    dsm_min = float(np.nanmin(dsm))
    base_height = dsm_max - dsm_min
    if use_veg and cdsm is not None:
        cdsm_max = float(np.nanmax(cdsm))
        # Use absolute CDSM max if it's higher than DSM relief.
        return max(base_height, cdsm_max - dsm_min)
    return base_height


def compute_and_cache_walls(
    surface_data: SurfaceData,
    aligned_rasters: dict,
    working_path: Path,
    *,
    pixel_size: float = 1.0,
    feedback: Any = None,
    progress_range: tuple[float, float] | None = None,
) -> None:
    """
    Compute wall heights/aspects from DSM and cache to working_dir.

    Args:
        surface_data: SurfaceData instance to update with computed walls.
        aligned_rasters: Dictionary with aligned raster data.
        working_path: Working directory for caching.
        pixel_size: Pixel size in metres (for pixel-size-keyed cache path).
        feedback: Optional QGIS QgsProcessingFeedback for progress/cancellation.
        progress_range: Optional (start_pct, end_pct) for QGIS progress sub-range.
    """
    logger.info("Computing walls from DSM and caching to working_dir...")
    walls_cache_dir = working_path / "walls" / pixel_size_tag(pixel_size)

    # Save resampled DSM to working_dir so wall computation can use it
    resampled_dir = working_path / "resampled"
    resampled_dir.mkdir(parents=True, exist_ok=True)
    resampled_dsm_path = resampled_dir / "dsm_resampled.tif"

    dsm_transform = aligned_rasters["dsm_transform"]
    io.save_raster(
        str(resampled_dsm_path),
        aligned_rasters["dsm_arr"],
        list(dsm_transform.to_gdal()) if hasattr(dsm_transform, "to_gdal") else dsm_transform,
        aligned_rasters["dsm_crs"],
    )

    # Generate walls using the walls module
    walls_module.generate_wall_hts(
        dsm_path=str(resampled_dsm_path),
        bbox=None,  # Already resampled to target extent
        out_dir=str(walls_cache_dir),
        feedback=feedback,
        progress_range=progress_range,
    )

    # Load the generated walls back into surface_data
    wall_hts_path = walls_cache_dir / "wall_hts.tif"
    wall_aspects_path = walls_cache_dir / "wall_aspects.tif"

    if wall_hts_path.exists() and wall_aspects_path.exists():
        wall_height_arr, _, _, _ = io.load_raster(str(wall_hts_path))
        wall_aspect_arr, _, _, _ = io.load_raster(str(wall_aspects_path))
        surface_data.wall_height = wall_height_arr
        surface_data.wall_aspect = wall_aspect_arr

        # Save cache metadata for wall validation on future runs
        dsm_arr = aligned_rasters["dsm_arr"]
        cdsm_arr = aligned_rasters.get("cdsm_arr")
        wall_pixel_size = aligned_rasters.get("pixel_size", pixel_size)
        metadata = CacheMetadata.from_arrays(dsm_arr, wall_pixel_size, cdsm_arr)
        metadata.save(walls_cache_dir)

        logger.info(f"  ✓ Walls computed and cached to {walls_cache_dir}")
    else:
        logger.warning("  ⚠ Wall generation completed but files not found")


def compute_and_cache_svf(
    surface_data: SurfaceData,
    aligned_rasters: dict,
    working_path: Path,
    trunk_ratio: float,
    on_tile_complete: Callable | None = None,
    tile_size: int | None = None,
    feedback: Any = None,
    progress_range: tuple[float, float] | None = None,
) -> None:
    """
    Compute SVF from DSM/CDSM/TDSM and cache to working_dir.

    Automatically tiles the computation for large grids to avoid GPU
    buffer size limits.

    Saves cache artifacts:
    - memmap/ for fast reload in Python API
    - svfs.zip for PrecomputedData.prepare() compatibility
    - shadowmats.npz for anisotropic sky model when export size is reasonable
      (otherwise shadow_memmaps/ is used directly)

    Args:
        surface_data: SurfaceData instance to update with computed SVF.
        aligned_rasters: Dictionary with aligned raster data.
        working_path: Working directory for caching.
        trunk_ratio: Trunk ratio for SVF computation.
        on_tile_complete: Optional callback(tile_idx, n_tiles) called after each tile
            (only invoked when tiling is used for large grids).
        feedback: Optional QGIS QgsProcessingFeedback for progress/cancellation.
        progress_range: Optional (start_pct, end_pct) for QGIS progress sub-range.
    """
    # Deferred imports to avoid a circular dependency on surface.py.
    from .precomputed import ShadowArrays, SvfArrays

    dsm_arr = aligned_rasters["dsm_arr"]
    cdsm_arr = aligned_rasters["cdsm_arr"]
    tdsm_arr = aligned_rasters["tdsm_arr"]
    pixel_size = aligned_rasters.get("pixel_size", 1.0)

    rows, cols = dsm_arr.shape
    use_veg = cdsm_arr is not None
    if use_veg:
        logger.info("Computing SVF from DSM/CDSM/TDSM...")
    else:
        logger.info("Computing SVF from DSM...")

    # Prepare vegetation arrays (Rust requires all three or none)
    if use_veg:
        cdsm_for_svf = np.asarray(cdsm_arr, dtype=np.float32)
        # Auto-generate TDSM if not provided
        if tdsm_arr is not None:
            tdsm_for_svf = np.asarray(tdsm_arr, dtype=np.float32)
        else:
            tdsm_for_svf = (cdsm_arr * trunk_ratio).astype(np.float32)
    else:
        cdsm_for_svf = np.zeros_like(dsm_arr, dtype=np.float32)
        tdsm_for_svf = np.zeros_like(dsm_arr, dtype=np.float32)

    # Height for shadow reach/buffer should be local relief, not absolute elevation.
    max_height = _max_shadow_height(dsm_arr, cdsm_arr, use_veg=use_veg)

    # Auto-detect whether tiling is needed based on real GPU/RAM limits.
    from ..tiling import compute_max_tile_pixels

    _max_pixels = compute_max_tile_pixels(context="svf")
    n_pixels = rows * cols
    needs_tiling = tile_size is not None or n_pixels > _max_pixels
    compress_exports = _should_compress_svf_exports(n_pixels)
    export_shadow_npz = _should_export_shadow_npz(n_pixels)
    if not compress_exports:
        logger.info(
            "  Large SVF export detected; using uncompressed cache files to reduce post-GPU CPU tail "
            "(set SOLWEIG_COMPRESS_MAX_PIXELS to tune)"
        )
    if not export_shadow_npz:
        logger.info(
            "  Large shadow cache detected; skipping shadowmats.npz export and keeping shadow_memmaps "
            "(set SOLWEIG_FORCE_SHADOW_NPZ=1 to force NPZ export)"
        )

    svf_cache_dir = working_path / "svf" / pixel_size_tag(pixel_size)
    svf_cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = CacheMetadata.from_arrays(dsm_arr, pixel_size, cdsm_arr)

    if needs_tiling:
        svf_data, (shmat_mm, vegshmat_mm, vbshmat_mm) = _compute_svf_tiled(
            np.asarray(dsm_arr, dtype=np.float32),
            cdsm_for_svf,
            tdsm_for_svf,
            pixel_size,
            use_veg,
            max_height,
            svf_cache_dir,
            on_tile_complete=on_tile_complete,
            tile_size=tile_size,
            feedback=feedback,
            progress_range=progress_range,
        )
        if svf_data is None:
            raise RuntimeError("SVF tiled computation returned None")
        n_patches = 153  # patch_option=2

        # Cache SVF arrays
        if feedback is not None and hasattr(feedback, "setProgressText"):
            feedback.setProgressText("Finalizing SVF cache...")
        memmap_dir = svf_cache_dir / "memmap"
        svf_data.to_memmap(memmap_dir, metadata=metadata)
        _save_svfs_zip(svf_data, svf_cache_dir, aligned_rasters, compress=compress_exports)
        metadata.save(svf_cache_dir)  # also at svf dir level for zip validation

        # Save shadow matrices as npz for compatibility when affordable.
        # For very large rasters, keep shadow_memmaps and skip expensive repacking.
        if export_shadow_npz:
            if feedback is not None and hasattr(feedback, "setProgressText"):
                feedback.setProgressText("Saving shadow matrices cache...")
            shadow_path = svf_cache_dir / "shadowmats.npz"
            save_fn = np.savez_compressed if compress_exports else np.savez
            save_fn(
                str(shadow_path),
                shadowmat=np.asarray(shmat_mm),
                vegshadowmat=np.asarray(vegshmat_mm),
                vbshmat=np.asarray(vbshmat_mm),
                patch_count=np.array(n_patches),
            )
            mode = "compressed" if compress_exports else "uncompressed"
            logger.info(f"  ✓ Shadow matrices saved as {shadow_path} ({mode})")
        else:
            shadow_path = svf_cache_dir / "shadowmats.npz"
            if shadow_path.exists():
                shadow_path.unlink()
            logger.info(f"  ✓ Shadow matrices cached as memmaps in {svf_cache_dir / 'shadow_memmaps'}")

        surface_data.svf = svf_data
        # Shadow matrices assembled from tiled memmaps (bitpacked uint8, on disk)
        surface_data.shadow_matrices = ShadowArrays(
            _shmat_u8=shmat_mm,
            _vegshmat_u8=vegshmat_mm,
            _vbshmat_u8=vbshmat_mm,
            _n_patches=n_patches,
        )
        logger.info(f"  ✓ SVF computed (tiled) and cached to {svf_cache_dir}")
    else:
        # Single-shot computation for grids that fit in GPU memory.
        # Use SkyviewRunner with threading + polling for progress and cancel.
        import threading

        from ..progress import ProgressReporter

        n_patches = 153  # patch_option=2

        runner = skyview.SkyviewRunner()
        result_box: list = [None]
        error_box: list = [None]

        def _run_svf():
            try:
                result_box[0] = runner.calculate_svf(
                    np.asarray(dsm_arr, dtype=np.float32),
                    cdsm_for_svf,
                    tdsm_for_svf,
                    pixel_size,
                    use_veg,
                    max_height,
                    2,  # patch_option
                    3.0,  # min_sun_elev_deg
                )
            except BaseException as e:
                error_box[0] = e

        thread = threading.Thread(target=_run_svf, daemon=True)
        thread.start()

        # Poll progress (153 patches)
        pbar = ProgressReporter(
            total=n_patches,
            desc="Computing Sky View Factor",
            feedback=feedback,
            progress_range=progress_range,
        )
        last = 0
        while thread.is_alive():
            thread.join(timeout=0.05)
            done = runner.progress()
            if done > last:
                pbar.update(done - last)
                last = done
            # Check QGIS cancellation
            if feedback is not None and hasattr(feedback, "isCanceled") and feedback.isCanceled():
                runner.cancel()
                thread.join(timeout=5.0)
                pbar.close()
                return
        if last < n_patches:
            pbar.update(n_patches - last)
        pbar.close()

        thread.join()
        if error_box[0] is not None:
            raise RuntimeError(f"SVF computation failed: {error_box[0]}") from error_box[0]
        svf_result = result_box[0]
        if svf_result is None:
            raise RuntimeError("SVF computation returned None (skyview.calculate_svf produced no result)")

        svf_data = SvfArrays.from_rust_result(svf_result, use_veg=use_veg)

        # Cache SVF arrays
        memmap_dir = svf_cache_dir / "memmap"
        svf_data.to_memmap(memmap_dir, metadata=metadata)
        _save_svfs_zip(svf_data, svf_cache_dir, aligned_rasters, compress=compress_exports)
        metadata.save(svf_cache_dir)  # also at svf dir level for zip validation

        # Save shadow matrices (only available in non-tiled mode)
        _save_shadow_matrices(svf_result, svf_cache_dir, compress=compress_exports)

        surface_data.svf = svf_data

        # Shadow matrices are bitpacked uint8 from Rust
        surface_data.shadow_matrices = ShadowArrays(
            _shmat_u8=np.array(svf_result.bldg_sh_matrix),
            _vegshmat_u8=np.array(svf_result.veg_sh_matrix),
            _vbshmat_u8=np.array(svf_result.veg_blocks_bldg_sh_matrix),
            _n_patches=n_patches,
        )

        logger.info(f"  ✓ SVF computed and cached to {svf_cache_dir}")


# _compute_svf_tiled extracted to surface_svf_tiled.py.
from .surface_svf_tiled import _compute_svf_tiled  # noqa: F401, E402
