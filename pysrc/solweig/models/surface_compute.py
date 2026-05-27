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

import json
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
    from .precomputed import SvfArrays
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


def _compute_svf_tiled(
    dsm_f32: np.ndarray,
    cdsm_f32: np.ndarray,
    tdsm_f32: np.ndarray,
    pixel_size: float,
    use_veg: bool,
    max_height: float,
    working_path: Path,
    on_tile_complete: Callable | None = None,
    tile_size: int | None = None,
    feedback: Any = None,
    progress_range: tuple[float, float] | None = None,
) -> tuple[SvfArrays, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute SVF using tiled processing for large grids.

    Automatically determines the largest safe tile size from the GPU
    buffer limit, divides the grid into overlapping tiles, computes
    SVF per tile, and stitches the core regions into full-size arrays.

    Shadow matrices are assembled into memory-mapped bitpacked uint8 files to
    avoid holding the full 3D arrays in RAM.

    Args:
        dsm_f32: DSM array (float32).
        cdsm_f32: Canopy DSM array (float32, zeros if no veg).
        tdsm_f32: Trunk DSM array (float32, zeros if no veg).
        pixel_size: Pixel size in meters.
        use_veg: Whether vegetation is present.
        max_height: Maximum height in the DSM (for buffer calculation).
        working_path: Directory for memmap files.
        on_tile_complete: Optional callback(tile_idx, n_tiles) called after each tile.

    Returns:
        Tuple of (SvfArrays, (shmat_mm, vegshmat_mm, vbshmat_mm))
        where the shadow matrix memmaps are bitpacked uint8 (rows, cols, n_pack).
    """
    from .. import _ensure_gpu_initialized
    from ..progress import ProgressReporter
    from ..tiling import calculate_buffer_distance, generate_tiles, validate_tile_size

    _ensure_gpu_initialized()
    rows, cols = dsm_f32.shape

    buffer_m = calculate_buffer_distance(max_height)
    buffer_pixels = int(np.ceil(buffer_m / pixel_size))

    # Compute the largest safe tile size from real GPU/RAM limits.
    # The full tile (core + 2*buffer) must fit, so subtract buffer from max side.
    from ..tiling import MIN_TILE_SIZE, compute_max_tile_side

    if tile_size is not None:
        core_tile_size = tile_size
    else:
        max_full_side = compute_max_tile_side(context="svf")
        core_tile_size = max(MIN_TILE_SIZE, max_full_side - 2 * buffer_pixels)

    adjusted_tile_size, warning = validate_tile_size(core_tile_size, buffer_pixels, pixel_size, context="svf")
    if warning:
        logger.warning(warning)

    tiles = generate_tiles(rows, cols, adjusted_tile_size, buffer_pixels)
    n_tiles = len(tiles)

    # Determine patch count from a small probe (patch_option=2 → 153 patches)
    n_patches = 153

    logger.info(
        f"  Tiled SVF: {rows}x{cols} raster, {n_tiles} tiles, "
        f"tile_size={adjusted_tile_size}, buffer={buffer_m:.0f}m ({buffer_pixels}px)"
    )

    # SVF field names on the Rust result object
    svf_fields = ["svf", "svf_north", "svf_east", "svf_south", "svf_west"]
    veg_fields = [
        "svf_veg",
        "svf_veg_north",
        "svf_veg_east",
        "svf_veg_south",
        "svf_veg_west",
        "svf_veg_blocks_bldg_sh",
        "svf_veg_blocks_bldg_sh_north",
        "svf_veg_blocks_bldg_sh_east",
        "svf_veg_blocks_bldg_sh_south",
        "svf_veg_blocks_bldg_sh_west",
    ]
    all_fields = svf_fields + veg_fields if use_veg else svf_fields

    # Pre-allocate output arrays as memmaps on disk to avoid massive RAM
    # use for very large rasters (e.g. >100M pixels).
    outputs: dict[str, np.ndarray] = {}
    svf_memmap_dir = working_path / "svf_memmaps"
    svf_memmap_dir.mkdir(parents=True, exist_ok=True)
    for name in all_fields:
        mm = np.memmap(
            svf_memmap_dir / f"{name}.dat",
            dtype=np.float32,
            mode="w+",
            shape=(rows, cols),
        )
        mm[:] = 1.0  # default for untouched pixels / masked edges
        outputs[name] = mm

    # Pre-allocate memmap files for shadow matrices (bitpacked uint8, on disk)
    memmap_dir = working_path / "shadow_memmaps"
    memmap_dir.mkdir(parents=True, exist_ok=True)
    n_pack = (n_patches + 7) // 8  # ceil(153/8) = 20
    sh_shape = (rows, cols, n_pack)
    shadow_meta = {
        "shape": [rows, cols, n_pack],
        "patch_count": n_patches,
        "shadowmat_file": "shmat.dat",
        "vegshadowmat_file": "vegshmat.dat",
        "vbshmat_file": "vbshmat.dat",
    }
    with (memmap_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(shadow_meta, f, indent=2)
    shmat_mm = np.memmap(
        memmap_dir / "shmat.dat",
        dtype=np.uint8,
        mode="w+",
        shape=sh_shape,
    )
    vegshmat_mm = np.memmap(
        memmap_dir / "vegshmat.dat",
        dtype=np.uint8,
        mode="w+",
        shape=sh_shape,
    )
    vbshmat_mm = np.memmap(
        memmap_dir / "vbshmat.dat",
        dtype=np.uint8,
        mode="w+",
        shape=sh_shape,
    )
    if not use_veg:
        vegshmat_mm[:] = 0
        vbshmat_mm[:] = 0

    # Progress: one bar per tile (matching timeseries progress style).
    # For QGIS feedback, a single reporter spans the full progress_range.
    _use_per_tile_bars = feedback is None and n_tiles > 1

    # Pipeline: overlap GPU computation of tile N+1 with CPU
    # result-copying of tile N.  SkyviewRunner.calculate_svf releases the
    # GIL inside py.allow_threads(), so a background thread can drive the
    # GPU while the main thread polls progress and does numpy bookkeeping.
    import threading

    def _submit_tile(tile):
        """Prepare inputs and run SVF on background thread with progress."""
        rs = tile.read_slice
        cs = tile.core_slice
        core_row_start = int(cs[0].start or 0)
        core_row_end = int(cs[0].stop or 0)
        core_col_start = int(cs[1].start or 0)
        core_col_end = int(cs[1].stop or 0)
        td = dsm_f32[rs].copy()
        tc = cdsm_f32[rs].copy()
        tt = tdsm_f32[rs].copy()
        mh = _max_shadow_height(td, tc, use_veg=use_veg)
        runner = skyview.SkyviewRunner()
        box = [None, None]  # [result, error]
        core_only = hasattr(runner, "calculate_svf_core")

        def _run():
            try:
                if core_only:
                    box[0] = runner.calculate_svf_core(
                        td,
                        tc,
                        tt,
                        pixel_size,
                        use_veg,
                        mh,
                        2,  # patch_option
                        3.0,  # min_sun_elev_deg
                        core_row_start,
                        core_row_end,
                        core_col_start,
                        core_col_end,
                    )
                else:
                    box[0] = runner.calculate_svf(
                        td,
                        tc,
                        tt,
                        pixel_size,
                        use_veg,
                        mh,
                        2,  # patch_option
                        3.0,  # min_sun_elev_deg
                    )
            except BaseException as e:
                box[1] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t, box, runner, core_only

    def _process_result(tile_result, tile, core_only):
        """Copy SVF + shadow matrices from a completed tile."""
        cs = tile.core_slice
        ws = tile.write_slice

        # Avoid redundant array copies: Rust returns numpy-backed arrays.
        svf_arrays = {name: np.asarray(getattr(tile_result, name)) for name in svf_fields}
        for name in svf_fields:
            outputs[name][ws] = svf_arrays[name] if core_only else svf_arrays[name][cs]
        if use_veg:
            veg_arrays = {name: np.asarray(getattr(tile_result, name)) for name in veg_fields}
            for name in veg_fields:
                outputs[name][ws] = veg_arrays[name] if core_only else veg_arrays[name][cs]
        # Shadow matrices are already bitpacked uint8 from Rust
        bldg = np.asarray(tile_result.bldg_sh_matrix)
        shmat_mm[ws] = bldg if core_only else bldg[cs]
        if use_veg:
            veg = np.asarray(tile_result.veg_sh_matrix)
            vb = np.asarray(tile_result.veg_blocks_bldg_sh_matrix)
            vegshmat_mm[ws] = veg if core_only else veg[cs]
            vbshmat_mm[ws] = vb if core_only else vb[cs]

    # Kick off first tile
    thread, box, runner, core_only = _submit_tile(tiles[0])

    pbar = None
    try:
        for tile_idx in range(n_tiles):
            # Per-tile progress bar (or single QGIS bar)
            if _use_per_tile_bars:
                if pbar is not None:
                    pbar.close()
                tile_desc = f"SVF tile {tile_idx + 1}/{n_tiles}"
                pbar = ProgressReporter(total=n_patches, desc=tile_desc)
            elif pbar is None:
                pbar = ProgressReporter(
                    total=n_tiles * n_patches,
                    desc="Computing SVF (tiled)",
                    feedback=feedback,
                    progress_range=progress_range,
                )
                pbar.set_description(f"SVF tile {tile_idx + 1}/{n_tiles}")
                pbar.set_text(f"Computing SVF — Tile {tile_idx + 1}/{n_tiles}")
            else:
                pbar.set_description(f"SVF tile {tile_idx + 1}/{n_tiles}")
                pbar.set_text(f"Computing SVF — Tile {tile_idx + 1}/{n_tiles}")

            # Poll per-patch progress while tile runs
            last_patch = 0
            cancelled = False
            while thread.is_alive():
                thread.join(timeout=0.05)
                done = runner.progress()
                if done > last_patch:
                    pbar.update(done - last_patch)
                    last_patch = done
                # Check QGIS cancellation within tile
                if pbar.is_cancelled():
                    runner.cancel()
                    thread.join(timeout=5.0)
                    cancelled = True
                    break
            if cancelled:
                logger.info("  SVF computation cancelled by user")
                break

            # Ensure progress accounts for all patches in this tile
            if last_patch < n_patches:
                pbar.update(n_patches - last_patch)

            # Check for errors
            if box[1] is not None:
                tile = tiles[tile_idx]
                raise RuntimeError(
                    f"SVF tile {tile_idx + 1}/{n_tiles} failed (read_slice={tile.read_slice}): {box[1]}"
                ) from box[1]
            cur_result = box[0]
            if cur_result is None:
                raise RuntimeError(
                    f"SVF tile {tile_idx + 1}/{n_tiles} returned None (skyview.calculate_svf produced no result)"
                )
            cur_core_only = core_only

            # Submit next tile (GPU starts while we copy results below)
            if tile_idx + 1 < n_tiles:
                thread, box, runner, core_only = _submit_tile(tiles[tile_idx + 1])

            # Copy results on main thread (overlaps with next GPU computation)
            _process_result(cur_result, tiles[tile_idx], cur_core_only)
            if on_tile_complete is not None:
                on_tile_complete(tile_idx, n_tiles)
    except BaseException:
        # Clean up partial memmap files so stale data doesn't persist
        import shutil

        for d in (svf_memmap_dir, memmap_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        raise
    finally:
        if pbar is not None:
            pbar.close()
    # Flush memmaps to disk
    shmat_mm.flush()
    vegshmat_mm.flush()
    vbshmat_mm.flush()

    # Shared ones memmap for non-vegetation cases (avoids 5x full-size copies).
    # When use_veg is True the veg outputs come from the real computation and
    # ``ones`` is unused, but we still need a valid ndarray for the type checker.
    if use_veg:
        ones = np.ones((1, 1), dtype=np.float32)
    else:
        ones = np.memmap(
            svf_memmap_dir / "ones.dat",
            dtype=np.float32,
            mode="w+",
            shape=(rows, cols),
        )
        ones[:] = 1.0

    svf_data = SvfArrays(
        svf=outputs["svf"],
        svf_north=outputs["svf_north"],
        svf_east=outputs["svf_east"],
        svf_south=outputs["svf_south"],
        svf_west=outputs["svf_west"],
        svf_veg=outputs["svf_veg"] if use_veg else ones,
        svf_veg_north=outputs["svf_veg_north"] if use_veg else ones,
        svf_veg_east=outputs["svf_veg_east"] if use_veg else ones,
        svf_veg_south=outputs["svf_veg_south"] if use_veg else ones,
        svf_veg_west=outputs["svf_veg_west"] if use_veg else ones,
        svf_aveg=outputs["svf_veg_blocks_bldg_sh"] if use_veg else ones,
        svf_aveg_north=outputs["svf_veg_blocks_bldg_sh_north"] if use_veg else ones,
        svf_aveg_east=outputs["svf_veg_blocks_bldg_sh_east"] if use_veg else ones,
        svf_aveg_south=outputs["svf_veg_blocks_bldg_sh_south"] if use_veg else ones,
        svf_aveg_west=outputs["svf_veg_blocks_bldg_sh_west"] if use_veg else ones,
    )

    # Flush all SVF memmaps to disk
    for arr in outputs.values():
        if hasattr(arr, "flush"):
            arr.flush()  # type: ignore[union-attr]
    if hasattr(ones, "flush"):
        ones.flush()  # type: ignore[union-attr]

    return svf_data, (shmat_mm, vegshmat_mm, vbshmat_mm)
