"""Internal tiled SVF computation helper for very large rasters.

Extracted from `models/surface_compute.py` because the function is
~340 lines and is invoked from a single call site (the
``compute_and_cache_svf`` orchestration). Keeping it here lets the
parent module stay focused on the wall + SVF caching wrapper logic.

The function takes pre-aligned DSM/CDSM/TDSM arrays plus tiling config
and returns a populated :class:`SvfArrays` + the three bitpacked shadow
matrices as memmaps. Used only when the grid exceeds the per-tile
pixel budget reported by `tiling.compute_max_tile_pixels(context="svf")`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import ComputationCancelled
from ..rustalgos import skyview
from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from .precomputed import SvfArrays

logger = get_logger(__name__)


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
    from .precomputed import SvfArrays  # deferred to avoid circular import via surface.py

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
    outputs: dict[str, np.memmap] = {}
    svf_memmap_dir = working_path / "svf_memmaps"
    svf_memmap_dir.mkdir(parents=True, exist_ok=True)
    for name in all_fields:
        mm = np.memmap(
            svf_memmap_dir / f"{name}.dat",
            dtype=np.float32,
            mode="w+",
            shape=(rows, cols),
        )
        # No blanket ``mm[:] = 1.0`` pre-fill: the tile write_slices partition
        # the raster exactly (generate_tiles cores cover every pixel once), so
        # every output pixel is overwritten by its tile in _process_result.
        # A whole-array pre-fill would dirty all ~30 GB of these mappings up
        # front (the cold-prepare RSS spike); w+ zero-init plus full tile
        # coverage gives byte-identical output without it.
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
        from .surface_compute import _max_shadow_height  # local import to avoid circular dep

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

        # Flush this tile's freshly written pages so the OS can write them back
        # and reclaim them: resident memory then tracks a few tiles instead of
        # accumulating the whole-raster mapping (writes are disjoint per tile).
        for name in svf_fields:
            outputs[name].flush()
        if use_veg:
            for name in veg_fields:
                outputs[name].flush()
        shmat_mm.flush()
        if use_veg:
            vegshmat_mm.flush()
            vbshmat_mm.flush()

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
                # Raising (instead of returning partial results) is load-bearing:
                # the except-branch below removes the partial memmap dirs, and the
                # caller never persists a half-computed SVF as a valid cache.
                logger.info("  SVF computation cancelled by user")
                raise ComputationCancelled(f"SVF tile {tile_idx + 1}/{n_tiles}")

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
        arr.flush()
    if hasattr(ones, "flush"):
        ones.flush()  # type: ignore[union-attr]

    return svf_data, (shmat_mm, vegshmat_mm, vbshmat_mm)
