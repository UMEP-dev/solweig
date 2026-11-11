"""Run a fixed set of raster and model operations on the current backend and print their fingerprints.

Executed as a script, once per backend, by ``tests/test_backend_parity.py``.
The last line of stdout is a JSON object mapping a label to a SHA-256 of the
resulting array (or to a small rounded value). The two backends must print
the same object.

Usage: python tests/backend_parity_probe.py <scratch dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tests.backend_fence import gdal_backend_requested, install_backend_fence  # noqa: E402

install_backend_fence()

import numpy as np  # noqa: E402
import solweig  # noqa: E402
from solweig import io  # noqa: E402

from tests.raster_helpers import read_band, tiff_structure, write_geotiff  # noqa: E402


def digest(array) -> str:
    array = np.ascontiguousarray(array)
    return f"{array.dtype}{array.shape}:{hashlib.sha256(array.tobytes()).hexdigest()[:16]}"


def scene(rows: int = 96, cols: int = 120) -> np.ndarray:
    terrain = np.linspace(0.0, 2.0, rows)[:, None] + np.linspace(0.0, 1.0, cols)[None, :]
    dsm = terrain.astype(np.float32)
    dsm[20:40, 30:55] += 14.0
    dsm[60:80, 70:100] += 9.0
    return dsm


def main(scratch: Path) -> dict:
    out: dict[str, object] = {}
    assert bool(io.GDAL_ENV) == gdal_backend_requested(), "backend does not match UMEP_USE_GDAL"
    dsm = scene()
    rows, cols = dsm.shape
    pixel = 1.0
    origin_x, top = 319000.0, 6400000.0 + rows * pixel
    gt = [origin_x, pixel, 0.0, top, 0.0, -pixel]

    # ── raster I/O ─────────────────────────────────────────────────────────
    dsm_path = scratch / "dsm.tif"
    holed = dsm.copy()
    holed[0:3, 0:4] = -9999.0
    write_geotiff(dsm_path, holed, origin_x=origin_x, origin_y_top=top, pixel_size=pixel, epsg=3006)
    loaded, transform, crs_wkt, nodata = io.load_raster(str(dsm_path))
    crs_wkt = crs_wkt or ""
    out["load_raster values (nodata -> NaN)"] = digest(loaded)
    out["load_raster transform"] = [round(float(v), 6) for v in transform]
    out["load_raster nodata"] = None if nodata is None else round(float(nodata), 3)
    out["load_raster has crs"] = bool(crs_wkt)
    out["get_raster_metadata shape"] = [io.get_raster_metadata(dsm_path)[k] for k in ("rows", "cols")]
    window = (slice(10, 50), slice(20, 90))
    out["read_raster_window"] = digest(io.read_raster_window(dsm_path, window))

    for use_cog in (True, False):
        path = scratch / f"saved_cog_{use_cog}.tif"
        io.save_raster(str(path), dsm, gt, crs_wkt, use_cog=use_cog, generate_preview=False)
        out[f"save_raster cog={use_cog} values"] = digest(read_band(path))
        back, transform2, _, _ = io.load_raster(str(path))
        out[f"save_raster cog={use_cog} reload"] = digest(back)
        out[f"save_raster cog={use_cog} transform"] = [round(float(v), 6) for v in transform2]

    nan_path = scratch / "nan_nodata.tif"
    with_nan = dsm.copy()
    with_nan[5:9, 5:9] = np.nan
    io.save_raster(str(nan_path), with_nan, gt, crs_wkt, no_data_val=np.nan, use_cog=False, generate_preview=False)
    out["save_raster NaN nodata values"] = digest(read_band(nan_path))

    for compress in (True, False):
        path = scratch / f"windowed_{compress}.tif"
        io.create_empty_raster(path, rows, cols, gt, crs_wkt, nodata=np.nan, compress=compress)
        out[f"create_empty compress={compress} unwritten"] = digest(read_band(path))
        for win in ((slice(0, 50), slice(0, cols)), (slice(50, rows), slice(0, 70))):
            io.write_raster_window(path, dsm[win], window=win)
        out[f"windowed compress={compress} values"] = digest(read_band(path))
        out[f"windowed compress={compress} layout"] = tiff_structure(path)["compress"]

    # ── surface preparation from files (no resampling) ─────────────────────
    clean_path = scratch / "dsm_clean.tif"
    write_geotiff(clean_path, dsm, origin_x=origin_x, origin_y_top=top, pixel_size=pixel, epsg=3006)
    surface = solweig.SurfaceData.prepare(dsm=str(clean_path), working_dir=str(scratch / "work"))
    assert surface.svf is not None, "prepare() returned a surface without SVF"
    out["prepare dsm"] = digest(surface.dsm)
    out["prepare wall_height"] = digest(surface.wall_height)
    out["prepare wall_aspect"] = digest(surface.wall_aspect)
    out["prepare svf"] = digest(surface.svf.svf)
    out["prepare pixel_size"] = round(float(surface.pixel_size), 6)

    # A second prepare on the same directory must load what the first one saved.
    again = solweig.SurfaceData.prepare(dsm=str(clean_path), working_dir=str(scratch / "work"))
    assert again.svf is not None, "prepare() reloaded a surface without SVF"
    out["prepare reload wall_height"] = digest(again.wall_height)
    out["prepare reload svf"] = digest(again.svf.svf)

    # ── a model run, untiled and tiled ─────────────────────────────────────
    weather = [
        solweig.Weather(datetime=datetime(2024, 7, 15, hour, 0), ta=24.0 + hour / 10, rh=50.0, global_rad=650.0)
        for hour in (10, 13)
    ]
    location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=1)
    runs: tuple[tuple[str, dict[str, Any]], ...] = (
        ("untiled", {}),
        ("tiled", {"tile_size": 256, "max_shadow_distance_m": 30.0}),
    )
    for tag, kwargs in runs:
        big = np.tile(dsm, (4, 4)) if tag == "tiled" else dsm
        surf = surface if tag == "untiled" else solweig.SurfaceData(dsm=big, pixel_size=1.0)
        if tag == "tiled":
            surf.compute_svf()
        run_dir = scratch / f"run_{tag}"
        solweig.calculate(surf, weather, location, output_dir=run_dir, outputs=["tmrt", "shadow", "utci"], **kwargs)
        for name in ("tmrt", "shadow", "utci"):
            tifs = sorted((run_dir / name).glob("*.tif"))
            out[f"{tag} {name} files"] = [t.name for t in tifs]
            for tif in tifs:
                out[f"{tag} {tif.name}"] = digest(read_band(tif))
            out[f"{tag} {name} layout"] = tiff_structure(tifs[0])["compress"]
        summary = sorted(p.name for p in (run_dir / "summary").glob("*.tif"))
        out[f"{tag} summary files"] = summary
        for name in summary:
            out[f"{tag} summary {name}"] = digest(io.load_raster(str(run_dir / "summary" / name))[0])
        out[f"{tag} leftover partial files"] = len(list(run_dir.rglob("*.partial.tif")))
    return out


if __name__ == "__main__":
    scratch = Path(sys.argv[1])
    scratch.mkdir(parents=True, exist_ok=True)
    result = main(scratch)
    print(json.dumps(result, sort_keys=True))
