"""Fetch and assemble the Madrid demo rasters from IGN (Spain) open data.

Downloads the public IGN/CNIG raster products covering the Madrid demo
extent and assembles the three rasters that ``demos/madrid-demo.py``
expects under ``temp/madrid/source/``:

- ``bdsm.tif`` — building heights above ground (2.5 m), from the IGN
  "MDSnE2,5" normalised building surface model (PNOA-LiDAR 2nd coverage).
- ``cdsm.tif`` — vegetation canopy heights above ground (2.5 m), from the
  IGN "MDSnV2,5" normalised vegetation surface model.
- ``dem.tif`` — terrain elevation (5 m), from the IGN MDT05.

No API key or registration is needed; the Centro de Descargas per-file
flow is public. Data: (c) Instituto Geografico Nacional de Espana,
CC BY 4.0 (https://pnoa.ign.es/pnoa-lidar). Total download is ~1-2 GB.

Usage::

    uv run python scripts/fetch_madrid_data.py            # full extent
    uv run python scripts/fetch_madrid_data.py --keep-tiles

Requires: requests, rasterio (already project dependencies).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Madrid demo extent (EPSG:25830), from demos/madrid-demo.py
XMIN, YMIN, XMAX, YMAX = 410679.0, 4442245.0, 465663.0, 4499872.0
CRS_EPSG = 25830

BASE = "https://centrodedescargas.cnig.es/CentroDescargas"
UA = "solweig-madrid-demo-fetch/1.0 (+https://github.com/UMEP-dev/solweig)"
PAGE_SIZE = 21  # archivosSerie pagination
POLITE_DELAY_S = 1.0

# Product series on Centro de Descargas
SERIES = {
    "bdsm": {"code": "MDSE2", "pixel": 2.5, "fill_nodata_with": 0.0},
    "cdsm": {"code": "MDSV2", "pixel": 2.5, "fill_nodata_with": 0.0},
    "dem": {"code": "MDT05", "pixel": 5.0, "fill_nodata_with": None},
}


def bbox_geojson_4326() -> str:
    """Demo extent as the lon/lat GeoJSON FeatureCollection the CNIG
    file-enumeration endpoint expects."""
    from rasterio.warp import transform as rio_transform

    xs = [XMIN, XMAX, XMAX, XMIN, XMIN]
    ys = [YMIN, YMIN, YMAX, YMAX, YMIN]
    lons, lats = rio_transform(f"EPSG:{CRS_EPSG}", "EPSG:4326", xs, ys)
    ring = [[round(lon, 6), round(lat, 6)] for lon, lat in zip(lons, lats, strict=True)]
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            ],
        }
    )


def enumerate_series(session: requests.Session, cod_serie: str) -> list[dict]:
    """List downloadable files for a series intersecting the demo extent.

    Parses the HTML fragments returned by ``archivosSerie`` (21 rows per
    page): file ids come from ``detalleArchivo?sec=<id>`` links and names
    from the adjacent text.
    """
    geo = bbox_geojson_4326()
    files: list[dict] = []
    page = 1
    total = None
    while True:
        r = session.get(
            f"{BASE}/archivosSerie",
            params={"numPagina": page, "codSerie": cod_serie, "coordenadas": geo},
            timeout=60,
        )
        r.raise_for_status()
        html = r.text
        if total is None:
            m = re.search(r'id="totalArchivos"[^>]*value="(\d+)"', html)
            total = int(m.group(1)) if m else None
        secs = re.findall(r"detalleArchivo\?sec=(\d+)", html)
        names = re.findall(r"([A-Z0-9][\w+.-]*\.(?:TIF|LAZ|ZIP|tif|laz|zip))", html)
        for sec, name in zip(secs, names, strict=False):
            files.append({"sec": sec, "name": name})
        if not secs:
            break
        page += 1
        if total is not None and len(files) >= total:
            break
        time.sleep(POLITE_DELAY_S)
    # De-duplicate, keep order
    seen = set()
    out = []
    for f in files:
        if f["sec"] not in seen:
            seen.add(f["sec"])
            out.append(f)
    print(f"  {cod_serie}: {len(out)} files intersect the extent")
    return out


def download_file(session: requests.Session, sec: str, dest: Path, retries: int = 5) -> Path:
    """Centro de Descargas per-file flow: initDescargaDir -> descargaDir."""
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE}/initDescargaDir", params={"secuencial": sec}, timeout=60)
            r.raise_for_status()
            sec_dir = r.json()["secuencialDescDir"]
            for action in ("descargaDir", "descargaDirS3"):
                resp = session.post(
                    f"{BASE}/{action}",
                    data={"secDescDirLA": sec_dir},
                    stream=True,
                    timeout=300,
                )
                if resp.status_code == 200 and "attachment" in resp.headers.get("Content-Disposition", ""):
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_content(1 << 20):
                            f.write(chunk)
                    if tmp.stat().st_size < 1024:
                        raise OSError(f"suspiciously small file for sec={sec}")
                    tmp.rename(dest)
                    return dest
                resp.close()
            raise OSError(f"no attachment response for sec={sec}")
        except (requests.RequestException, OSError, KeyError, ValueError) as e:
            wait = 2**attempt
            print(f"    retry {attempt + 1}/{retries} for {dest.name} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to download sec={sec} -> {dest}")


def mosaic_to_extent(tile_paths: list[Path], pixel: float, fill_nodata_with, out_path: Path):
    """Mosaic IGN COG tiles onto the demo extent grid and write a GeoTIFF."""
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        arr, transform = merge(
            srcs,
            bounds=(XMIN, YMIN, XMAX, YMAX),
            res=(pixel, pixel),
            nodata=srcs[0].nodata,
        )
    finally:
        for s in srcs:
            s.close()
    band = arr[0].astype(np.float32)
    nodata_out = None
    if fill_nodata_with is not None:
        # nDSM products: nodata means "no building/vegetation here" -> 0
        if srcs[0].nodata is not None:
            band[band == np.float32(srcs[0].nodata)] = fill_nodata_with
        band[~np.isfinite(band)] = fill_nodata_with
        band[band < 0] = 0.0
    else:
        nodata_out = srcs[0].nodata

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=band.shape[0],
        width=band.shape[1],
        count=1,
        dtype="float32",
        crs=f"EPSG:{CRS_EPSG}",
        transform=transform,
        compress="deflate",
        tiled=True,
        bigtiff="if_safer",
        nodata=nodata_out,
    ) as dst:
        dst.write(band, 1)
    print(f"  wrote {out_path} ({band.shape[1]}x{band.shape[0]} @ {pixel} m)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="temp/madrid/source", help="output directory for the three rasters")
    parser.add_argument("--tiles", default="temp/madrid/tiles", help="directory for downloaded source tiles")
    parser.add_argument("--keep-tiles", action="store_true", help="keep downloaded tiles (default: keep anyway)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    tiles_dir = Path(args.tiles)
    session = requests.Session()
    session.headers["User-Agent"] = UA

    for product, cfg in SERIES.items():
        out_path = out_dir / f"{product}.tif"
        if out_path.exists():
            print(f"{product}: {out_path} already present, skipping")
            continue
        code = str(cfg["code"])
        pixel = float(cfg["pixel"])  # type: ignore[arg-type]
        fill = cfg["fill_nodata_with"]
        print(f"{product}: enumerating series {code} ...")
        files = enumerate_series(session, code)
        if not files:
            print(f"  ERROR: no files found for {code} — endpoint layout may have changed", file=sys.stderr)
            return 1
        tile_paths = []
        for i, f in enumerate(files):
            dest = tiles_dir / code / f["name"]
            print(f"  [{i + 1}/{len(files)}] {f['name']}")
            tile_paths.append(download_file(session, f["sec"], dest))
            time.sleep(POLITE_DELAY_S)
        out_dir.mkdir(parents=True, exist_ok=True)
        mosaic_to_extent(tile_paths, pixel, fill, out_path)

    print("done — rasters staged for demos/madrid-demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
