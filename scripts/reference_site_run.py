"""Run the reference implementation (umep-solweig, Python) at the Gothenburg validation sites.

First half of the end-to-end check of the port against the reference with the
2026a ground scheme. It runs ``solweig_run`` from a umep-solweig checkout at
the three validation sites, with the scheme off and on, and stores the hourly
values at each measurement pixel. ``scripts/compare_with_reference.py`` then
runs the port on the same inputs and compares.

The inputs mirror ``scripts/validate_ground_scheme.py``: met clock plus one
hour (``tests/validation/met_clock.py``), site parameters with the scheme
blocks filled in from ``default_materials.json``, anisotropic sky, cylinder.
Sky view factors and shadow matrices come from the reference's own tool, and
so do the wall rasters where a site has none.

Run it with a Python that has GDAL (``osgeo``), numpy, scipy and pandas; this
package is not imported. If PROJ cannot find its data, pass ``--proj-data``.

Usage:
    <python with GDAL> scripts/reference_site_run.py <umep-solweig checkout> <work dir>
    <python with GDAL> scripts/reference_site_run.py <checkout> <work dir> --site gvc --scheme 1
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys
import warnings
from configparser import ConfigParser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VALIDATION = REPO / "tests" / "validation"
SCHEME_BLOCKS = (
    "Heat capacity",
    "Thermal_diffusivity",
    "OHM_coefficients",
    "Tg_ini coefficients",
    "Tm_ini coefficients",
)
FIELDS = ("Tmrt", "Lup", "Ldown", "Kdown", "Shadow")

#: Inputs per site. ``poi`` is (row, col), as in tests/validation/test_poi_sweep_all_sites.py.
SITES: dict[str, dict] = {
    "kronenhuset": {
        "dir": "kronenhuset",
        "dsm": "DSM_KR.tif",
        "dem": "DEM_KR.tif",
        "cdsm": "CDSM_KR.asc",
        "lc": "landcover_KR.tif",
        "wh": None,
        "wa": None,
        "params": "parametersforsolweig_KR.json",
        "poi": (51, 117),
        "met": {"20051007": "MetFile_Prepared.txt"},
    },
    "gustav_adolfs": {
        "dir": "gustav_adolfs",
        "dsm": "DSM_GA.tif",
        "dem": "DEM_GA.tif",
        "cdsm": "CDSM_GA.tif",
        "lc": "LC_GA.tif",
        "wh": "wall_height_GA.tif",
        "wa": "wall_aspect_GA.tif",
        "params": "parametersforsolweig_GA.json",
        "poi": (33, 77),
        "met": {
            "20051011": "MetFile_20051011.txt",
            "20060726": "MetFile_20060726.txt",
            "20060801": "MetFile_20060801.txt",
        },
    },
    "gvc": {
        "dir": "gvc",
        "dsm": "DSM_GVC_1m.tif",
        "dem": "DEM_GVC_1m.tif",
        "cdsm": "CDSM_GVC_1m.tif",
        "lc": "landcover_1m_GVC.tif",
        "wh": "wall_height_GVC_1m.tif",
        "wa": "wall_aspect_GVC_1m.tif",
        "params": "parametersforsolweig_GVC.json",
        "poi": (51, 122),
        "met": {
            "20100707": "MetFile20100707_Prepared.txt",
            "20100710": "MetFile20100710_Prepared.txt",
            "20100712": "MetFile20100712_Prepared.txt",
        },
    },
}


def _shift_met(src: Path, dest: Path) -> None:
    """Write the met file with one hour added to every row label."""
    lines = src.read_text().splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        year, doy, hour = int(parts[0]), int(parts[1]), int(parts[2]) + 1
        if hour == 24:
            hour, doy = 0, doy + 1
        out.append(" ".join([str(year), str(doy), str(hour), *parts[3:]]))
    dest.write_text("\n".join(out) + "\n")


def _scheme_params(site_json: Path, dest: Path) -> list[str]:
    """Site parameters with the scheme blocks filled in from the bundled defaults."""
    site = json.loads(site_json.read_text())
    defaults = json.loads((REPO / "pysrc" / "solweig" / "data" / "default_materials.json").read_text())
    added = [name for name in SCHEME_BLOCKS if name not in site]
    for name in added:
        site[name] = defaults[name]
    dest.write_text(json.dumps(site, indent=2))
    return added


def run_site(site: str, scheme: int, checkout: Path, work: Path) -> None:
    from osgeo import gdal
    from umep_solweig.skyviewfactor import processAlgorithm as sky_view_factor
    from umep_solweig.solweig_run import solweig_run

    gdal.UseExceptions()
    cfg = SITES[site]
    data = VALIDATION / cfg["dir"]
    prep = work / f"{site}_prepared"
    prep.mkdir(parents=True, exist_ok=True)

    if not (prep / "shadowmats.npz").exists():
        sky_view_factor(
            input_dsm=data / cfg["dsm"],
            input_cdsm=data / cfg["cdsm"],
            use_gpu=False,
            aniso=True,
            outputDir=str(prep),
            outputFile=str(prep / "svf.tif"),
        )
    if cfg["wh"] is None:
        from umep_solweig.wallalgorithm import processAlgorithm as wall_tool

        wall_height, wall_aspect = prep / "wall_height.tif", prep / "wall_aspect.tif"
        if not wall_aspect.exists():
            wall_tool(
                str(data / cfg["dsm"]),
                wall_limit=3.0,
                use_gpu=False,
                output_height_path=str(wall_height),
                calculate_aspect=True,
                output_aspect_path=str(wall_aspect),
            )
    else:
        wall_height, wall_aspect = data / cfg["wh"], data / cfg["wa"]

    params = prep / "parameters.json"
    added = _scheme_params(data / cfg["params"], params)
    dsm = gdal.Open(str(data / cfg["dsm"]))
    pixel = abs(dsm.GetGeoTransform()[1])
    print(f"{site}: {dsm.RasterYSize} x {dsm.RasterXSize} px of {pixel} m; blocks from defaults: {added}")

    template = ConfigParser()
    template.read(checkout / "tests" / "tests_data" / "configsolweig.ini")
    row, col = cfg["poi"]
    for day, met_name in sorted(cfg["met"].items()):
        out = work / f"{site}_{day}_s{scheme}"
        out.mkdir(parents=True, exist_ok=True)
        met = out / "met_shifted.txt"
        _shift_met(data / met_name, met)
        config = ConfigParser()
        config.read_dict({"DEFAULT": dict(template.defaults())})
        config["DEFAULT"].update(
            output_dir=str(out),
            working_dir=str(out),
            para_json_path=str(params),
            filepath_dsm=str(data / cfg["dsm"]),
            filepath_cdsm=str(data / cfg["cdsm"]),
            filepath_dem=str(data / cfg["dem"]),
            filepath_lc=str(data / cfg["lc"]),
            filepath_wh=str(wall_height),
            filepath_wa=str(wall_aspect),
            input_svf=str(prep / "svfs.zip"),
            input_aniso=str(prep / "shadowmats.npz"),
            input_met=str(met),
            lat="57.7",
            lon="12.0",
            utc="1",
            scale=str(1.0 / pixel),
            cyl="1",
            aniso="1",
            landcover="1",
            demforbuild="0",
            groundmodel=str(scheme),
            outgoinglongwave=str(scheme),
            calculation_mode="cpu",
            outputtreeplanter="0",
            outputkdiff="0",
        )
        config_path = out / "config.ini"
        with open(config_path, "w") as fh:
            config.write(fh)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            solweig_run(str(config_path), None)

        series: dict[str, dict[int, float]] = {}
        for field in FIELDS:
            values = {}
            for path in sorted(glob.glob(str(out / f"{field}_*.tif"))):
                code = Path(path).stem.split("_")[-1]  # HHMM followed by D or N
                if not code[:4].isdigit():
                    continue
                raster = gdal.Open(path)
                values[int(code[:2])] = float(raster.ReadAsArray()[row, col])
            series[field] = values
        (out / "poi.json").write_text(json.dumps(series, indent=1))
        tmrt = series["Tmrt"]
        print(
            f"  {day} scheme={scheme}: {len(tmrt)} hours, POI Tmrt {min(tmrt.values()):.1f} to {max(tmrt.values()):.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reference implementation (umep-solweig, Python) at the Gothenburg validation sites."
    )
    parser.add_argument("checkout", type=Path, help="umep-solweig checkout (its src/ is put on sys.path)")
    parser.add_argument("work", type=Path, help="folder for the reference outputs")
    parser.add_argument("--site", choices=sorted(SITES), action="append", help="default: all three")
    parser.add_argument("--scheme", type=int, choices=(0, 1), action="append", help="default: both")
    parser.add_argument("--proj-data", type=Path, help="value for PROJ_DATA, if PROJ cannot find its data")
    args = parser.parse_args()

    if args.proj_data:
        os.environ["PROJ_DATA"] = str(args.proj_data)
    sys.path.insert(0, str(args.checkout / "src"))
    warnings.simplefilter("ignore")
    for site in args.site or sorted(SITES):
        for scheme in args.scheme or (0, 1):
            run_site(site, scheme, args.checkout, args.work)


if __name__ == "__main__":
    main()
