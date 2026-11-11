"""Pre-release check inside a real QGIS: the plugin and the GDAL backend, no mocks.

The test suite replaces QGIS with mocks and, outside one CI job, never runs on
the GDAL backend. This script closes that gap by hand, once per release, on a
machine with QGIS installed. It runs inside the QGIS application itself (its
Python, GDAL, NumPy and Qt), and does three things:

1. runs ``tests/backend_parity_probe.py`` on the GDAL backend, with rasterio
   and shapely fenced off, and writes the fingerprints;
2. loads the plugin exactly as QGIS does and runs its Processing algorithms
   (surface preparation, a single-timestep calculation, and a sub-hourly UMEP
   met series) on a small synthetic site;
3. writes ``temp/qgis_release_check/result.json`` and quits QGIS.

Run it through the QGIS application binary, not ``qgis_process`` or the bundled
``python``: on macOS only the application is allowed to load a compiled
extension that QGIS did not sign.

    macOS:    /Applications/QGIS.app/Contents/MacOS/QGIS --noplugins --code scripts/qgis_release_check.py
    Windows:  qgis-bin.exe --noplugins --code scripts\\qgis_release_check.py
    Linux:    qgis --noplugins --code scripts/qgis_release_check.py

Then compare with the rasterio backend:

    python scripts/qgis_release_check.py --compare

which runs the same probe in the project environment and reports any
fingerprint that differs. ``--noplugins`` keeps an installed copy of the
plugin from registering the same provider.

The package under test is the source tree (``pysrc/``, with the extension
built by ``maturin develop --release``), so the check runs before publishing.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _find_repo() -> Path:
    """The checkout this script belongs to. QGIS runs ``--code`` files without ``__file__``."""
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        pass
    try:
        from qgis.PyQt.QtCore import QCoreApplication

        args = list(QCoreApplication.arguments())
    except Exception:  # noqa: BLE001
        args = list(sys.argv)
    for flag in ("--code", "-f"):
        if flag in args[:-1]:
            return Path(args[args.index(flag) + 1]).resolve().parent.parent
    return Path.cwd()


REPO = _find_repo()
OUT = REPO / "temp" / "qgis_release_check"


def _fingerprints_differ(a: dict, b: dict) -> list[str]:
    lines = [f"only in QGIS: {k}" for k in sorted(set(a) - set(b))]
    lines += [f"only in rasterio: {k}" for k in sorted(set(b) - set(a))]
    lines += [f"{k}: QGIS {a[k]} != rasterio {b[k]}" for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    return lines


def compare() -> int:
    """Run the probe on the rasterio backend here and compare with the QGIS run."""
    import shutil
    import subprocess

    result_path = OUT / "result.json"
    if not result_path.exists():
        print(f"No QGIS result at {result_path}. Run the script inside QGIS first.")
        return 2
    result = json.loads(result_path.read_text())
    print(f"QGIS run: QGIS {result.get('qgis')}, GDAL {result.get('gdal')}, NumPy {result.get('numpy')}, "
          f"Python {result.get('python')}, solweig {result.get('solweig')}")  # fmt: skip
    failures = list(result.get("failures", []))
    try:
        import tomllib

        current = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    except Exception:  # noqa: BLE001
        current = None
    if current and result.get("solweig") != current:
        failures.append(
            f"the QGIS run tested solweig {result.get('solweig')} but the checkout is {current}: "
            "rebuild (maturin develop --release) and run the script inside QGIS again"
        )
    for step in result.get("steps", []):
        print(f"  {'ok  ' if step['ok'] else 'FAIL'} {step['name']} {step.get('detail', '')}")

    scratch = OUT / "probe_rasterio"
    if scratch.exists():
        shutil.rmtree(scratch)
    env = {k: v for k, v in os.environ.items() if k != "UMEP_USE_GDAL"}
    env["SOLWEIG_NO_GPU"] = "1"
    done = subprocess.run(
        [sys.executable, str(REPO / "tests" / "backend_parity_probe.py"), str(scratch)],
        capture_output=True, text=True, env=env,
    )  # fmt: skip
    if done.returncode != 0:
        print(done.stderr[-3000:])
        print("FAIL the probe did not run on the rasterio backend")
        return 1
    rasterio_prints = json.loads(done.stdout.strip().splitlines()[-1])
    qgis_prints = result.get("fingerprints") or {}
    differences = _fingerprints_differ(qgis_prints, rasterio_prints) if qgis_prints else ["no fingerprints from QGIS"]
    print(f"  {'ok  ' if not differences else 'FAIL'} backend parity: {len(qgis_prints)} fingerprints compared")
    for line in differences:
        print(f"       {line}")
    if failures or differences:
        for failure in failures:
            print(f"FAILURE: {failure}")
        return 1
    print("ALL OK")
    return 0


def _write_dsm(path: Path) -> None:
    import numpy as np
    from osgeo import gdal, osr

    rows, cols = 96, 120
    terrain = np.linspace(0.0, 2.0, rows)[:, None] + np.linspace(0.0, 1.0, cols)[None, :]
    dsm = terrain.astype(np.float32)
    dsm[20:40, 30:55] += 14.0
    dsm[60:80, 70:100] += 9.0
    ds = gdal.GetDriverByName("GTiff").Create(str(path), cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([319000.0, 1.0, 0.0, 6400096.0, 0.0, -1.0])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3006)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(dsm)
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds = None


def _write_umep_met(path: Path) -> None:
    header = "%iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd\n"
    rows = [
        f"2024 197 {hour} {minute} -999 -999 -999 -999 -999 2 50 24 101.3 0 600 0 -999 0 0 0 0 150 600 180\n"
        for hour in (11, 12)
        for minute in (0, 10, 20, 30, 40, 50)
    ]
    path.write_text(header + "".join(rows))


def _finite_fraction(tif: Path) -> float:
    import numpy as np
    from osgeo import gdal

    arr = gdal.Open(str(tif)).ReadAsArray()
    return float(np.isfinite(arr).mean())


def run_inside_qgis() -> dict:
    import shutil

    result: dict = {"steps": [], "failures": [], "fingerprints": None}

    def step(name: str, fn) -> object:
        try:
            detail = fn()
            result["steps"].append({"name": name, "ok": True, "detail": str(detail or "")})
            return detail
        except Exception as exc:  # noqa: BLE001
            result["steps"].append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
            result["failures"].append(f"{name}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return None

    os.environ["UMEP_USE_GDAL"] = "1"
    os.environ["SOLWEIG_NO_GPU"] = "1"  # the comparison run uses the CPU path too
    for entry in (REPO / "qgis_plugin", REPO / "pysrc", REPO):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))

    import numpy as np
    from osgeo import gdal
    from qgis.core import Qgis, QgsApplication

    result.update(qgis=Qgis.QGIS_VERSION, gdal=gdal.__version__, numpy=np.__version__, python=sys.version.split()[0])

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    from tests.backend_fence import install_backend_fence

    def fence():
        installed = install_backend_fence(host_packages_ok=True)
        if not installed:
            raise RuntimeError("fence not installed")
        loaded = sorted({n.split(".")[0] for n in sys.modules} & {"rasterio", "shapely", "geopandas", "fiona"})
        if loaded:
            raise RuntimeError(f"already loaded by the host: {loaded}")
        return "rasterio, shapely, geopandas, fiona blocked"

    step("import fence", fence)

    def package():
        import solweig

        result["solweig"] = solweig.__version__
        result["solweig_file"] = solweig.__file__
        if not solweig.io.GDAL_ENV:
            raise RuntimeError("solweig is not on the GDAL backend")
        if REPO not in Path(solweig.__file__).resolve().parents:
            raise RuntimeError(f"solweig was imported from {solweig.__file__}, not from this checkout")
        return f"{solweig.__version__} from the checkout, GDAL backend"

    step("solweig imports on the GDAL backend", package)

    def probe():
        from tests import backend_parity_probe

        result["fingerprints"] = backend_parity_probe.main(OUT / "probe_gdal")
        return f"{len(result['fingerprints'])} fingerprints"

    step("backend parity probe", probe)

    # ── the plugin, loaded as QGIS loads it ────────────────────────────────
    plugins_dir = Path(QgsApplication.pkgDataPath()) / "python" / "plugins"
    if str(plugins_dir) not in sys.path:
        sys.path.append(str(plugins_dir))

    def load_plugin():
        import processing  # noqa: F401
        from processing.core.Processing import Processing

        Processing.initialize()
        import solweig_qgis

        try:
            from qgis.utils import iface
        except ImportError:
            iface = None
        plugin = solweig_qgis.classFactory(iface)
        plugin.initProcessing()
        result["_plugin"] = plugin
        provider = QgsApplication.processingRegistry().providerById("solweig")
        names = sorted(alg.name() for alg in provider.algorithms())
        expected = {"surface_preprocessing", "solweig_calculation", "svf_preprocessing", "epw_import"}
        if not expected <= set(names):
            raise RuntimeError(f"algorithms registered: {names}")
        if not solweig_qgis._SOLWEIG_AVAILABLE or solweig_qgis._SOLWEIG_OUTDATED:
            raise RuntimeError(f"plugin dependency check failed: {solweig_qgis._SOLWEIG_IMPORT_ERROR}")
        return f"{len(names)} algorithms, solweig source: {solweig_qgis._SOLWEIG_SOURCE}"

    step("plugin loads and registers its provider", load_plugin)

    dsm_path = OUT / "dsm.tif"
    surface_dir = OUT / "surface"
    _write_dsm(dsm_path)

    def prepare():
        import processing

        processing.run("solweig:surface_preprocessing", {"DSM": str(dsm_path), "OUTPUT_DIR": str(surface_dir)})
        walls = list(surface_dir.glob("walls/px*/wall_hts.tif"))
        svf = list(surface_dir.glob("svf/px*"))
        if not walls or not svf or not (surface_dir / "parametersforsolweig.json").exists():
            raise RuntimeError(f"missing outputs in {surface_dir}")
        return "walls, SVF and parameters written"

    step("algorithm: prepare surface data", prepare)

    def prepare_again_with_wall_limit():
        import processing
        from osgeo import gdal as _gdal

        (before_path,) = surface_dir.glob("walls/px*/wall_hts.tif")
        before = _gdal.Open(str(before_path)).ReadAsArray()
        processing.run(
            "solweig:surface_preprocessing",
            {"DSM": str(dsm_path), "OUTPUT_DIR": str(surface_dir), "WALL_LIMIT": 10.0},
        )
        after = _gdal.Open(str(before_path)).ReadAsArray()
        if not (after[60:80, 70:100].max() == 0.0 and before[60:80, 70:100].max() > 0.0 and after.max() > 10.0):
            raise RuntimeError("a 10 m wall limit did not remove the 9 m block's walls from the saved raster")
        processing.run("solweig:surface_preprocessing", {"DSM": str(dsm_path), "OUTPUT_DIR": str(surface_dir)})
        restored = _gdal.Open(str(before_path)).ReadAsArray()
        if not np.array_equal(restored, before):
            raise RuntimeError("the default wall limit did not restore the walls")
        return "custom limit applied and restored"

    step("algorithm: minimum wall height reaches the saved walls", prepare_again_with_wall_limit)

    def single():
        import processing
        from qgis.PyQt.QtCore import QDate, QDateTime, QTime

        out = OUT / "run_single"
        processing.run(
            "solweig:solweig_calculation",
            {
                "PREPARED_SURFACE_DIR": str(surface_dir),
                "WEATHER_SOURCE": 0,
                "DATETIME": QDateTime(QDate(2024, 7, 15), QTime(13, 0)),
                "TEMPERATURE": 25.0,
                "HUMIDITY": 50.0,
                "GLOBAL_RADIATION": 700.0,
                "COMPUTE_UTCI": True,
                "OUTPUT_DIR": str(out),
            },
        )
        tmrt = sorted(out.glob("tmrt/*.tif"))
        utci = sorted(out.glob("utci/*.tif"))
        if len(tmrt) != 1 or len(utci) != 1:
            raise RuntimeError(f"expected one Tmrt and one UTCI raster, found {len(tmrt)} and {len(utci)}")
        fraction = _finite_fraction(tmrt[0])
        if fraction < 0.5:
            raise RuntimeError(f"only {fraction:.0%} of Tmrt is finite")
        return f"Tmrt {fraction:.0%} finite, UTCI written"

    step("algorithm: single-timestep calculation", single)

    def series():
        import processing

        met = OUT / "met_10min.txt"
        _write_umep_met(met)
        counts = {}
        for hourly_only in (False, True):
            out = OUT / f"run_series_hourly_{hourly_only}"
            processing.run(
                "solweig:solweig_calculation",
                {
                    "PREPARED_SURFACE_DIR": str(surface_dir),
                    "WEATHER_SOURCE": 2,
                    "UMEP_MET_FILE": str(met),
                    "UMEP_HOURLY_ONLY": hourly_only,
                    "OUTPUT_DIR": str(out),
                },
            )
            tifs = sorted(out.glob("tmrt/*.tif"))
            counts[hourly_only] = len(tifs)
            if list(out.rglob("*.partial.tif")):
                raise RuntimeError("partial files left behind")
            if not (out / "summary").is_dir() or not list((out / "summary").glob("*.tif")):
                raise RuntimeError("no summary grids")
        if counts != {False: 12, True: 2}:
            raise RuntimeError(f"timestep counts {counts}, expected 12 for every record and 2 for hourly only")
        return "12 timesteps for every record, 2 for on-the-hour only"

    step("algorithm: sub-hourly UMEP met series", series)

    def unload():
        plugin = result.pop("_plugin", None)
        if plugin is not None:
            plugin.unload()
        return ""

    step("plugin unloads", unload)
    result.pop("_plugin", None)
    return result


def _main_inside_qgis() -> None:
    code = 1
    try:
        result = run_inside_qgis()
        (OUT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
        code = 0 if not result["failures"] else 1
    except BaseException:  # noqa: BLE001  (must always reach the exit below)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "result.json").write_text(
            json.dumps({"steps": [], "failures": [f"script crashed:\n{traceback.format_exc()}"]}, indent=2)
        )
    finally:
        sys.stdout.flush()
        os._exit(code)  # leave QGIS without waiting for the GUI


if "--compare" in sys.argv:
    if __name__ == "__main__":
        sys.exit(compare())
else:
    try:
        import qgis.core  # noqa: F401

        _inside_qgis = True
    except ImportError:
        _inside_qgis = False
    if _inside_qgis:
        _main_inside_qgis()
    elif __name__ == "__main__":
        print(__doc__)
        sys.exit(2)
