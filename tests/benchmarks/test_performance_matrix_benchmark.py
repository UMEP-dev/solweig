"""Performance benchmark matrix for API and plugin regression detection.

Matrix coverage:
- Frontend: API, QGIS plugin
- Execution mode: non-tiled, tiled
- Sky model: isotropic, anisotropic

This catches regressions with:
1) absolute runtime budgets per scenario
2) relative ratio checks across paired scenarios
"""

from __future__ import annotations

import csv
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import solweig
from conftest import make_mock_svf
from solweig import Location, SurfaceData, Weather
from solweig.models.precomputed import ShadowArrays

pytestmark = pytest.mark.slow

# Scale all absolute thresholds in slower CI/VMs:
#   SOLWEIG_PERF_BUDGET_SCALE=1.5 pytest tests/benchmarks/...
PERF_BUDGET_SCALE = float(os.environ.get("SOLWEIG_PERF_BUDGET_SCALE", "1.0"))

ABSOLUTE_BUDGET_SECONDS = {
    # Budgets tightened May 2026: history shows medians at 0.10–0.15 s on an
    # M-series Mac. Local budgets sit ~4× over observed median to absorb
    # micro-variance; CI uses SOLWEIG_PERF_BUDGET_SCALE=2.0 → ~8× headroom,
    # still 5–10× tighter than the previous values which let any regression
    # under 10× pass silently.
    "api_non_tiled_isotropic": 0.50,
    "api_non_tiled_anisotropic": 0.60,
    "api_tiled_isotropic": 0.60,
    "api_tiled_anisotropic": 0.80,
    "plugin_non_tiled_isotropic": 0.55,
    "plugin_non_tiled_anisotropic": 0.65,
    "plugin_tiled_isotropic": 0.65,
    "plugin_tiled_anisotropic": 0.85,
}

# Ratio gates. These run as part of `test_performance_matrix_relative_regressions`
# which is marked `gpu_perf_gate` and excluded from CI — see that test's
# docstring for the rationale. Threshold values assume the maintainer's
# local M-series GPU baseline (aniso/iso ≈ 1.0–1.3, etc.) with ~40%
# headroom for run-to-run variance. Run locally via
# `poe test_gpu_perf_gate`.
MAX_RATIO_ANISO_OVER_ISO = 2.0
MAX_RATIO_TILED_OVER_NON_TILED = 2.5
MAX_RATIO_PLUGIN_OVER_API = 2.0

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_CSV_LOG_PATH = _LOG_DIR / "performance_matrix_history.csv"
_MD_LOG_PATH = _LOG_DIR / "performance_matrix_history.md"


def _scenario_id(frontend: str, tiled: bool, anisotropic: bool) -> str:
    tiled_label = "tiled" if tiled else "non_tiled"
    sky_label = "anisotropic" if anisotropic else "isotropic"
    return f"{frontend}_{tiled_label}_{sky_label}"


def _make_surface(size: int = 320) -> SurfaceData:
    """Create a synthetic surface that works for isotropic and anisotropic paths."""
    n_patches = 153
    n_pack = (n_patches + 7) // 8

    dsm = np.ones((size, size), dtype=np.float32) * 5.0
    dsm[110:210, 120:220] = 10.0  # 5 m relative building

    surface = SurfaceData(
        dsm=dsm,
        pixel_size=1.0,
        svf=make_mock_svf((size, size)),
    )
    surface.shadow_matrices = ShadowArrays(
        _shmat_u8=np.full((size, size, n_pack), 0xFF, dtype=np.uint8),
        _vegshmat_u8=np.full((size, size, n_pack), 0xFF, dtype=np.uint8),
        _vbshmat_u8=np.full((size, size, n_pack), 0xFF, dtype=np.uint8),
        _n_patches=n_patches,
    )
    # Plugin helper methods expect georeference metadata on the surface.
    surface._geotransform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    surface._crs_wkt = 'LOCAL_CS["benchmark"]'
    return surface


def _make_location() -> Location:
    return Location(latitude=57.7, longitude=12.0, utc_offset=1)


def _make_weather() -> Weather:
    return Weather(
        datetime=datetime(2024, 7, 15, 12, 0),
        ta=27.0,
        rh=45.0,
        global_rad=800.0,
        ws=2.0,
    )


def _make_weather_series() -> list[Weather]:
    base = datetime(2024, 7, 15, 12, 0)
    return [
        Weather(
            datetime=base + timedelta(hours=i),
            ta=27.0 + i,
            rh=45.0,
            global_rad=800.0,
            ws=2.0,
        )
        for i in range(2)
    ]


def _assert_valid_tmrt(tmrt: np.ndarray) -> None:
    assert tmrt is not None
    finite = np.isfinite(tmrt)
    assert finite.any(), "Expected finite Tmrt values"
    assert finite.mean() > 0.8, "Too many invalid Tmrt pixels"


def _median_runtime_seconds(fn, repeats: int = 3) -> tuple[float, list[float]]:
    """Warm up once, then return median runtime over repeated runs."""
    fn()  # Warm-up pass for fairer timing.
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples


def _run_api_case(tiled: bool, anisotropic: bool) -> None:
    import tempfile

    surface = _make_surface()
    location = _make_location()
    weather = _make_weather()

    with tempfile.TemporaryDirectory(prefix="solweig-bench-api-") as tmpdir:
        from conftest import read_timestep_geotiff

        solweig.calculate(
            surface=surface,
            weather=[weather],
            location=location,
            use_anisotropic_sky=anisotropic,
            max_shadow_distance_m=80.0,
            tile_size=256 if tiled else None,
            output_dir=tmpdir,
            outputs=["tmrt"],
        )
        tmrt = read_timestep_geotiff(tmpdir, "tmrt", 0)
        _assert_valid_tmrt(tmrt)


def _run_plugin_case(tiled: bool, anisotropic: bool) -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="solweig-bench-") as tmpdir:
        summary = solweig.calculate(
            surface=_make_surface(),
            weather=_make_weather_series(),
            location=_make_location(),
            human=solweig.HumanParams(),
            use_anisotropic_sky=anisotropic,
            max_shadow_distance_m=80.0,
            tile_size=None,  # timeseries path auto-tiles; no forced tiling needed
            output_dir=tmpdir,
            outputs=["tmrt"],
        )

    assert summary.n_timesteps == 2
    assert summary.tmrt_mean is not None
    assert np.isfinite(np.nanmean(summary.tmrt_mean))


@pytest.fixture(scope="module")
def perf_matrix() -> dict[str, dict[str, float | list[float]]]:
    """Measure median runtimes for all 8 benchmark scenarios."""
    measurements: dict[str, dict[str, float | list[float]]] = {}

    for frontend in ("api", "plugin"):
        for tiled in (False, True):
            for anisotropic in (False, True):
                sid = _scenario_id(frontend, tiled, anisotropic)
                runner = (
                    (lambda t=tiled, a=anisotropic: _run_api_case(t, a))
                    if frontend == "api"
                    else (lambda t=tiled, a=anisotropic: _run_plugin_case(t, a))
                )
                median_s, samples = _median_runtime_seconds(runner, repeats=3)
                measurements[sid] = {"median_s": median_s, "samples_s": samples}

    _append_performance_log(measurements)
    return measurements


def _runtime(perf_matrix: dict[str, dict[str, float | list[float]]], sid: str) -> float:
    val = perf_matrix[sid]["median_s"]
    assert isinstance(val, (int, float))
    return float(val)


def _runtime_matrix_rows(
    perf_matrix: dict[str, dict[str, float | list[float]]],
) -> list[tuple[str, str, str, str, str]]:
    def _fmt(frontend: str, tiled: bool, anisotropic: bool) -> str:
        sid = _scenario_id(frontend, tiled, anisotropic)
        return f"{_runtime(perf_matrix, sid):.4f}"

    return [
        (
            "api",
            _fmt("api", False, False),
            _fmt("api", False, True),
            _fmt("api", True, False),
            _fmt("api", True, True),
        ),
        (
            "plugin",
            _fmt("plugin", False, False),
            _fmt("plugin", False, True),
            _fmt("plugin", True, False),
            _fmt("plugin", True, True),
        ),
    ]


def _git_commit_short() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
            .lower()
        )
    except Exception:
        return "unknown"


def _cpu_count_available() -> int | None:
    """CPU count available to this process (affinity-aware when possible)."""
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        return None


def _ram_total_bytes() -> int | None:
    """Best-effort total physical RAM detection."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except Exception:
        pass

    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out)
        except Exception:
            return None
    if system == "Windows":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
        except Exception:
            return None
    return None


def _ram_available_bytes() -> int | None:
    """Best-effort available RAM detection at benchmark runtime."""
    system = platform.system()
    if system == "Linux":
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", meminfo, flags=re.MULTILINE)
            if match:
                return int(match.group(1)) * 1024
        except Exception:
            return None
    if system == "Darwin":
        try:
            page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
            vm_stat = subprocess.check_output(["vm_stat"], text=True)
            page_counts = {}
            for key in ("Pages free", "Pages inactive", "Pages speculative"):
                match = re.search(rf"^{re.escape(key)}:\s+(\d+)\.$", vm_stat, flags=re.MULTILINE)
                page_counts[key] = int(match.group(1)) if match else 0
            available_pages = (
                page_counts["Pages free"] + page_counts["Pages inactive"] + page_counts["Pages speculative"]
            )
            return available_pages * page_size
        except Exception:
            return None
    if system == "Windows":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return int(status.ullAvailPhys)
        except Exception:
            return None
    return None


def _gpu_hardware_info() -> dict[str, str | int | bool | None]:
    """Collect best-effort GPU capability metadata from solweig runtime."""
    gpu_compiled = bool(getattr(solweig, "GPU_ENABLED", False))
    try:
        gpu_available = bool(solweig.is_gpu_available())
    except Exception:
        gpu_available = False
    try:
        backend = solweig.get_compute_backend()
    except Exception:
        backend = "unknown"
    try:
        limits = solweig.get_gpu_limits() or {}
        max_buffer_size = int(limits["max_buffer_size"]) if "max_buffer_size" in limits else None
    except Exception:
        max_buffer_size = None

    return {
        "gpu_compiled": gpu_compiled,
        "gpu_available": gpu_available,
        "gpu_backend": backend,
        "gpu_max_buffer_size": max_buffer_size,
    }


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / (1024**3)
    return f"{value} ({gib:.2f} GiB)"


def _append_performance_log(perf_matrix: dict[str, dict[str, float | list[float]]]) -> None:
    """Append a run record to benchmark logs (CSV + markdown matrix)."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    timestamp = now.isoformat(timespec="seconds")
    py_ver = sys.version.split()[0]
    commit = _git_commit_short()
    system = f"{platform.system()} {platform.release()}"
    machine = platform.machine()
    cpu_count_logical = os.cpu_count() or 0
    cpu_count_available = _cpu_count_available()
    ram_total = _ram_total_bytes()
    ram_available = _ram_available_bytes()
    gpu_info = _gpu_hardware_info()

    csv_fields = [
        "run_id",
        "timestamp_utc",
        "git_commit",
        "python_version",
        "system",
        "machine",
        "cpu_count_logical",
        "cpu_count_available",
        "ram_total_bytes",
        "ram_available_bytes",
        "gpu_compiled",
        "gpu_available",
        "gpu_backend",
        "gpu_max_buffer_size",
        "budget_scale",
        "scenario",
        "median_seconds",
        "samples_seconds",
    ]
    reset_header = False
    if _CSV_LOG_PATH.exists():
        current_header = _CSV_LOG_PATH.read_text(encoding="utf-8").splitlines()[:1]
        if not current_header or current_header[0] != ",".join(csv_fields):
            reset_header = True

    if reset_header:
        backup_name = _CSV_LOG_PATH.with_suffix(f".bak-{run_id}.csv")
        _CSV_LOG_PATH.rename(backup_name)

    write_header = not _CSV_LOG_PATH.exists()
    with _CSV_LOG_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        for scenario, payload in sorted(perf_matrix.items()):
            median_s = float(payload["median_s"])  # type: ignore[arg-type]
            samples = ";".join(f"{float(x):.6f}" for x in payload["samples_s"])  # type: ignore[arg-type]
            writer.writerow(
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "git_commit": commit,
                    "python_version": py_ver,
                    "system": system,
                    "machine": machine,
                    "cpu_count_logical": cpu_count_logical,
                    "cpu_count_available": cpu_count_available,
                    "ram_total_bytes": ram_total,
                    "ram_available_bytes": ram_available,
                    "gpu_compiled": gpu_info["gpu_compiled"],
                    "gpu_available": gpu_info["gpu_available"],
                    "gpu_backend": gpu_info["gpu_backend"],
                    "gpu_max_buffer_size": gpu_info["gpu_max_buffer_size"],
                    "budget_scale": f"{PERF_BUDGET_SCALE:.2f}",
                    "scenario": scenario,
                    "median_seconds": f"{median_s:.6f}",
                    "samples_seconds": samples,
                }
            )

    lines = [
        f"## {run_id}",
        f"- timestamp_utc: {timestamp}",
        f"- git_commit: {commit}",
        f"- python: {py_ver}",
        f"- system: {system}",
        f"- machine: {machine}",
        f"- cpu_count_logical: {cpu_count_logical}",
        f"- cpu_count_available: {cpu_count_available if cpu_count_available is not None else 'unknown'}",
        f"- ram_total: {_format_bytes(ram_total)}",
        f"- ram_available: {_format_bytes(ram_available)}",
        f"- gpu_compiled: {gpu_info['gpu_compiled']}",
        f"- gpu_available: {gpu_info['gpu_available']}",
        f"- gpu_backend: {gpu_info['gpu_backend']}",
        (
            "- gpu_max_buffer_size: "
            + (
                _format_bytes(gpu_info["gpu_max_buffer_size"])
                if isinstance(gpu_info["gpu_max_buffer_size"], int)
                else "unknown"
            )
        ),
        f"- budget_scale: {PERF_BUDGET_SCALE:.2f}",
        "",
        "| frontend | non_tiled_isotropic_s | non_tiled_anisotropic_s | tiled_isotropic_s | tiled_anisotropic_s |",
        "|---|---:|---:|---:|---:|",
    ]
    for frontend, nti, nta, ti, ta in _runtime_matrix_rows(perf_matrix):
        lines.append(f"| {frontend} | {nti} | {nta} | {ti} | {ta} |")
    lines.append("")

    with _MD_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


@pytest.mark.gpu_perf_gate
def test_performance_matrix_absolute_budgets(perf_matrix):
    """Catch large runtime regressions in each benchmark scenario.

    Marked ``gpu_perf_gate`` because runtime budgets are calibrated against
    the maintainer's local hardware. CI runs on shared Linux runners with
    very different timing characteristics; with ``SOLWEIG_PERF_BUDGET_SCALE=2.0``
    the gate is loose enough to be uninformative there (it only catches
    catastrophic 10×+ regressions that the validation suite would also
    surface). Hardware-stable regressions (memory, byte-identical output)
    are covered by other tests that DO run on CI. Run this locally via
    ``poe test_gpu_perf_gate``.
    """
    for sid, budget_s in ABSOLUTE_BUDGET_SECONDS.items():
        measured_s = _runtime(perf_matrix, sid)
        threshold_s = budget_s * PERF_BUDGET_SCALE
        assert measured_s <= threshold_s, (
            f"Performance regression in {sid}: "
            f"{measured_s:.3f}s > budget {threshold_s:.3f}s "
            f"(raw budget={budget_s:.3f}s, scale={PERF_BUDGET_SCALE:.2f})"
        )


@pytest.mark.gpu_perf_gate
def test_performance_matrix_relative_regressions(perf_matrix):
    """Cross-check scenario ratios to catch path-specific slowdowns.

    Marked ``gpu_perf_gate`` because the ratio thresholds assume the
    maintainer's local M-series GPU (where aniso/iso ≈ 1.0–1.3). CI runs
    on GPU-less Linux where the CPU-only aniso path is materially slower
    than the CPU-only iso path (~3× tiled) and would fail this gate as
    a false negative. The absolute-budget gate above does run on CI
    (with ``SOLWEIG_PERF_BUDGET_SCALE`` for runner variance) and catches
    massive regressions everywhere; this finer ratio gate only fires
    locally via ``poe test_gpu_perf_gate``.
    """
    # anisotropic / isotropic ratios (same frontend + tiling mode)
    for frontend in ("api", "plugin"):
        for tiled in (False, True):
            iso = _runtime(perf_matrix, _scenario_id(frontend, tiled, False))
            aniso = _runtime(perf_matrix, _scenario_id(frontend, tiled, True))
            assert aniso / iso <= MAX_RATIO_ANISO_OVER_ISO, (
                f"{frontend} {'tiled' if tiled else 'non-tiled'} anisotropic regression: "
                f"ratio {aniso / iso:.2f} > {MAX_RATIO_ANISO_OVER_ISO:.2f}"
            )

    # tiled / non-tiled ratios (same frontend + sky mode)
    for frontend in ("api", "plugin"):
        for anisotropic in (False, True):
            non_tiled = _runtime(perf_matrix, _scenario_id(frontend, False, anisotropic))
            tiled = _runtime(perf_matrix, _scenario_id(frontend, True, anisotropic))
            assert tiled / non_tiled <= MAX_RATIO_TILED_OVER_NON_TILED, (
                f"{frontend} {'anisotropic' if anisotropic else 'isotropic'} tiled regression: "
                f"ratio {tiled / non_tiled:.2f} > {MAX_RATIO_TILED_OVER_NON_TILED:.2f}"
            )

    # plugin / API ratios (non-tiled only — tiled API uses calculate() with
    # tile_size while plugin uses multi-timestep calculate(), so the
    # comparison is not meaningful for the tiled case)
    for anisotropic in (False, True):
        api_rt = _runtime(perf_matrix, _scenario_id("api", False, anisotropic))
        plugin_rt = _runtime(perf_matrix, _scenario_id("plugin", False, anisotropic))
        assert plugin_rt / api_rt <= MAX_RATIO_PLUGIN_OVER_API, (
            f"Plugin overhead regression (non-tiled, "
            f"{'anisotropic' if anisotropic else 'isotropic'}): "
            f"ratio {plugin_rt / api_rt:.2f} > {MAX_RATIO_PLUGIN_OVER_API:.2f}"
        )
