"""GPU vs CPU runtime ratio benchmark.

Complement to `test_performance_matrix_benchmark.py`, which runs whichever
backend is active. This test runs the **same scenario back-to-back** with
GPU on and GPU off (`solweig.disable_gpu()`), so we get a measured
speedup ratio every time it runs.

Why it exists:
- The perf matrix records whether GPU was available but mixes GPU and CPU
  runs across history rows. There's no single number for "GPU is X×
  faster than CPU on this codebase right now."
- This benchmark produces that number. Trend it over time via the
  appended log to see when a code change makes the GPU path relatively
  slower (a regression) or relatively faster (a wgsl/buffer-cache win).
- Skipped automatically when no GPU is available, so CI on GPU-less
  runners is unaffected.

The assertions are intentionally loose — the goal is a measurement
floor, not a tight gate. CI runs on GPU-less hosts skip; local M-series
runs against the floor.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import solweig
from solweig import Location, SurfaceData, Weather, calculate

pytestmark = pytest.mark.slow

# Same logging directory as the existing perf-matrix benchmark.
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_GPU_CPU_LOG = _LOG_DIR / "gpu_cpu_ratio_history.md"

# Minimum acceptable GPU/CPU speedup. The GPU path is usually 2–4× the
# CPU on M-series for our scenarios, so anything under 1.0 is a
# regression (GPU slower than CPU) and anything under 0.5 is alarming
# (GPU at least 2× slower — fix immediately). The benchmark warns rather
# than fails between 1.0 and 0.5 so CI flakiness doesn't turn a
# slow-but-functional GPU into a red build.
MIN_GPU_CPU_SPEEDUP_FAIL = 0.5
MIN_GPU_CPU_SPEEDUP_WARN = 1.0


def _make_surface(size: int = 256) -> SurfaceData:
    """A modestly sized scene where GPU dispatch overhead is amortised."""
    from conftest import make_mock_svf

    rng = np.random.default_rng(42)
    dsm = np.ones((size, size), dtype=np.float32) * 5.0
    # A few buildings of varying height for shadow structure.
    for _ in range(8):
        r, c = rng.integers(15, size - 30, 2)
        h, w = rng.integers(10, 25, 2)
        dsm[r : r + h, c : c + w] = rng.uniform(10.0, 25.0)
    land_cover = np.ones((size, size), dtype=np.int32) * 5
    land_cover[dsm > 8] = 2  # buildings tagged
    return SurfaceData(
        dsm=dsm,
        land_cover=land_cover,
        pixel_size=1.0,
        svf=make_mock_svf((size, size)),
    )


def _run_calculation(surface: SurfaceData, location: Location, weather: Weather) -> None:
    """Single-timestep run; output discarded (only timing matters)."""
    surface.preprocess()
    with tempfile.TemporaryDirectory(prefix="solweig-gpu-cpu-") as td:
        calculate(
            surface,
            [weather],
            location,
            output_dir=Path(td),
            outputs=["tmrt"],
        )


def _median_runtime(fn, repeats: int = 5) -> tuple[float, list[float]]:
    """Warm up once, then return median and all samples."""
    fn()  # warm-up dispatch (lazy GPU context init, JIT caches, etc.)
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples


def _append_log_entry(
    gpu_median: float, cpu_median: float, ratio: float, gpu_samples: list[float], cpu_samples: list[float]
) -> None:
    """Append one row to the GPU/CPU ratio history log."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    backend = solweig.get_gpu_limits() or {}
    backend_name = backend.get("backend", "unknown")
    new_file = not _GPU_CPU_LOG.exists()
    lines = []
    if new_file:
        lines.append("# GPU vs CPU runtime ratio history\n\n")
        lines.append(
            "One row per benchmark run. `gpu_s` and `cpu_s` are medians of 5 runs each\n"
            "(after warm-up). `ratio` is `cpu_s / gpu_s`; higher means GPU is faster.\n\n"
        )
        lines.append("| timestamp | backend | gpu_s | cpu_s | ratio | gpu_samples | cpu_samples |\n")
        lines.append("|---|---|---:|---:|---:|---|---|\n")
    lines.append(
        f"| {timestamp} | {backend_name} | {gpu_median:.4f} | {cpu_median:.4f} | {ratio:.2f} | "
        f"{','.join(f'{s:.4f}' for s in gpu_samples)} | {','.join(f'{s:.4f}' for s in cpu_samples)} |\n"
    )
    with _GPU_CPU_LOG.open("a", encoding="utf-8") as f:
        f.writelines(lines)


def test_gpu_cpu_runtime_ratio():
    """Measure and assert the GPU/CPU runtime ratio on a fixed scenario.

    Skipped when no GPU is available (CI). Local M-series Macs typically
    show a 2–4× GPU speedup on this scenario; the assertions are loose
    to absorb run-to-run variance.
    """
    if not solweig.is_gpu_available():
        pytest.skip("GPU not available on this host")
    if os.environ.get("SOLWEIG_NO_GPU"):
        pytest.skip("SOLWEIG_NO_GPU is set")

    location = Location(latitude=37.98, longitude=23.73, utc_offset=2)
    weather = Weather(
        datetime=datetime(2024, 7, 15, 12, 0),
        ta=30.0,
        rh=50.0,
        global_rad=800.0,
        ws=2.0,
    )

    # --- GPU run ---
    solweig.enable_gpu()
    solweig.reset_gpu_metrics()
    gpu_median, gpu_samples = _median_runtime(
        lambda: _run_calculation(_make_surface(), location, weather),
    )
    gpu_dispatches = solweig.gpu_dispatch_count()
    gpu_fallbacks = solweig.gpu_fallback_count()
    if gpu_dispatches == 0:
        pytest.skip(
            "GPU path did not dispatch even though is_gpu_available() is True — host-specific compatibility issue."
        )
    assert gpu_fallbacks == 0, (
        f"GPU run unexpectedly fell back to CPU {gpu_fallbacks} times — "
        "the recorded GPU time isn't measuring the GPU path."
    )

    # --- CPU run ---
    solweig.disable_gpu()
    solweig.reset_gpu_metrics()
    try:
        cpu_median, cpu_samples = _median_runtime(
            lambda: _run_calculation(_make_surface(), location, weather),
        )
        assert solweig.gpu_dispatch_count() == 0, "CPU run dispatched the GPU"
    finally:
        solweig.enable_gpu()  # restore for subsequent tests

    ratio = cpu_median / gpu_median if gpu_median > 0 else float("inf")
    _append_log_entry(gpu_median, cpu_median, ratio, gpu_samples, cpu_samples)

    # The hard floor — if the GPU is slower than half CPU speed, something
    # is genuinely wrong (driver fallback, buffer thrash, shader bug).
    assert ratio >= MIN_GPU_CPU_SPEEDUP_FAIL, (
        f"GPU is significantly slower than CPU on this scenario: "
        f"GPU median {gpu_median:.3f}s, CPU median {cpu_median:.3f}s, "
        f"ratio {ratio:.2f} < {MIN_GPU_CPU_SPEEDUP_FAIL} floor. "
        f"Check `tests/benchmarks/logs/gpu_cpu_ratio_history.md` for trend."
    )

    # Soft warning — GPU should at least be no slower than CPU.
    if ratio < MIN_GPU_CPU_SPEEDUP_WARN:
        import warnings as _warnings

        _warnings.warn(
            f"GPU/CPU speedup ratio {ratio:.2f} < {MIN_GPU_CPU_SPEEDUP_WARN} — "
            f"GPU is not faster than CPU on this scenario "
            f"(GPU {gpu_median:.3f}s, CPU {cpu_median:.3f}s). "
            "May indicate a GPU-path regression worth investigating.",
            stacklevel=2,
        )
