"""
Benchmark: UMEP Python shadowingfunction_wallheight_23 vs Rust calculate_shadows_wall_ht_25

Uses Athens demo fixture data (400x400 DSM), optionally tiled to 800x800 (4 tiles) or 1600x1600 (16 tiles).

Usage:
    uv run python scripts/benchmark_shadows.py
    uv run python scripts/benchmark_shadows.py --repeats 5 --tiles 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from solweig.rustalgos import shadowing

FIXTURES = Path(__file__).parent.parent / "tests" / "golden" / "fixtures"

SUN_POSITIONS = [
    {"name": "morning", "azimuth": 90.0, "altitude": 30.0},
    {"name": "noon", "azimuth": 180.0, "altitude": 60.0},
    {"name": "afternoon", "azimuth": 270.0, "altitude": 45.0},
]


def load_data(tiles: int = 1):
    """Load Athens fixture data, optionally tiled into a larger grid.

    Args:
        tiles: Number of tiles (1, 4, or 16). 4 → 2×2 grid, 16 → 4×4 grid.
    """
    if tiles not in (1, 4, 16):
        raise ValueError(f"tiles must be 1, 4, or 16; got {tiles}")
    n = {1: 1, 4: 2, 16: 4}[tiles]

    dsm = np.load(FIXTURES / "input_dsm.npy").astype(np.float32)
    cdsm = np.load(FIXTURES / "input_cdsm.npy").astype(np.float32)
    tdsm = np.load(FIXTURES / "input_tdsm.npy").astype(np.float32)
    bush = np.load(FIXTURES / "input_bush.npy").astype(np.float32)
    wall_ht = np.load(FIXTURES / "input_wall_ht.npy").astype(np.float32)
    wall_asp = np.load(FIXTURES / "input_wall_asp.npy").astype(np.float32)
    params = dict(np.load(FIXTURES / "input_params.npz"))
    scale = float(params["scale"])
    amaxvalue = float(params["amaxvalue"])

    if n > 1:
        dsm = np.tile(dsm, (n, n))
        cdsm = np.tile(cdsm, (n, n))
        tdsm = np.tile(tdsm, (n, n))
        bush = np.tile(bush, (n, n))
        wall_ht = np.tile(wall_ht, (n, n))
        wall_asp = np.tile(wall_asp, (n, n))

    return dsm, cdsm, tdsm, bush, wall_ht, wall_asp, scale, amaxvalue


def time_fn(fn, repeats: int):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def benchmark_umep(dsm, cdsm, tdsm, bush, wall_ht, wall_asp, scale, amaxvalue, pos, repeats):
    sys.path.insert(0, str(Path(__file__).parent.parent))  # for tests.reference_package
    from tests.reference_package import use_umep_solweig

    use_umep_solweig()
    from umep_solweig.util.SEBESOLWEIGCommonFiles.shadowingfunction_wallheight_23 import (
        shadowingfunction_wallheight_23,
    )

    # wall_asp in degrees for UMEP (fixture stores degrees)
    wall_asp_deg = wall_asp

    def run():
        shadowingfunction_wallheight_23(
            dsm,
            cdsm,
            tdsm,
            pos["azimuth"],
            pos["altitude"],
            scale,
            amaxvalue,
            bush,
            wall_ht,
            wall_asp_deg,
        )

    # warmup
    run()
    return time_fn(run, repeats)


def benchmark_rust(dsm, cdsm, tdsm, bush, wall_ht, wall_asp, scale, amaxvalue, pos, repeats):
    # wall_asp stored in degrees in fixture (generated from wallaspect * pi/180 then back?)
    # Check generate_fixtures.py: wall_asp = SWC.raster_data.wallaspect * pi/180 → radians
    # So fixture is in radians already
    wall_asp_rad = wall_asp

    def run():
        shadowing.calculate_shadows_wall_ht_25(
            azimuth_deg=pos["azimuth"],
            altitude_deg=pos["altitude"],
            scale=scale,
            max_local_dsm_ht=amaxvalue,
            dsm=dsm,
            veg_canopy_dsm=cdsm,
            veg_trunk_dsm=tdsm,
            bush=bush,
            walls=wall_ht,
            aspect=wall_asp_rad,
        )

    # warmup
    run()
    return time_fn(run, repeats)


def _fmt_samples(samples: list[float]) -> str:
    avg = sum(samples) / len(samples)
    return f"min={min(samples):.3f}s  avg={avg:.3f}s  max={max(samples):.3f}s"


def _print_table(rows: list[tuple[str, list[float], float | None]]) -> None:
    """Print a benchmark result table.

    rows: [(label, samples, speedup_vs_umep), ...]
    speedup_vs_umep is None for the UMEP baseline row.
    """
    col_label = 16
    col_timing = 36
    col_speedup = 14
    rule = "-" * (col_label + col_timing + col_speedup)

    header = f"{'backend':<{col_label}}{'timing':<{col_timing}}{'speedup':>{col_speedup}}"
    print(f"  {header}")
    print(f"  {rule}")
    for label, samples, speedup in rows:
        timing = _fmt_samples(samples)
        speedup_str = f"{speedup:.1f}x faster" if speedup is not None else "—"
        print(f"  {label:<{col_label}}{timing:<{col_timing}}{speedup_str:>{col_speedup}}")


def main(repeats: int, pos: str, tiles: int):
    parser = argparse.ArgumentParser(description="Benchmark UMEP Python vs Rust shadow calculation")
    parser.add_argument(
        "--repeats", type=int, default=repeats, help=f"Number of timed repetitions (default: {repeats})"
    )
    parser.add_argument(
        "--pos",
        choices=["morning", "noon", "afternoon", "all"],
        default=pos,
        help=f"Sun position to benchmark (default: {pos})",
    )
    parser.add_argument(
        "--tiles",
        type=int,
        choices=[1, 4, 16],
        default=tiles,
        help=f"Number of Athens tiles to stitch (1=400x400, 4=800x800, 16=1600x1600; default: {tiles})",
    )
    args = parser.parse_args()

    print("Loading Athens fixture data...")
    dsm, cdsm, tdsm, bush, wall_ht, wall_asp, scale, amaxvalue = load_data(args.tiles)
    h, w = dsm.shape
    tile_str = f"{args.tiles} tile{'s' if args.tiles > 1 else ''}"
    print(f"  Grid: {w}×{h}  ({tile_str})  |  scale={scale}m  |  max_ht={amaxvalue:.1f}m  |  repeats={args.repeats}")

    sun_positions = SUN_POSITIONS if args.pos == "all" else [p for p in SUN_POSITIONS if p["name"] == args.pos]

    for sun_pos in sun_positions:
        print(f"\n  Sun: {sun_pos['name']}  (az={sun_pos['azimuth']}°, alt={sun_pos['altitude']}°)")
        print()

        rows: list[tuple[str, list[float], float | None]] = []

        # --- UMEP Python ---
        umep_avg = None
        try:
            umep_samples = benchmark_umep(
                dsm, cdsm, tdsm, bush, wall_ht, wall_asp, scale, amaxvalue, sun_pos, args.repeats
            )
            umep_avg = sum(umep_samples) / len(umep_samples)
            rows.append(("UMEP Python", umep_samples, None))
        except ImportError:
            rows.append(("UMEP Python", [], None))

        # --- Rust CPU ---
        shadowing.disable_gpu()
        rust_cpu_samples = benchmark_rust(
            dsm, cdsm, tdsm, bush, wall_ht, wall_asp, scale, amaxvalue, sun_pos, args.repeats
        )
        rust_cpu_avg = sum(rust_cpu_samples) / len(rust_cpu_samples)
        cpu_speedup = umep_avg / rust_cpu_avg if umep_avg is not None else None
        rows.append(("Rust (CPU)", rust_cpu_samples, cpu_speedup))

        # --- Rust GPU ---
        shadowing.enable_gpu()
        if shadowing.is_gpu_enabled():
            rust_gpu_samples = benchmark_rust(
                dsm, cdsm, tdsm, bush, wall_ht, wall_asp, scale, amaxvalue, sun_pos, args.repeats
            )
            rust_gpu_avg = sum(rust_gpu_samples) / len(rust_gpu_samples)
            gpu_speedup = umep_avg / rust_gpu_avg if umep_avg is not None else None
            rows.append(("Rust (GPU)", rust_gpu_samples, gpu_speedup))
        else:
            rows.append(("Rust (GPU)", [], None))

        _print_table(rows)


if __name__ == "__main__":
    main(repeats=3, pos="noon", tiles=16)
