"""
POI sensitivity sweep for all three Gothenburg validation sites.

Runs the full SOLWEIG pipeline for each site/day, then evaluates Tmrt at a grid
of candidate pixels.  Outputs georeferenced RMSE/R²/bias rasters so the spatial
pattern of model-observation agreement can be visualised in QGIS or similar.

Usage:
    pytest tests/validation/test_poi_sweep_all_sites.py -v -s --tb=short
    pytest tests/validation/test_poi_sweep_all_sites.py -v -s -k kronenhuset
    pytest tests/validation/test_poi_sweep_all_sites.py -v -s -k gvc
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio

from validation.met_clock import load_site_met

# ---------------------------------------------------------------------------
# Site configurations
# ---------------------------------------------------------------------------

VALIDATION_DIR = Path(__file__).parent

SITES: dict[str, dict[str, Any]] = {
    "kronenhuset": {
        "dsm": str(VALIDATION_DIR / "kronenhuset" / "DSM_KR.tif"),
        "dem": str(VALIDATION_DIR / "kronenhuset" / "DEM_KR.tif"),
        "cdsm": str(VALIDATION_DIR / "kronenhuset" / "CDSM_KR.asc"),
        "land_cover": str(VALIDATION_DIR / "kronenhuset" / "landcover_KR.tif"),
        "measurements_csv": VALIDATION_DIR / "kronenhuset" / "measurements_kr.csv",
        "params_json": VALIDATION_DIR / "kronenhuset" / "parametersforsolweig_KR.json",
        "met_files": {
            "20051007": VALIDATION_DIR / "kronenhuset" / "MetFile_Prepared.txt",
        },
        "day_key": None,  # single-day CSV, no day column
        "hour_key": "hour",
        "current_poi": (51, 117),
        "lat": 57.7,
        "lon": 12.0,
        "utc_offset": 1,
        "step": 2,
        "out_dir": VALIDATION_DIR / "kronenhuset" / "poi_sweep_results",
    },
    "gustav_adolfs": {
        "dsm": str(VALIDATION_DIR / "gustav_adolfs" / "DSM_GA.tif"),
        "dem": str(VALIDATION_DIR / "gustav_adolfs" / "DEM_GA.tif"),
        "cdsm": str(VALIDATION_DIR / "gustav_adolfs" / "CDSM_GA.tif"),
        "land_cover": str(VALIDATION_DIR / "gustav_adolfs" / "LC_GA.tif"),
        "measurements_csv": VALIDATION_DIR / "gustav_adolfs" / "measurements_ga.csv",
        "params_json": VALIDATION_DIR / "gustav_adolfs" / "parametersforsolweig_GA.json",
        "met_files": {
            "20051011": VALIDATION_DIR / "gustav_adolfs" / "MetFile_20051011.txt",
            "20060726": VALIDATION_DIR / "gustav_adolfs" / "MetFile_20060726.txt",
            "20060801": VALIDATION_DIR / "gustav_adolfs" / "MetFile_20060801.txt",
        },
        "day_key": "day",
        "hour_key": "hour",
        "current_poi": (33, 77),
        "lat": 57.7,
        "lon": 12.0,
        "utc_offset": 1,
        "step": 2,
        "out_dir": VALIDATION_DIR / "gustav_adolfs" / "poi_sweep_results",
    },
    "gvc": {
        "dsm": str(VALIDATION_DIR / "gvc" / "DSM_GVC_1m.tif"),
        "dem": str(VALIDATION_DIR / "gvc" / "DEM_GVC_1m.tif"),
        "cdsm": str(VALIDATION_DIR / "gvc" / "CDSM_GVC_1m.tif"),
        "land_cover": str(VALIDATION_DIR / "gvc" / "landcover_1m_GVC.tif"),
        "measurements_csv": VALIDATION_DIR / "gvc" / "measurements_gvc.csv",
        "params_json": VALIDATION_DIR / "gvc" / "parametersforsolweig_GVC.json",
        "met_files": {
            "20100707": VALIDATION_DIR / "gvc" / "MetFile20100707_Prepared.txt",
            "20100710": VALIDATION_DIR / "gvc" / "MetFile20100710_Prepared.txt",
            "20100712": VALIDATION_DIR / "gvc" / "MetFile20100712_Prepared.txt",
        },
        "day_key": "date",
        "hour_key": "hour",
        "current_poi": (51, 122),
        "lat": 57.7,
        "lon": 12.0,
        "utc_offset": 1,
        "step": 3,
        "out_dir": VALIDATION_DIR / "gvc" / "poi_sweep_results",
    },
}

pytestmark = [
    pytest.mark.validation,
    pytest.mark.slow,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_measurements(cfg: dict) -> dict[str, list[dict]]:
    """Load measurement CSV into {day_code: [row_dicts]}."""
    days: dict[str, list[dict]] = defaultdict(list)
    with open(cfg["measurements_csv"]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row[cfg["day_key"]] if cfg["day_key"] else list(cfg["met_files"].keys())[0]
            parsed = {}
            for k, v in row.items():
                if k == cfg["day_key"]:
                    continue
                parsed[k] = float(v) if v else float("nan")
            days[day].append(parsed)
    return dict(days)


def _candidate_pixels(dsm_path: str, dem_path: str, step: int = 3) -> list[tuple[int, int]]:
    """Return ground-level pixels (height < 1m) sampled every `step` pixels."""
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
    with rasterio.open(dem_path) as src:
        dem = src.read(1)

    height = dsm - dem
    rows, cols = dsm.shape
    border = 5
    candidates = []
    for r in range(border, rows - border, step):
        for c in range(border, cols - border, step):
            if height[r, c] < 1.0:
                candidates.append((r, c))
    return candidates


def _score_pixel(model_tmrt: dict[int, float], obs: list[dict], hour_key: str) -> dict:
    """Compute RMSE / MAE / bias / R² for a single pixel."""
    matched = []
    for o in obs:
        h = int(o[hour_key])
        if h in model_tmrt and not np.isnan(o["Tmrt"]) and not np.isnan(model_tmrt[h]):
            matched.append((o["Tmrt"], model_tmrt[h]))

    if len(matched) < 3:
        return {"rmse": np.inf, "mae": np.inf, "bias": np.inf, "r2": -np.inf, "n": len(matched)}

    obs_arr = np.array([m[0] for m in matched])
    mod_arr = np.array([m[1] for m in matched])
    diff = mod_arr - obs_arr

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    r2 = float(np.corrcoef(obs_arr, mod_arr)[0, 1] ** 2) if len(matched) > 2 else float("nan")

    return {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2, "n": len(matched)}


def _save_rasters(dsm_path: str, dem_path: str, agg: dict, out_dir: Path, current_poi: tuple[int, int] | None = None):
    """Save RMSE/R²/bias as PNG heatmaps with building outlines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
    with rasterio.open(dem_path) as src:
        dem = src.read(1)

    height = dsm - dem
    rows, cols = dsm.shape

    rmse_raster = np.full((rows, cols), np.nan, dtype=np.float32)
    r2_raster = np.full((rows, cols), np.nan, dtype=np.float32)
    bias_raster = np.full((rows, cols), np.nan, dtype=np.float32)

    for (r, c), stats in agg.items():
        rmse_raster[r, c] = stats["mean_rmse"]
        r2_raster[r, c] = stats["mean_r2"]
        bias_raster[r, c] = stats["mean_bias"]

    # Mask buildings
    building_mask = height > 1.0
    rmse_raster[building_mask] = np.nan
    r2_raster[building_mask] = np.nan
    bias_raster[building_mask] = np.nan

    out_dir.mkdir(exist_ok=True, parents=True)

    # Find top 5% pixels by RMSE
    ranked = sorted(agg.keys(), key=lambda k: agg[k]["mean_rmse"])
    n_top = max(1, len(ranked) // 20)
    top_rows = np.array([rc[0] for rc in ranked[:n_top]])
    top_cols = np.array([rc[1] for rc in ranked[:n_top]])

    layers = [
        ("rmse", rmse_raster, "RdYlGn_r", "RMSE (deg C)", None),
        ("r2", r2_raster, "RdYlGn", "R-squared", None),
        ("bias", bias_raster, "RdBu_r", "Bias (deg C)", "diverging"),
    ]

    for name, data, cmap, label, norm_type in layers:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Show buildings as grey background
        bg = np.full((rows, cols, 3), 1.0)  # white
        bg[building_mask] = [0.7, 0.7, 0.7]  # grey buildings
        ax.imshow(bg, aspect="equal")

        # Determine colour normalisation
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            plt.close(fig)
            continue

        if norm_type == "diverging":
            vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")
        else:
            im = ax.imshow(data, cmap=cmap, aspect="equal")

        plt.colorbar(im, ax=ax, label=label, shrink=0.8)

        # Highlight top 5% pixels with blue border
        ax.scatter(
            top_cols,
            top_rows,
            s=18,
            facecolors="none",
            edgecolors="#2563eb",
            linewidths=0.8,
            label=f"Top 5% ({n_top} pixels)",
            zorder=5,
        )

        # Mark current POI (always, even if not in candidate set)
        if current_poi:
            ax.plot(
                current_poi[1],
                current_poi[0],
                "s",
                color="black",
                markersize=10,
                markeredgewidth=2,
                markerfacecolor="none",
                label=f"Current POI {current_poi}",
            )
        ax.legend(loc="upper right", fontsize=8)

        ax.set_title(f"POI Sweep: {label}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        out_path = out_dir / f"poi_sweep_{name}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return out_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _data_present(site_name: str) -> bool:
    cfg = SITES[site_name]
    return cfg["measurements_csv"].exists() and Path(cfg["dsm"]).exists()


@pytest.mark.skipif(not _data_present("kronenhuset"), reason="Kronenhuset data not present")
def test_poi_sweep_kronenhuset(tmp_path):
    _run_sweep("kronenhuset", tmp_path)


@pytest.mark.skipif(not _data_present("gustav_adolfs"), reason="Gustav Adolfs data not present")
def test_poi_sweep_gustav_adolfs(tmp_path):
    _run_sweep("gustav_adolfs", tmp_path)


@pytest.mark.skipif(not _data_present("gvc"), reason="GVC data not present")
def test_poi_sweep_gvc(tmp_path):
    _run_sweep("gvc", tmp_path)


def _run_sweep(site_name: str, tmp_path):
    """Run POI sweep for a single site."""
    import solweig
    from conftest import read_timestep_geotiff
    from solweig import Location
    from solweig.loaders import load_params
    from solweig.models.config import HumanParams

    cfg = SITES[site_name]
    all_obs = _load_measurements(cfg)
    candidates = _candidate_pixels(cfg["dsm"], cfg["dem"], step=cfg["step"])
    day_codes = sorted(cfg["met_files"].keys())
    n_days = len(day_codes)

    print(f"\n{'=' * 90}")
    print(f"  {site_name} — POI Sensitivity Sweep")
    print(f"  {len(candidates)} candidates, {n_days} day(s)")
    print(f"{'=' * 90}")

    surface = solweig.SurfaceData.prepare(
        dsm=cfg["dsm"],
        dem=cfg["dem"],
        cdsm=cfg["cdsm"],
        land_cover=cfg["land_cover"],
        working_dir=str(tmp_path / f"{site_name}_sweep"),
    )
    location = Location(
        latitude=cfg["lat"],
        longitude=cfg["lon"],
        utc_offset=cfg["utc_offset"],
        altitude=10.0,
    )

    materials = None
    if cfg["params_json"].exists():
        materials = load_params(str(cfg["params_json"]))

    if materials is not None:
        abs_k = getattr(getattr(materials, "Tmrt_params", None), "absK", 0.70)
        abs_l = getattr(getattr(materials, "Tmrt_params", None), "absL", 0.95)
        human = HumanParams(abs_k=abs_k, abs_l=abs_l)
    else:
        human = HumanParams(abs_l=0.95)

    pixel_scores: dict[tuple[int, int], list[dict]] = {c: [] for c in candidates}

    for day_code in day_codes:
        obs = all_obs[day_code]
        met_file = cfg["met_files"][day_code]
        weather = load_site_met(met_file)

        output_dir = tmp_path / f"{site_name}_{day_code}"
        solweig.calculate(
            surface=surface,
            weather=weather,
            location=location,
            output_dir=output_dir,
            outputs=["tmrt"],
            use_anisotropic_sky=True,
            materials=materials,
            human=human,
        )

        grids = {}
        for i, w in enumerate(weather):
            grids[w.datetime.hour] = read_timestep_geotiff(output_dir, "tmrt", i)

        for r, c in candidates:
            model_tmrt = {h: float(grid[r, c]) for h, grid in grids.items()}
            score = _score_pixel(model_tmrt, obs, cfg["hour_key"])
            score["day"] = day_code
            pixel_scores[(r, c)].append(score)

    # Aggregate across days
    agg = {}
    for (r, c), day_scores in pixel_scores.items():
        valid = [s for s in day_scores if np.isfinite(s["rmse"])]
        if len(valid) < n_days:
            continue
        agg[(r, c)] = {
            "mean_rmse": np.mean([s["rmse"] for s in valid]),
            "mean_mae": np.mean([s["mae"] for s in valid]),
            "mean_bias": np.mean([s["bias"] for s in valid]),
            "mean_r2": np.mean([s["r2"] for s in valid]),
            "total_n": sum(s["n"] for s in valid),
            "day_scores": valid,
        }

    ranked = sorted(agg.items(), key=lambda x: x[1]["mean_rmse"])

    # Print top 20
    header = (
        f"{'Rank':>4s}  {'Row':>4s} {'Col':>4s}  {'RMSE':>7s} {'MAE':>7s} "
        f"{'Bias':>7s} {'R²':>6s}  {'n':>3s}  Per-day RMSE"
    )
    print(header)
    print("-" * len(header) + "-" * 30)

    current_poi = cfg["current_poi"]
    current_rank = None

    for i, ((r, c), stats) in enumerate(ranked[:20]):
        day_rmses = " ".join(f"{s['rmse']:5.1f}" for s in stats["day_scores"])
        marker = " <-- current" if (r, c) == current_poi else ""
        print(
            f"{i + 1:4d}  {r:4d} {c:4d}  "
            f"{stats['mean_rmse']:7.2f} {stats['mean_mae']:7.2f} "
            f"{stats['mean_bias']:+7.2f} {stats['mean_r2']:6.3f}  "
            f"{stats['total_n']:3d}  [{day_rmses}]{marker}"
        )

    for i, ((r, c), _) in enumerate(ranked):
        if (r, c) == current_poi:
            current_rank = i + 1
            break

    if current_rank is not None:
        stats = agg[current_poi]
        day_rmses = " ".join(f"{s['rmse']:5.1f}" for s in stats["day_scores"])
        print(
            f"\nCurrent POI {current_poi} rank: "
            f"{current_rank}/{len(ranked)} -- "
            f"RMSE={stats['mean_rmse']:.2f}, R²={stats['mean_r2']:.3f} "
            f"[{day_rmses}]"
        )
    else:
        print(f"\nCurrent POI {current_poi} not in candidate set")

    best_r, best_c = ranked[0][0]
    best = ranked[0][1]
    print(f"\nBest pixel: ({best_r}, {best_c}) -- RMSE={best['mean_rmse']:.2f}, R²={best['mean_r2']:.3f}")
    for s in best["day_scores"]:
        print(f"    {s['day']}: RMSE={s['rmse']:.2f}, Bias={s['bias']:+.2f}, R²={s['r2']:.3f}, n={s['n']}")

    # Save rasters
    out_dir = _save_rasters(cfg["dsm"], cfg["dem"], agg, cfg["out_dir"], current_poi=current_poi)
    print(f"\n  PNGs saved to {out_dir}/")
    print(f"    poi_sweep_rmse.png  -- Mean RMSE across {n_days} day(s)")
    print(f"    poi_sweep_r2.png    -- Mean R² across {n_days} day(s)")
    print("    poi_sweep_bias.png  -- Mean bias (deg C)")
