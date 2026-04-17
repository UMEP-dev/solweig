"""
Per-site timeseries discrepancy plots for the three Gothenburg validation sites.

Generates a single PNG per site that stitches together every measurement day
into one continuous timeline (with day-boundary separators), and stacks five
panels — Tmrt, K↓, K↑, L↓, L↑ — showing observed vs modelled values at the
POI under the anisotropic sky model. The Tmrt panel additionally shades hours
where the modelled shadow fraction at the POI is below 0.5, so shade-timing
mismatches that drive abrupt Tmrt divergences become visible.

Per-panel annotations report RMSE / Bias across the full stitched series.

Outputs:
    tests/validation/<site>/timeseries_plots/timeseries.png

Usage:
    pytest tests/validation/test_timeseries_plots.py -v -s
    pytest tests/validation/test_timeseries_plots.py -v -s -k gvc
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

VALIDATION_DIR = Path(__file__).parent

SITES: dict[str, dict[str, Any]] = {
    "kronenhuset": {
        "label": "Kronenhuset",
        "dsm": str(VALIDATION_DIR / "kronenhuset" / "DSM_KR.tif"),
        "dem": str(VALIDATION_DIR / "kronenhuset" / "DEM_KR.tif"),
        "cdsm": str(VALIDATION_DIR / "kronenhuset" / "CDSM_KR.asc"),
        "land_cover": str(VALIDATION_DIR / "kronenhuset" / "landcover_KR.tif"),
        "measurements_csv": VALIDATION_DIR / "kronenhuset" / "measurements_kr.csv",
        "params_json": VALIDATION_DIR / "kronenhuset" / "parametersforsolweig_KR.json",
        "poi_geojson": VALIDATION_DIR / "kronenhuset" / "poi.geojson",
        "met_files": {
            "20051007": VALIDATION_DIR / "kronenhuset" / "MetFile_Prepared.txt",
        },
        "day_key": None,
        "out_dir": VALIDATION_DIR / "kronenhuset" / "timeseries_plots",
    },
    "gustav_adolfs": {
        "label": "Gustav Adolfs torg",
        "dsm": str(VALIDATION_DIR / "gustav_adolfs" / "DSM_GA.tif"),
        "dem": str(VALIDATION_DIR / "gustav_adolfs" / "DEM_GA.tif"),
        "cdsm": str(VALIDATION_DIR / "gustav_adolfs" / "CDSM_GA.tif"),
        "land_cover": str(VALIDATION_DIR / "gustav_adolfs" / "LC_GA.tif"),
        "measurements_csv": VALIDATION_DIR / "gustav_adolfs" / "measurements_ga.csv",
        "params_json": VALIDATION_DIR / "gustav_adolfs" / "parametersforsolweig_GA.json",
        "poi_geojson": VALIDATION_DIR / "gustav_adolfs" / "poi.geojson",
        "met_files": {
            "20051011": VALIDATION_DIR / "gustav_adolfs" / "MetFile_20051011.txt",
            "20060726": VALIDATION_DIR / "gustav_adolfs" / "MetFile_20060726.txt",
            "20060801": VALIDATION_DIR / "gustav_adolfs" / "MetFile_20060801.txt",
        },
        "day_key": "day",
        "out_dir": VALIDATION_DIR / "gustav_adolfs" / "timeseries_plots",
    },
    "gvc": {
        "label": "GVC",
        "dsm": str(VALIDATION_DIR / "gvc" / "DSM_GVC_1m.tif"),
        "dem": str(VALIDATION_DIR / "gvc" / "DEM_GVC_1m.tif"),
        "cdsm": str(VALIDATION_DIR / "gvc" / "CDSM_GVC_1m.tif"),
        "land_cover": str(VALIDATION_DIR / "gvc" / "landcover_1m_GVC.tif"),
        "measurements_csv": VALIDATION_DIR / "gvc" / "measurements_gvc.csv",
        "params_json": VALIDATION_DIR / "gvc" / "parametersforsolweig_GVC.json",
        "poi_geojson": VALIDATION_DIR / "gvc" / "poi.geojson",
        "met_files": {
            "20100707": VALIDATION_DIR / "gvc" / "MetFile20100707_Prepared.txt",
            "20100710": VALIDATION_DIR / "gvc" / "MetFile20100710_Prepared.txt",
            "20100712": VALIDATION_DIR / "gvc" / "MetFile20100712_Prepared.txt",
        },
        "day_key": "date",
        "out_dir": VALIDATION_DIR / "gvc" / "timeseries_plots",
    },
}

# Site location is the same for all three (central Gothenburg).
LAT, LON, UTC_OFFSET = 57.7, 12.0, 1

# Components: (csv key, model output key, panel label, units)
COMPONENTS: list[tuple[str, str, str, str]] = [
    ("Tmrt", "tmrt", "Tmrt", "°C"),
    ("Kdown", "kdown", "K↓", "W/m²"),
    ("Kup", "kup", "K↑", "W/m²"),
    ("Ldown", "ldown", "L↓", "W/m²"),
    ("Lup", "lup", "L↑", "W/m²"),
]

pytestmark = [
    pytest.mark.validation,
    pytest.mark.slow,
]


def _load_measurements(cfg: dict) -> dict[str, list[dict]]:
    """Load measurement CSV into ``{day_code: [row_dicts]}``."""
    days: dict[str, list[dict]] = defaultdict(list)
    with open(cfg["measurements_csv"]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row[cfg["day_key"]] if cfg["day_key"] else next(iter(cfg["met_files"].keys()))
            parsed = {}
            for k, v in row.items():
                if k == cfg["day_key"]:
                    continue
                parsed[k] = float(v) if v else float("nan")
            days[day].append(parsed)
    return dict(days)


def _stats(obs: np.ndarray, mod: np.ndarray) -> tuple[float, float]:
    """Return (RMSE, bias) over finite paired values."""
    mask = np.isfinite(obs) & np.isfinite(mod)
    if not mask.any():
        return float("nan"), float("nan")
    diff = mod[mask] - obs[mask]
    return float(np.sqrt(np.mean(diff**2))), float(np.mean(diff))


def _plot_site(site_name: str, day_records: list[dict], out_dir: Path) -> Path:
    """Draw the stitched-timeline figure for one site.

    ``day_records`` is a list of ``{day_code, hours, obs:{comp:array}, mod:{comp:array}, shadow:array}``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = SITES[site_name]
    n_panels = len(COMPONENTS)

    # Build a single stitched x-axis. Each day occupies ``len(hours)`` slots,
    # plus a 1-slot gap between consecutive days for visual separation.
    GAP = 1
    day_starts: list[int] = []
    cursor = 0
    per_day_x: list[np.ndarray] = []
    for rec in day_records:
        n = len(rec["hours"])
        per_day_x.append(np.arange(cursor, cursor + n))
        day_starts.append(cursor)
        cursor += n + GAP
    total_len = cursor

    # Concatenate per-component obs/mod for global stats.
    all_obs = {comp: [] for comp, *_ in COMPONENTS}
    all_mod = {comp: [] for comp, *_ in COMPONENTS}
    for rec in day_records:
        for comp, *_ in COMPONENTS:
            all_obs[comp].append(rec["obs"][comp])
            all_mod[comp].append(rec["mod"][comp])
    all_obs = {k: np.concatenate(v) for k, v in all_obs.items()}
    all_mod = {k: np.concatenate(v) for k, v in all_mod.items()}

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(11, 2.0 * n_panels + 0.5),
        sharex=True,
        constrained_layout=True,
    )

    for ax_i, (comp_csv, _comp_mod, label, units) in enumerate(COMPONENTS):
        ax = axes[ax_i]

        # Per-day lines so the gap renders cleanly (no false interpolation).
        for x, rec in zip(per_day_x, day_records, strict=True):
            obs = rec["obs"][comp_csv]
            mod = rec["mod"][comp_csv]
            ax.plot(x, obs, color="#111111", lw=1.4, label="Observed" if x is per_day_x[0] else None)
            ax.plot(x, mod, color="#d62728", lw=1.4, ls="--", label="Modelled" if x is per_day_x[0] else None)

        # Shade panel: mark hours where POI is in modelled shade.
        if comp_csv == "Tmrt":
            for x, rec in zip(per_day_x, day_records, strict=True):
                shadow = rec["shadow"]
                in_shade = shadow < 0.5
                # Draw a half-hour-wide axvspan at each shaded slot.
                for xi, shaded in zip(x, in_shade, strict=True):
                    if shaded:
                        ax.axvspan(xi - 0.5, xi + 0.5, color="#cccccc", alpha=0.45, lw=0)

        # Day separators and labels (labels sit inside top panel to avoid
        # colliding with suptitle).
        for start, rec in zip(day_starts, day_records, strict=True):
            if start > 0:
                ax.axvline(start - GAP / 2.0, color="#888888", lw=0.6, ls=":")
            if ax_i == 0:
                ax.text(
                    start + len(rec["hours"]) / 2.0 - 0.5,
                    0.03,
                    rec["day_code"],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#444444",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                )

        rmse, bias = _stats(all_obs[comp_csv], all_mod[comp_csv])
        ax.text(
            0.005,
            0.95,
            f"{label}   RMSE={rmse:.1f} {units}   Bias={bias:+.1f} {units}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
        ax.set_ylabel(f"{label} ({units})", fontsize=9)
        ax.grid(True, axis="y", lw=0.4, color="#dddddd")
        ax.tick_params(labelsize=8)

        if ax_i == 0:
            handles = [
                plt.Line2D([], [], color="#111111", lw=1.4, label="Observed"),
                plt.Line2D([], [], color="#d62728", lw=1.4, ls="--", label="Modelled"),
                plt.Rectangle((0, 0), 1, 1, color="#cccccc", alpha=0.45, label="POI in shade"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    # Hour ticks on the bottom panel.
    tick_pos = []
    tick_lab = []
    for x, rec in zip(per_day_x, day_records, strict=True):
        for xi, h in zip(x, rec["hours"], strict=True):
            tick_pos.append(xi)
            tick_lab.append(str(int(h)))
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_lab, fontsize=7)
    axes[-1].set_xlabel("Hour of day (UTC+1)", fontsize=9)
    axes[-1].set_xlim(-0.5, total_len - GAP - 0.5)

    fig.suptitle(
        f"{cfg['label']} — POI timeseries (anisotropic sky)",
        fontsize=11,
        y=1.005,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timeseries.png"
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _data_present(site_name: str) -> bool:
    cfg = SITES[site_name]
    return cfg["measurements_csv"].exists() and Path(cfg["dsm"]).exists()


@pytest.mark.skipif(not _data_present("kronenhuset"), reason="Kronenhuset data not present")
def test_timeseries_kronenhuset(tmp_path):
    _run_site("kronenhuset", tmp_path)


@pytest.mark.skipif(not _data_present("gustav_adolfs"), reason="Gustav Adolfs data not present")
def test_timeseries_gustav_adolfs(tmp_path):
    _run_site("gustav_adolfs", tmp_path)


@pytest.mark.skipif(not _data_present("gvc"), reason="GVC data not present")
def test_timeseries_gvc(tmp_path):
    _run_site("gvc", tmp_path)


def _run_site(site_name: str, tmp_path):
    """Run the full pipeline for every day of one site, gather POI timeseries,
    then plot."""
    import solweig
    from conftest import poi_from_geojson, read_timestep_geotiff
    from solweig import Location
    from solweig.loaders import load_params
    from solweig.models.config import HumanParams

    cfg = SITES[site_name]
    poi = poi_from_geojson(cfg["poi_geojson"], cfg["dsm"])
    all_obs = _load_measurements(cfg)
    day_codes = sorted(cfg["met_files"].keys())

    print(f"\n{'=' * 80}\n  {cfg['label']} — timeseries plot ({len(day_codes)} day(s), POI={poi})\n{'=' * 80}")

    surface = solweig.SurfaceData.prepare(
        dsm=cfg["dsm"],
        dem=cfg["dem"],
        cdsm=cfg["cdsm"],
        land_cover=cfg["land_cover"],
        working_dir=str(tmp_path / f"{site_name}_work"),
    )
    location = Location(latitude=LAT, longitude=LON, utc_offset=UTC_OFFSET, altitude=10.0)

    materials = load_params(str(cfg["params_json"])) if cfg["params_json"].exists() else None
    if materials is not None:
        abs_k = getattr(getattr(materials, "Tmrt_params", None), "absK", 0.70)
        abs_l = getattr(getattr(materials, "Tmrt_params", None), "absL", 0.95)
        human = HumanParams(abs_k=abs_k, abs_l=abs_l)
    else:
        human = HumanParams(abs_l=0.95)

    day_records: list[dict] = []
    for day_code in day_codes:
        obs_rows = all_obs[day_code]
        met_file = cfg["met_files"][day_code]
        weather = solweig.Weather.from_umep_met(met_file, resample_hourly=False)

        output_dir = tmp_path / f"{site_name}_{day_code}"
        solweig.calculate(
            surface=surface,
            weather=weather,
            location=location,
            output_dir=output_dir,
            outputs=["tmrt", "kdown", "kup", "ldown", "lup", "shadow"],
            use_anisotropic_sky=True,
            materials=materials,
            human=human,
        )

        # POI series indexed by hour-of-day.
        model_by_hour: dict[int, dict[str, float]] = {}
        shadow_by_hour: dict[int, float] = {}
        for i, w in enumerate(weather):
            h = w.datetime.hour
            row: dict[str, float] = {}
            for _comp_csv, comp_mod, *_ in COMPONENTS:
                arr = read_timestep_geotiff(output_dir, comp_mod, i)
                row[comp_mod] = float(arr[poi[0], poi[1]])
            model_by_hour[h] = row
            try:
                shadow_arr = read_timestep_geotiff(output_dir, "shadow", i)
                shadow_by_hour[h] = float(shadow_arr[poi[0], poi[1]])
            except FileNotFoundError:
                shadow_by_hour[h] = float("nan")

        # Match observation hours to model hours.
        hours: list[int] = []
        obs_per_comp: dict[str, list[float]] = {comp: [] for comp, *_ in COMPONENTS}
        mod_per_comp: dict[str, list[float]] = {comp: [] for comp, *_ in COMPONENTS}
        shadow: list[float] = []
        for o in obs_rows:
            h = int(o["hour"])
            if h not in model_by_hour:
                continue
            hours.append(h)
            for comp_csv, comp_mod, *_ in COMPONENTS:
                obs_per_comp[comp_csv].append(o.get(comp_csv, float("nan")))
                mod_per_comp[comp_csv].append(model_by_hour[h][comp_mod])
            shadow.append(shadow_by_hour.get(h, float("nan")))

        day_records.append(
            {
                "day_code": day_code,
                "hours": np.array(hours),
                "obs": {k: np.array(v, dtype=float) for k, v in obs_per_comp.items()},
                "mod": {k: np.array(v, dtype=float) for k, v in mod_per_comp.items()},
                "shadow": np.array(shadow, dtype=float),
            }
        )

    out = _plot_site(site_name, day_records, cfg["out_dir"])
    print(f"  -> {out}")
