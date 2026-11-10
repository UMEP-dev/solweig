"""Field comparison of the opt-in UMEP 2026a ground scheme at the Gothenburg sites.

Runs the baseline model and the scheme (``use_ground_scheme`` +
``use_outgoing_longwave``, anisotropic sky) on the three validation sites and
prints POI Tmrt RMSE / bias against the measurements for each site-day, plus
the Lup bias where Lup was measured. This is the source of the scheme table in
VALIDATION.md; rerun it whenever the scheme path changes.

The site parameter files predate the scheme, so the blocks it needs (heat
capacity, thermal diffusivity, OHM, Tg_ini, Tm_ini) are filled in from the
bundled ``default_materials.json`` when a site file lacks them.

Usage:
    uv run python scripts/validate_ground_scheme.py
    uv run python scripts/validate_ground_scheme.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

from validation.met_clock import load_site_met  # noqa: E402
from validation.test_poi_sweep_all_sites import SITES, _load_measurements, _score_pixel  # noqa: E402

SCHEME_BLOCKS = (
    "Heat capacity",
    "Thermal_diffusivity",
    "OHM_coefficients",
    "Tg_ini coefficients",
    "Tm_ini coefficients",
)


def _scheme_params(site_json: Path, work_dir: Path):
    """Site parameters with the scheme blocks filled in from the bundled defaults."""
    from solweig.loaders import load_params

    defaults_path = REPO / "pysrc" / "solweig" / "data" / "default_materials.json"
    site = json.loads(site_json.read_text())
    defaults = json.loads(defaults_path.read_text())
    added = [name for name in SCHEME_BLOCKS if name not in site]
    for name in added:
        site[name] = defaults[name]
    merged = work_dir / f"{site_json.stem}_scheme.json"
    merged.write_text(json.dumps(site, indent=2))
    return load_params(str(merged)), added


def _poi_series(output_dir: Path, field: str, weather, poi) -> dict[int, float]:
    from solweig.io import load_raster

    tifs = sorted((output_dir / field).glob("*.tif"))
    series = {}
    for w, tif in zip(weather, tifs, strict=True):
        arr, _, _, _ = load_raster(str(tif))
        series[w.datetime.hour] = float(arr[poi[0], poi[1]])
    return series


def _lup_bias(model: dict[int, float], obs: list[dict], hour_key: str) -> float | None:
    pairs = [
        (model[int(o[hour_key])], o["Lup"])
        for o in obs
        if "Lup" in o and int(o[hour_key]) in model and np.isfinite(o["Lup"])
    ]
    if len(pairs) < 3:
        return None
    return float(np.mean([m - o for m, o in pairs]))


def run_site(site_name: str, work_dir: Path) -> list[dict]:
    import solweig
    from solweig import Location
    from solweig.loaders import load_params
    from solweig.models.config import HumanParams

    cfg = SITES[site_name]
    all_obs = _load_measurements(cfg)
    surface = solweig.SurfaceData.prepare(
        dsm=cfg["dsm"],
        dem=cfg["dem"],
        cdsm=cfg["cdsm"],
        land_cover=cfg["land_cover"],
        working_dir=str(work_dir / f"{site_name}_work"),
    )
    location = Location(latitude=cfg["lat"], longitude=cfg["lon"], utc_offset=cfg["utc_offset"], altitude=10.0)
    base_materials = load_params(str(cfg["params_json"]))
    scheme_materials, added = _scheme_params(cfg["params_json"], work_dir)
    tmrt_params = getattr(base_materials, "Tmrt_params", None)
    human = HumanParams(abs_k=getattr(tmrt_params, "absK", 0.70), abs_l=getattr(tmrt_params, "absL", 0.95))
    poi = cfg["current_poi"]

    rows = []
    for day_code in sorted(cfg["met_files"]):
        obs = all_obs[day_code]
        row: dict = {"site": site_name, "day": day_code, "blocks_from_defaults": added}
        variants: list[tuple[str, Any, dict[str, Any]]] = [
            ("baseline", base_materials, {}),
            ("scheme", scheme_materials, {"use_ground_scheme": True, "use_outgoing_longwave": True}),
        ]
        for label, materials, flags in variants:
            weather = load_site_met(cfg["met_files"][day_code])
            out = work_dir / f"{site_name}_{day_code}_{label}"
            solweig.calculate(
                surface=surface,
                weather=weather,
                location=location,
                output_dir=out,
                outputs=["tmrt", "lup"],
                use_anisotropic_sky=True,
                materials=materials,
                human=human,
                **flags,
            )
            score = _score_pixel(_poi_series(out, "tmrt", weather, poi), obs, cfg["hour_key"])
            row[label] = {
                "rmse": round(score["rmse"], 3),
                "bias": round(score["bias"], 3),
                "r2": round(score["r2"], 3),
                "n": score["n"],
                "lup_bias": _lup_bias(_poi_series(out, "lup", weather, poi), obs, cfg["hour_key"]),
            }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Field comparison of the opt-in 2026a ground scheme at the Gothenburg sites."
    )
    parser.add_argument("--json", type=Path, help="also write the results to this JSON file")
    args = parser.parse_args()

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="solweig_scheme_validation_") as tmp:
        for site_name in SITES:
            rows.extend(run_site(site_name, Path(tmp)))

    print("\nPOI Tmrt against measurements, RMSE / bias (degC); Lup bias (W/m2)")
    print(f"{'site':14s} {'day':9s} {'baseline':>14s} {'scheme':>14s} {'Lup base':>9s} {'Lup scheme':>10s}")
    for row in rows:
        base, scheme = row["baseline"], row["scheme"]

        def lup(v):
            return "n/a" if v is None else f"{v:+.1f}"

        print(
            f"{row['site']:14s} {row['day']:9s} "
            f"{base['rmse']:6.1f} / {base['bias']:+5.1f} {scheme['rmse']:6.1f} / {scheme['bias']:+5.1f} "
            f"{lup(base['lup_bias']):>9s} {lup(scheme['lup_bias']):>10s}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
