"""Compare the port with the reference implementation at the Gothenburg validation sites.

Second half of the end-to-end check. ``scripts/reference_site_run.py`` writes
the reference's hourly values at each measurement pixel; this script runs the
port on the same inputs (the setup of ``scripts/validate_ground_scheme.py``)
and prints, per site-day and with the scheme off and on:

- Tmrt RMSE and bias against the measurements, for the port and the reference;
- the largest hourly difference between the two in Tmrt and Lup.

It exits with status 1 when an hourly Tmrt difference exceeds ``--tmrt-tol``.
Small differences are expected: each side computes its own sky view factors
and wall rasters.

Usage:
    uv run python scripts/compare_with_reference.py <work dir of reference_site_run.py>
    uv run python scripts/compare_with_reference.py <work dir> --hourly
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # tests/validation imports ``tests.raster_helpers``
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

from validate_ground_scheme import _poi_series, _scheme_params  # noqa: E402
from validation.met_clock import load_site_met  # noqa: E402
from validation.test_poi_sweep_all_sites import SITES, _load_measurements, _score_pixel  # noqa: E402


def compare_site(site: str, scheme: int, reference_dir: Path, work: Path, hourly: bool) -> float:
    """Run the port at one site and return the largest hourly Tmrt difference (degC)."""
    import solweig
    from solweig import Location
    from solweig.loaders import load_params
    from solweig.models.config import HumanParams

    cfg = SITES[site]
    all_obs = _load_measurements(cfg)
    poi = cfg["current_poi"]
    surface = solweig.SurfaceData.prepare(
        dsm=cfg["dsm"],
        dem=cfg["dem"],
        cdsm=cfg["cdsm"],
        land_cover=cfg["land_cover"],
        working_dir=str(work / f"{site}_s{scheme}_work"),
    )
    location = Location(latitude=cfg["lat"], longitude=cfg["lon"], utc_offset=cfg["utc_offset"], altitude=10.0)
    base = load_params(str(cfg["params_json"]))
    materials = _scheme_params(cfg["params_json"], work)[0] if scheme else base
    tmrt_params = getattr(base, "Tmrt_params", None)
    human = HumanParams(abs_k=getattr(tmrt_params, "absK", 0.70), abs_l=getattr(tmrt_params, "absL", 0.95))
    scheme_on = bool(scheme)

    worst = 0.0
    for day in sorted(cfg["met_files"]):
        ref_file = reference_dir / f"{site}_{day}_s{scheme}" / "poi.json"
        if not ref_file.exists():
            print(f"{site} {day} scheme={scheme}: no reference run in {reference_dir}")
            continue
        ref = {
            field: {int(h): v for h, v in values.items()} for field, values in json.loads(ref_file.read_text()).items()
        }
        weather = load_site_met(cfg["met_files"][day])
        out = work / f"{site}_{day}_s{scheme}"
        solweig.calculate(
            surface=surface,
            weather=weather,
            location=location,
            output_dir=out,
            outputs=["tmrt", "lup"],
            use_anisotropic_sky=True,
            materials=materials,
            human=human,
            use_ground_scheme=scheme_on,
            use_outgoing_longwave=scheme_on,
        )
        tmrt = _poi_series(out, "tmrt", weather, poi)
        lup = _poi_series(out, "lup", weather, poi)
        obs = all_obs[day]
        port_score = _score_pixel(tmrt, obs, cfg["hour_key"])
        ref_score = _score_pixel(ref["Tmrt"], obs, cfg["hour_key"])
        # The shifted met file starts at 01:00, so hour 0 is the last timestep.
        hours = [h for h in sorted(set(tmrt) & set(ref["Tmrt"])) if h != 0] + [0]
        d_tmrt = np.array([tmrt[h] - ref["Tmrt"][h] for h in hours])
        d_lup = np.array([lup[h] - ref["Lup"][h] for h in hours])
        worst = max(worst, float(np.abs(d_tmrt).max()))
        print(
            f"{site:14s} {day} scheme={scheme}  port {port_score['rmse']:5.1f} / {port_score['bias']:+5.1f}"
            f"   reference {ref_score['rmse']:5.1f} / {ref_score['bias']:+5.1f}"
            f"   max |dTmrt| {np.abs(d_tmrt).max():4.2f}  max |dLup| {np.abs(d_lup).max():4.1f}"
        )
        if hourly:
            print("    hour   " + " ".join(f"{h:5d}" for h in hours))
            print("    dTmrt  " + " ".join(f"{v:5.2f}" for v in d_tmrt))
            print("    dLup   " + " ".join(f"{v:5.1f}" for v in d_lup))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the port with the reference implementation at the Gothenburg validation sites."
    )
    parser.add_argument("reference_dir", type=Path, help="work folder written by scripts/reference_site_run.py")
    parser.add_argument("--site", choices=sorted(SITES), action="append", help="default: all three")
    parser.add_argument("--scheme", type=int, choices=(0, 1), action="append", help="default: both")
    parser.add_argument("--hourly", action="store_true", help="print the hourly differences in run order")
    parser.add_argument("--tmrt-tol", type=float, default=1.0, help="largest accepted hourly Tmrt difference (degC)")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    print("Tmrt against measurements as RMSE / bias (degC); port minus reference per hour (degC, W/m2)")
    worst = 0.0
    with tempfile.TemporaryDirectory(prefix="solweig_reference_compare_") as tmp:
        for site in args.site or list(SITES):
            for scheme in args.scheme or (0, 1):
                worst = max(worst, compare_site(site, scheme, args.reference_dir, Path(tmp), args.hourly))
    print(f"\nlargest hourly Tmrt difference: {worst:.2f} degC (tolerance {args.tmrt_tol})")
    if worst > args.tmrt_tol:
        sys.exit(1)


if __name__ == "__main__":
    main()
