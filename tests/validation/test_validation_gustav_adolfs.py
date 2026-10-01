"""
Field-data validation tests using the Gustav Adolfs torg radiation dataset.

Dataset: Lindberg et al. (2008) - radiation budget measurements
Location: Gustav Adolfs torg, central Gothenburg, Sweden (57.7°N, 12.0°E)
Dates: 2005-10-11 (autumn), 2006-07-26 and 2006-08-01 (summer)
CRS: EPSG:3006 (SWEREF99 12 00), 2m pixel resolution, 116×104 grid
Measurements: K↓, K↑, Kn/Ke/Ks/Kw, L↓, L↑, Ln/Le/Ls/Lw, Sstr, Tmrt, Ta, RH

This is the second Gothenburg validation site from Lindberg et al. (2008),
providing multi-day (autumn + summer) validation of the radiation budget
and Tmrt at an open city square.

Geodata (DSM, DEM, CDSM, landcover, walls) and met files are in
tests/validation/gustav_adolfs/. SVFs are computed by SurfaceData.prepare().

Reference:
    Lindberg, F., Holmer, B., & Thorsson, S. (2008). SOLWEIG 1.0 — Modelling
    spatial variations of 3D radiant fluxes and mean radiant temperature in
    complex urban settings. International Journal of Biometeorology, 52, 697–713.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from validation.met_clock import load_site_met

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

GA_DIR = Path(__file__).parent / "gustav_adolfs"
MEASUREMENTS_CSV = GA_DIR / "measurements_ga.csv"
GA_PARAMS_JSON = GA_DIR / "parametersforsolweig_GA.json"
POI_GEOJSON = GA_DIR / "poi.geojson"

# Site location
LAT, LON, UTC_OFFSET = 57.7, 12.0, 1

# Met files keyed by measurement day code
MET_FILES = {
    "20051011": GA_DIR / "MetFile_20051011.txt",
    "20060726": GA_DIR / "MetFile_20060726.txt",
    "20060801": GA_DIR / "MetFile_20060801.txt",
}

pytestmark = [
    pytest.mark.skipif(not MEASUREMENTS_CSV.exists(), reason="GA measurement CSV not present"),
    pytest.mark.validation,
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_ga_measurements() -> dict[str, list[dict]]:
    """Load Gustav Adolfs torg radiation measurements from CSV.

    Returns dict mapping day code (e.g. '20051011') to list of dicts with keys:
        hour, Kdown, Kup, Kn, Ke, Ks, Kw,
        Ldown, Lup, Ln, Le, Ls, Lw, Sstr, Tmrt, Ta, RH
    """
    from collections import defaultdict

    days: dict[str, list[dict]] = defaultdict(list)
    with open(MEASUREMENTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = row["day"]
            parsed = {"day": day}
            for k, v in row.items():
                if k == "day":
                    continue
                parsed[k] = float(v) if v else float("nan")
            days[day].append(parsed)
    return dict(days)


# ---------------------------------------------------------------------------
# Test: Data loading and sanity checks
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Verify that the measurement data loads correctly."""

    def test_csv_loads(self):
        obs = load_ga_measurements()
        assert len(obs) == 3, f"Expected 3 days, got {len(obs)}"
        assert "20051011" in obs
        assert "20060726" in obs
        assert "20060801" in obs
        assert len(obs["20051011"]) == 12
        assert len(obs["20060726"]) == 16
        assert len(obs["20060801"]) == 16

    def test_radiation_physical_range(self):
        """Shortwave and longwave should be in physical ranges."""
        obs = load_ga_measurements()
        for day, rows in obs.items():
            for o in rows:
                h = int(o["hour"])
                # Longwave downwelling: 250–450 W/m²
                assert 250 < o["Ldown"] < 450, f"Ldown={o['Ldown']:.0f} out of range at {day} hour {h}"
                # Longwave upwelling: 300–570 W/m² (hot summer surfaces can be high)
                assert 300 < o["Lup"] < 570, f"Lup={o['Lup']:.0f} out of range at {day} hour {h}"

    def test_tmrt_physical_range(self):
        """Tmrt should be physical for Gothenburg."""
        obs = load_ga_measurements()
        for day, rows in obs.items():
            for o in rows:
                if np.isnan(o["Tmrt"]):
                    continue
                # Summer days can have high Tmrt; autumn is cooler
                assert -5 < o["Tmrt"] < 70, f"Tmrt={o['Tmrt']:.1f}°C out of range at {day} hour {int(o['hour'])}"


# ---------------------------------------------------------------------------
# Test: Full SOLWEIG pipeline validation against field measurements
# ---------------------------------------------------------------------------


_geodata_present = GA_DIR.exists() and (GA_DIR / "DSM_GA.tif").exists()


class TestFullPipelineValidation:
    """Run the full SOLWEIG pipeline on the Gustav Adolfs torg site and
    validate radiation outputs and Tmrt against the field measurements.

    Tests each day separately and reports aggregate statistics.
    """

    @pytest.fixture
    def poi(self):
        """POI (row, col) from the measurement station GeoJSON."""
        from conftest import poi_from_geojson

        return poi_from_geojson(POI_GEOJSON, GA_DIR / "DSM_GA.tif")

    @pytest.fixture
    def surface(self, tmp_path):
        """Load SurfaceData from the GA rasters."""
        import solweig

        surface = solweig.SurfaceData.prepare(
            dsm=str(GA_DIR / "DSM_GA.tif"),
            dem=str(GA_DIR / "DEM_GA.tif"),
            cdsm=str(GA_DIR / "CDSM_GA.tif"),
            land_cover=str(GA_DIR / "LC_GA.tif"),
            working_dir=str(tmp_path / "ga_work"),
        )
        return surface

    @pytest.fixture
    def location(self):
        from solweig import Location

        return Location(latitude=LAT, longitude=LON, utc_offset=UTC_OFFSET, altitude=10.0)

    @pytest.fixture
    def ga_materials(self):
        """Load the GA-specific materials/parameters JSON."""
        from solweig.loaders import load_params

        if GA_PARAMS_JSON.exists():
            return load_params(str(GA_PARAMS_JSON))
        return None

    @pytest.fixture
    def ga_human(self, ga_materials):
        """Human params matching the GA config (absL=0.95, absK=0.70)."""
        from solweig.models.config import HumanParams

        if ga_materials is not None:
            abs_k = getattr(getattr(ga_materials, "Tmrt_params", None), "absK", 0.70)
            abs_l = getattr(getattr(ga_materials, "Tmrt_params", None), "absL", 0.95)
            return HumanParams(abs_k=abs_k, abs_l=abs_l)
        return HumanParams(abs_l=0.95)

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="GA geodata not present")
    @pytest.mark.parametrize("day_code", ["20051011", "20060726", "20060801"])
    def test_tmrt_vs_observations(self, poi, surface, location, ga_materials, ga_human, tmp_path, day_code):
        """Compare SOLWEIG Tmrt at POI against measured Tmrt for a single day.

        Reports RMSE, MAE, Bias, R² for both isotropic and anisotropic sky modes.
        """
        import solweig
        from conftest import read_timestep_geotiff

        all_obs = load_ga_measurements()
        obs = all_obs[day_code]
        met_file = MET_FILES[day_code]
        weather = load_site_met(met_file)

        results_by_mode: dict[str, dict] = {}
        for mode_name, aniso in [("Isotropic", False), ("Anisotropic", True)]:
            output_dir = tmp_path / f"tmrt_{day_code}_{mode_name.lower()}"
            solweig.calculate(
                surface=surface,
                weather=weather,
                location=location,
                output_dir=output_dir,
                outputs=["tmrt"],
                use_anisotropic_sky=aniso,
                materials=ga_materials,
                human=ga_human,
            )

            model_tmrt = {}
            for i, w in enumerate(weather):
                tmrt = read_timestep_geotiff(output_dir, "tmrt", i)
                model_tmrt[w.datetime.hour] = tmrt[poi[0], poi[1]]

            matched = []
            for o in obs:
                h = int(o["hour"])
                if h in model_tmrt and not np.isnan(o["Tmrt"]):
                    matched.append(
                        {
                            "hour": h,
                            "ta": o["Ta"],
                            "obs_tmrt": o["Tmrt"],
                            "mod_tmrt": model_tmrt[h],
                        }
                    )

            obs_arr = np.array([m["obs_tmrt"] for m in matched])
            mod_arr = np.array([m["mod_tmrt"] for m in matched])
            diff_arr = mod_arr - obs_arr

            results_by_mode[mode_name] = {
                "matched": matched,
                "obs": obs_arr,
                "mod": mod_arr,
                "rmse": float(np.sqrt(np.mean(diff_arr**2))),
                "mae": float(np.mean(np.abs(diff_arr))),
                "bias": float(np.mean(diff_arr)),
                "r2": float(np.corrcoef(obs_arr, mod_arr)[0, 1] ** 2) if len(obs_arr) > 2 else float("nan"),
            }

        iso = results_by_mode["Isotropic"]
        ani = results_by_mode["Anisotropic"]

        print(f"\n{'=' * 80}")
        print(f"Gustav Adolfs torg Tmrt Validation — Day {day_code}")
        print(f"{'=' * 80}")
        header = f"{'Hour':>4s} {'Ta':>5s} {'Obs':>6s}  {'Iso':>6s} {'Diff':>6s}  {'Aniso':>6s} {'Diff':>6s}"
        print(header)
        print(f"{'-' * len(header)}")
        for mi, ma in zip(iso["matched"], ani["matched"], strict=True):
            h = mi["hour"]
            di = mi["mod_tmrt"] - mi["obs_tmrt"]
            da = ma["mod_tmrt"] - ma["obs_tmrt"]
            print(
                f"{h:4d} {mi['ta']:5.1f} {mi['obs_tmrt']:6.1f}  "
                f"{mi['mod_tmrt']:6.1f} {di:+6.1f}  "
                f"{ma['mod_tmrt']:6.1f} {da:+6.1f}"
            )

        print(f"\n{'Metric':>10s} {'Iso':>10s} {'Aniso':>12s}")
        print(f"{'-' * 36}")
        print(f"{'RMSE':>10s} {iso['rmse']:9.2f}° {ani['rmse']:11.2f}°")
        print(f"{'MAE':>10s} {iso['mae']:9.2f}° {ani['mae']:11.2f}°")
        print(f"{'Bias':>10s} {iso['bias']:+9.2f}° {ani['bias']:+11.2f}°")
        print(f"{'R²':>10s} {iso['r2']:9.3f}  {ani['r2']:11.3f}")

        assert ani["rmse"] < 20.0, f"Tmrt RMSE={ani['rmse']:.2f}°C exceeds 20°C threshold for day {day_code}"

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="GA geodata not present")
    @pytest.mark.parametrize("day_code", ["20051011", "20060726", "20060801"])
    def test_radiation_budget_vs_observations(self, poi, surface, location, ga_materials, ga_human, tmp_path, day_code):
        """Compare SOLWEIG radiation outputs at POI against measured fluxes.

        Validates K↓, K↑, L↓, L↑ for each day.
        """
        import solweig
        from conftest import read_timestep_geotiff

        all_obs = load_ga_measurements()
        obs = all_obs[day_code]
        met_file = MET_FILES[day_code]
        weather = load_site_met(met_file)

        components = ["Kdown", "Kup", "Ldown", "Lup"]
        model_keys = ["kdown", "kup", "ldown", "lup"]

        results_by_mode: dict[str, dict] = {}
        for mode_name, aniso in [("Isotropic", False), ("Anisotropic", True)]:
            output_dir = tmp_path / f"rad_{day_code}_{mode_name.lower()}"
            solweig.calculate(
                surface=surface,
                weather=weather,
                location=location,
                output_dir=output_dir,
                outputs=["tmrt", "kdown", "kup", "ldown", "lup"],
                use_anisotropic_sky=aniso,
                materials=ga_materials,
                human=ga_human,
            )

            model_rad = {}
            for i, w in enumerate(weather):
                model_rad[w.datetime.hour] = {
                    mk: read_timestep_geotiff(output_dir, mk, i)[poi[0], poi[1]] for mk in model_keys
                }

            errors = {c: [] for c in components}
            matched = []
            for o in obs:
                h = int(o["hour"])
                if h not in model_rad:
                    continue
                mr = model_rad[h]
                matched.append({"hour": h, "obs": o, "mod": mr})
                for comp, mk in zip(components, model_keys, strict=True):
                    errors[comp].append(mr[mk] - o[comp])

            stats = {}
            for comp in components:
                err = np.array(errors[comp])
                stats[comp] = {
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "mae": float(np.mean(np.abs(err))),
                    "bias": float(np.mean(err)),
                }

            results_by_mode[mode_name] = {
                "matched": matched,
                "stats": stats,
            }

        iso = results_by_mode["Isotropic"]
        ani = results_by_mode["Anisotropic"]
        assert len(ani["matched"]) >= 5, f"Only {len(ani['matched'])} matched hours for day {day_code}"

        # Print summary
        for comp, mk in zip(components, model_keys, strict=True):
            print(f"\n{'=' * 60}")
            print(f"  {comp} (day {day_code}): Observed vs Iso vs Aniso (W/m²)")
            print(f"{'=' * 60}")
            hdr = f"{'Hour':>4s} {'Obs':>7s}  {'Iso':>7s} {'Diff':>6s}  {'Aniso':>7s} {'Diff':>6s}"
            print(hdr)
            print(f"{'-' * len(hdr)}")
            for mi, ma in zip(iso["matched"], ani["matched"], strict=True):
                h = mi["hour"]
                o_val = mi["obs"][comp]
                i_val = mi["mod"][mk]
                a_val = ma["mod"][mk]
                print(f"{h:4d} {o_val:7.1f}  {i_val:7.1f} {i_val - o_val:+6.0f}  {a_val:7.1f} {a_val - o_val:+6.0f}")

        print(f"\n{'=' * 60}")
        print(f"  Summary (day {day_code}): RMSE / MAE / Bias  (W/m²)")
        print(f"{'=' * 60}")
        print(f"{'Component':>10s}  {'--- Isotropic ---':>20s}  {'--- Anisotropic ---':>20s}")
        print(f"{'':>10s}  {'RMSE':>6s} {'MAE':>6s} {'Bias':>7s}  {'RMSE':>6s} {'MAE':>6s} {'Bias':>7s}")
        print(f"{'-' * 60}")
        for comp in components:
            si = iso["stats"][comp]
            sa = ani["stats"][comp]
            print(
                f"{comp:>10s}  {si['rmse']:6.1f} {si['mae']:6.1f} {si['bias']:+7.1f}  "
                f"{sa['rmse']:6.1f} {sa['mae']:6.1f} {sa['bias']:+7.1f}"
            )

        # Kdown has high RMSE at open sites due to shadow/sunlit sensitivity
        kdown_rmse = ani["stats"]["Kdown"]["rmse"]
        assert kdown_rmse < 350.0, f"Kdown RMSE={kdown_rmse:.1f} W/m² exceeds 350 threshold for day {day_code}"

        ldown_rmse = ani["stats"]["Ldown"]["rmse"]
        assert ldown_rmse < 85.0, f"Ldown RMSE={ldown_rmse:.1f} W/m² exceeds 85 threshold for day {day_code}"
