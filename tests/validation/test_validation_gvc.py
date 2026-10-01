"""
Field-data validation tests using the Geovetenskapens center (GVC) radiation dataset.

Dataset: Lindberg & Grimmond (2011) - radiation measurements at GVC square
Location: Geovetenskap center, University of Gothenburg, Sweden (57.7°N, 12.0°E)
Dates: 2010-07-07, 2010-07-10, 2010-07-12 (summer, Site 1)
CRS: EPSG:3006 (SWEREF99 12 00), 2m pixel resolution, 180×180 grid
Measurements: K↓, K↑, Kn/Ke/Ks/Kw, L↓, L↑, Ln/Le/Ls/Lw, Sstr, Tmrt

This is the third Gothenburg validation site, providing summer-only
multi-day validation. The GVC square has more vegetation than the other
sites, testing the canopy radiation scheme.

Geodata (DSM, DEM, CDSM, landcover, walls) and met files are in
tests/validation/gvc/. SVFs are computed by SurfaceData.prepare().

Reference:
    Lindberg, F. & Grimmond, C.S.B. (2011). The influence of vegetation and
    building morphology on shadow patterns and mean radiant temperature in
    urban areas: model development and evaluation. Theoretical and Applied
    Climatology, 105, 311–323.
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

GVC_DIR = Path(__file__).parent / "gvc"
MEASUREMENTS_CSV = GVC_DIR / "measurements_gvc.csv"
POI_GEOJSON = GVC_DIR / "poi.geojson"

# Site location
LAT, LON, UTC_OFFSET = 57.7, 12.0, 1

# Met files keyed by measurement day code
MET_FILES = {
    "20100707": GVC_DIR / "MetFile20100707_Prepared.txt",
    "20100710": GVC_DIR / "MetFile20100710_Prepared.txt",
    "20100712": GVC_DIR / "MetFile20100712_Prepared.txt",
}

pytestmark = [
    pytest.mark.skipif(not MEASUREMENTS_CSV.exists(), reason="GVC measurement CSV not present"),
    pytest.mark.validation,
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_gvc_measurements() -> dict[str, list[dict]]:
    """Load GVC radiation measurements from CSV.

    Returns dict mapping day code (e.g. '20100707') to list of dicts with keys:
        hour, Kdown, Kup, Ks, Kn, Kw, Ke,
        Ldown, Lup, Ls, Ln, Lw, Le, Sstr, Tmrt
    """
    from collections import defaultdict

    days: dict[str, list[dict]] = defaultdict(list)
    with open(MEASUREMENTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"]
            parsed = {"date": date}
            for k, v in row.items():
                if k == "date":
                    continue
                parsed[k] = float(v) if v else float("nan")
            days[date].append(parsed)
    return dict(days)


# ---------------------------------------------------------------------------
# Test: Data loading and sanity checks
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Verify that the measurement data loads correctly."""

    def test_csv_loads(self):
        obs = load_gvc_measurements()
        assert len(obs) == 3, f"Expected 3 days, got {len(obs)}"
        assert "20100707" in obs
        assert "20100710" in obs
        assert "20100712" in obs

    def test_radiation_physical_range(self):
        """Shortwave and longwave should be in physical ranges."""
        obs = load_gvc_measurements()
        for day, rows in obs.items():
            for o in rows:
                h = int(o["hour"])
                assert 250 < o["Ldown"] < 450, f"Ldown={o['Ldown']:.0f} out of range at {day} hour {h}"
                assert 300 < o["Lup"] < 500, f"Lup={o['Lup']:.0f} out of range at {day} hour {h}"

    def test_tmrt_physical_range(self):
        """Tmrt should be physical for July in Gothenburg."""
        obs = load_gvc_measurements()
        for day, rows in obs.items():
            for o in rows:
                assert 5 < o["Tmrt"] < 75, f"Tmrt={o['Tmrt']:.1f}°C out of range at {day} hour {int(o['hour'])}"


# ---------------------------------------------------------------------------
# Test: Full SOLWEIG pipeline validation against field measurements
# ---------------------------------------------------------------------------


_geodata_present = GVC_DIR.exists() and (GVC_DIR / "DSM_GVC_1m.tif").exists()


class TestFullPipelineValidation:
    """Run the full SOLWEIG pipeline on the GVC site and validate
    radiation outputs and Tmrt against the 2010 field measurements.
    """

    @pytest.fixture
    def poi(self):
        """POI (row, col) from the Site 1 measurement station GeoJSON."""
        from conftest import poi_from_geojson

        return poi_from_geojson(POI_GEOJSON, GVC_DIR / "DSM_GVC_1m.tif")

    @pytest.fixture
    def surface(self, tmp_path):
        """Load SurfaceData from the GVC rasters."""
        import solweig

        surface = solweig.SurfaceData.prepare(
            dsm=str(GVC_DIR / "DSM_GVC_1m.tif"),
            dem=str(GVC_DIR / "DEM_GVC_1m.tif"),
            cdsm=str(GVC_DIR / "CDSM_GVC_1m.tif"),
            land_cover=str(GVC_DIR / "landcover_1m_GVC.tif"),
            working_dir=str(tmp_path / "gvc_work"),
        )
        return surface

    @pytest.fixture
    def location(self):
        from solweig import Location

        return Location(latitude=LAT, longitude=LON, utc_offset=UTC_OFFSET, altitude=10.0)

    @pytest.fixture
    def gvc_materials(self):
        """Load GVC-specific materials/parameters JSON."""
        from solweig.loaders import load_params

        params_json = GVC_DIR / "parametersforsolweig_GVC.json"
        if params_json.exists():
            return load_params(str(params_json))
        return None

    @pytest.fixture
    def gvc_human(self, gvc_materials):
        """Human params (absL=0.95, absK=0.70)."""
        from solweig.models.config import HumanParams

        if gvc_materials is not None:
            abs_k = getattr(getattr(gvc_materials, "Tmrt_params", None), "absK", 0.70)
            abs_l = getattr(getattr(gvc_materials, "Tmrt_params", None), "absL", 0.95)
            return HumanParams(abs_k=abs_k, abs_l=abs_l)
        return HumanParams(abs_l=0.95)

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="GVC geodata not present")
    @pytest.mark.parametrize("day_code", ["20100707", "20100710", "20100712"])
    def test_tmrt_vs_observations(self, poi, surface, location, gvc_materials, gvc_human, tmp_path, day_code):
        """Compare SOLWEIG Tmrt at POI against measured Tmrt for a single day."""
        import solweig
        from conftest import read_timestep_geotiff

        all_obs = load_gvc_measurements()
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
                materials=gvc_materials,
                human=gvc_human,
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
        print(f"GVC Tmrt Validation — Day {day_code} (Lindberg & Grimmond 2011)")
        print(f"{'=' * 80}")
        header = f"{'Hour':>4s} {'Obs':>6s}  {'Iso':>6s} {'Diff':>6s}  {'Aniso':>6s} {'Diff':>6s}"
        print(header)
        print(f"{'-' * len(header)}")
        for mi, ma in zip(iso["matched"], ani["matched"], strict=True):
            h = mi["hour"]
            di = mi["mod_tmrt"] - mi["obs_tmrt"]
            da = ma["mod_tmrt"] - ma["obs_tmrt"]
            print(f"{h:4d} {mi['obs_tmrt']:6.1f}  {mi['mod_tmrt']:6.1f} {di:+6.1f}  {ma['mod_tmrt']:6.1f} {da:+6.1f}")

        print(f"\n{'Metric':>10s} {'Iso':>10s} {'Aniso':>12s}")
        print(f"{'-' * 36}")
        print(f"{'RMSE':>10s} {iso['rmse']:9.2f}° {ani['rmse']:11.2f}°")
        print(f"{'MAE':>10s} {iso['mae']:9.2f}° {ani['mae']:11.2f}°")
        print(f"{'Bias':>10s} {iso['bias']:+9.2f}° {ani['bias']:+11.2f}°")
        print(f"{'R²':>10s} {iso['r2']:9.3f}  {ani['r2']:11.3f}")

        assert ani["rmse"] < 17.0, f"Tmrt RMSE={ani['rmse']:.2f}°C exceeds 17°C threshold for day {day_code}"

    @pytest.mark.slow
    @pytest.mark.skipif(not _geodata_present, reason="GVC geodata not present")
    @pytest.mark.parametrize("day_code", ["20100707", "20100710", "20100712"])
    def test_radiation_budget_vs_observations(
        self, poi, surface, location, gvc_materials, gvc_human, tmp_path, day_code
    ):
        """Compare SOLWEIG radiation outputs at POI against measured fluxes."""
        import solweig
        from conftest import read_timestep_geotiff

        all_obs = load_gvc_measurements()
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
                materials=gvc_materials,
                human=gvc_human,
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

        print(f"\n{'=' * 60}")
        print(f"  Radiation Summary (day {day_code}): RMSE / MAE / Bias  (W/m²)")
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
        assert ldown_rmse < 75.0, f"Ldown RMSE={ldown_rmse:.1f} W/m² exceeds 75 threshold for day {day_code}"
