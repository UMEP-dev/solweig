"""Smoke test for an installed wheel.

Run by the publish workflow against the wheel it has just built, before the
release job. It checks that the package imports, that the compiled extension
loads, and that one small run writes a plausible Tmrt raster. It is not a
numerical test; those run in the test workflow.

Usage: python scripts/wheel_smoke.py [--allow-source]

Without ``--allow-source`` it fails when solweig is imported from this
checkout (an editable install), since that would not test the wheel.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np


def main() -> int:
    import solweig
    from solweig import rustalgos  # noqa: F401  (the compiled extension)
    from solweig.io import load_raster

    here = Path(__file__).resolve().parent.parent
    if "--allow-source" not in sys.argv and here in Path(solweig.__file__).resolve().parents:
        print(f"solweig was imported from the source tree ({solweig.__file__}), not from the wheel")
        return 1

    dsm = np.zeros((40, 40), dtype=np.float32)
    dsm[15:25, 15:25] = 12.0
    surface = solweig.SurfaceData(dsm=dsm, pixel_size=1.0)
    surface.compute_svf()
    weather = solweig.Weather(datetime=datetime(2024, 7, 15, 13, 0), ta=25.0, rh=50.0, global_rad=700.0)
    location = solweig.Location(latitude=57.7, longitude=12.0, utc_offset=1)

    with tempfile.TemporaryDirectory() as tmp:
        solweig.calculate(surface, [weather], location, output_dir=tmp, outputs=["tmrt", "utci"])
        tifs = sorted((Path(tmp) / "tmrt").glob("*.tif"))
        if len(tifs) != 1:
            print(f"expected one Tmrt raster, found {len(tifs)}")
            return 1
        tmrt = load_raster(str(tifs[0]))[0]

    ground = tmrt[dsm == 0.0]
    if not np.all(np.isfinite(ground)):
        print("Tmrt has non-finite values over ground")
        return 1
    if not (float(ground.min()) > 20.0 and float(ground.max()) < 80.0 and float(np.ptp(ground)) > 5.0):
        print(f"Tmrt out of the plausible range: min {ground.min():.1f}, max {ground.max():.1f}")
        return 1

    print(
        f"solweig {version('solweig')} on Python {sys.version.split()[0]} ({sys.platform}): "
        f"Tmrt {ground.min():.1f} to {ground.max():.1f} degC, GPU available: {solweig.is_gpu_available()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
