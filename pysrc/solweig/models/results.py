"""Result container returned by SOLWEIG calculations.

Defines :class:`SolweigResult`, which holds the output grids (Tmrt,
radiation components, shadow mask) and provides convenience methods
for computing thermal comfort indices (UTCI, PET) and saving outputs
to GeoTIFF.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..models import HumanParams
    from .state import ThermalState
    from .surface import SurfaceData
    from .weather import Weather

logger = get_logger(__name__)


@dataclass
class SolweigResult:
    """
    Results from a SOLWEIG calculation.

    All output grids have the same shape as the input DSM.

    Attributes:
        tmrt: Mean Radiant Temperature grid (°C).
        utci: Universal Thermal Climate Index grid (°C). Optional.
        pet: Physiological Equivalent Temperature grid (°C). Optional.
        shadow: Shadow fraction (1.0=sunlit, 0.0=fully shaded, float).
        kdown: Downwelling shortwave radiation (W/m²).
        kup: Upwelling shortwave radiation (W/m²).
        ldown: Downwelling longwave radiation (W/m²).
        lup: Upwelling longwave radiation (W/m²).
        state: Thermal state for multi-timestep chaining. Optional.
            When state parameter was passed to calculate(), this contains
            the updated state for the next timestep.
    """

    tmrt: NDArray[np.floating]
    shadow: NDArray[np.floating] | None = None
    kdown: NDArray[np.floating] | None = None
    kup: NDArray[np.floating] | None = None
    ldown: NDArray[np.floating] | None = None
    lup: NDArray[np.floating] | None = None
    utci: NDArray[np.floating] | None = None
    pet: NDArray[np.floating] | None = None
    state: ThermalState | None = None

    def to_geotiff(
        self,
        output_dir: str | Path,
        timestamp: dt | None = None,
        outputs: list[str] | None = None,
        surface: SurfaceData | None = None,
        transform: list[float] | None = None,
        crs_wkt: str | None = None,
    ) -> None:
        """
        Save results to GeoTIFF files.

        Creates one GeoTIFF file per output variable per timestep.
        Filename pattern: {output}_{YYYYMMDD}_{HHMM}.tif

        Args:
            output_dir: Directory to write GeoTIFF files.
            timestamp: Timestamp for filename. If None, uses current time.
            outputs: List of outputs to save. Options: "tmrt", "utci", "pet",
                "shadow", "kdown", "kup", "ldown", "lup".
                Default: ["tmrt"] (only save Mean Radiant Temperature).
            surface: SurfaceData object (if loaded via from_geotiff, contains CRS/transform).
                If provided and transform/crs_wkt not specified, uses surface metadata.
            transform: GDAL-style geotransform [x_origin, pixel_width, 0,
                y_origin, 0, -pixel_height]. If None, attempts to use surface metadata,
                otherwise uses identity transform.
            crs_wkt: Coordinate reference system in WKT format. If None, attempts to use
                surface metadata, otherwise no CRS set.

        Example:
            # `SolweigResult` is the per-timestep internal result; user code
            # rarely constructs one directly (most workflows iterate via
            # `calculate()` and consume its `TimeseriesSummary`). Given a
            # `SolweigResult` from an advanced single-step path:

            # With surface metadata (carries CRS/transform from from_geotiff)
            >>> result.to_geotiff("output/", timestamp=weather.datetime, surface=surface)

            # Without surface metadata (explicit transform/CRS)
            >>> result.to_geotiff(
            ...     "output/",
            ...     timestamp=datetime(2025, 7, 15, 12, 0),
            ...     outputs=["tmrt", "utci", "pet"],
            ...     transform=[0, 1, 0, 0, 0, -1],
            ...     crs_wkt="EPSG:32633",
            ... )
        """
        from .. import io

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Default outputs: just tmrt
        if outputs is None:
            outputs = ["tmrt"]

        # Default timestamp: current time
        if timestamp is None:
            timestamp = dt.now()

        # Format timestamp for filename
        ts_str = timestamp.strftime("%Y%m%d_%H%M")

        # Use surface metadata if available and not overridden
        if surface is not None:
            if transform is None and surface.geotransform is not None:
                transform = surface.geotransform
            if crs_wkt is None and surface.crs is not None:
                crs_wkt = surface.crs

        # Default transform: identity (top-left origin, 1m pixels)
        if transform is None:
            height, width = self.tmrt.shape
            transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]

        # Default CRS: empty string (no CRS)
        if crs_wkt is None:
            crs_wkt = ""

        # Map output names to arrays
        available_outputs = {
            "tmrt": self.tmrt,
            "utci": self.utci,
            "pet": self.pet,
            "shadow": self.shadow,
            "kdown": self.kdown,
            "kup": self.kup,
            "ldown": self.ldown,
            "lup": self.lup,
        }

        # Save each requested output
        for name in outputs:
            if name not in available_outputs:
                logger.warning(f"Unknown output '{name}', skipping. Valid: {list(available_outputs.keys())}")
                continue

            array = available_outputs[name]
            if array is None:
                logger.warning(f"Output '{name}' is None (not computed), skipping.")
                continue

            # Write to GeoTIFF in component subdirectory
            comp_dir = output_dir / name
            comp_dir.mkdir(parents=True, exist_ok=True)
            filepath = comp_dir / f"{name}_{ts_str}.tif"
            io.save_raster(
                out_path_str=str(filepath),
                data_arr=array,
                trf_arr=transform,
                crs_wkt=crs_wkt,
                no_data_val=np.nan,
            )
            logger.debug(f"Saved {name} to {filepath}")

    def compute_utci(
        self,
        weather_or_ta: Weather | float,
        rh: float | None = None,
        wind: float | None = None,
    ) -> NDArray[np.floating]:
        """
        Compute UTCI (Universal Thermal Climate Index) from this result's Tmrt.

        Can be called with either a Weather object or individual values:
            utci = result.compute_utci(weather)
            utci = result.compute_utci(ta=25.0, rh=50.0, wind=2.0)

        Args:
            weather_or_ta: Either a Weather object, or air temperature in °C.
            rh: Relative humidity in % (required if weather_or_ta is float).
            wind: Wind speed at 10m height in m/s. Default 1.0 if not provided.

        Returns:
            UTCI grid (°C) with same shape as tmrt.

        Example:
            # Given a `SolweigResult` from an advanced single-step path
            # (user code rarely constructs one directly; `calculate()` returns
            # a `TimeseriesSummary` that already aggregates UTCI for you):

            # Pattern A: Pass weather object (convenient)
            utci = result.compute_utci(weather)

            # Pattern B: Pass individual values (explicit)
            utci = result.compute_utci(25.0, rh=50.0, wind=2.0)
        """
        from ..postprocess import compute_utci_grid
        from .weather import Weather as WeatherClass

        # Duck-type check avoids isinstance failure from dual module import paths
        if hasattr(weather_or_ta, "ta") and hasattr(weather_or_ta, "rh"):
            w: WeatherClass = weather_or_ta  # type: ignore[assignment]
            return compute_utci_grid(self.tmrt, w.ta, w.rh, w.ws)
        else:
            # Individual values
            ta = float(weather_or_ta)
            if rh is None:
                raise ValueError("rh is required when ta is provided as a float")
            return compute_utci_grid(self.tmrt, ta, rh, wind if wind is not None else 1.0)

    def compute_pet(
        self,
        weather_or_ta: Weather | float,
        rh: float | None = None,
        wind: float | None = None,
        human: HumanParams | None = None,
    ) -> NDArray[np.floating]:
        """
        Compute PET (Physiological Equivalent Temperature) from this result's Tmrt.

        Can be called with either a Weather object or individual values:
            pet = result.compute_pet(weather)
            pet = result.compute_pet(ta=25.0, rh=50.0, wind=2.0)

        Args:
            weather_or_ta: Either a Weather object, or air temperature in °C.
            rh: Relative humidity in % (required if weather_or_ta is float).
            wind: Wind speed at 10m height in m/s. Default 1.0 if not provided.
            human: Human body parameters. Uses defaults if not provided.

        Returns:
            PET grid (°C) with same shape as tmrt.

        Note:
            PET uses an iterative solver and is ~50× slower than UTCI.

        Example:
            # Given a `SolweigResult` from an advanced single-step path
            # (user code rarely constructs one directly; `calculate()` returns
            # a `TimeseriesSummary` that already aggregates PET if you request
            # `outputs=["pet"]`):

            # Pattern A: Pass weather object (convenient)
            pet = result.compute_pet(weather)

            # Pattern B: Pass individual values with custom human params
            pet = result.compute_pet(
                25.0, rh=50.0, wind=2.0,
                human=HumanParams(weight=70, height=1.65)
            )
        """
        from ..postprocess import compute_pet_grid
        from .weather import Weather as WeatherClass

        # Duck-type check avoids isinstance failure from dual module import paths
        if hasattr(weather_or_ta, "ta") and hasattr(weather_or_ta, "rh"):
            w: WeatherClass = weather_or_ta  # type: ignore[assignment]
            return compute_pet_grid(self.tmrt, w.ta, w.rh, w.ws, human)
        else:
            # Individual values
            ta = float(weather_or_ta)
            if rh is None:
                raise ValueError("rh is required when ta is provided as a float")
            return compute_pet_grid(self.tmrt, ta, rh, wind if wind is not None else 1.0, human)
