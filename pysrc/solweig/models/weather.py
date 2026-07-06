"""Weather and location data models.

Defines :class:`Location` (geographic coordinates and UTC offset) and
:class:`Weather` (per-timestep meteorological observations).  Derived
fields such as sun position and the direct/diffuse radiation split are
computed lazily via :meth:`Weather.compute_derived`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..physics import sun_position as sp
from ..physics.clearnessindex_2013b import clearnessindex_2013b
from ..physics.diffusefraction import diffusefraction
from ..solweig_logging import get_logger

# Location and _transform_to_wgs84 were extracted to models/location.py to
# keep this module under the 700-line hot-file threshold. Re-export so
# ``from solweig.models.weather import Location`` (and the existing
# ``solweig.models`` re-export) keep working unchanged.
from .location import Location, _transform_to_wgs84  # noqa: F401

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# Module-level cache for altmax: keyed by (date, lat, lon, utc_offset) to
# avoid repeating the expensive 96-iteration sun position loop for every
# timestep on the same day.
_altmax_cache: dict[tuple, float] = {}


@dataclass
class Weather:
    """
    Weather/meteorological data for a single timestep.

    Only basic measurements are required. Derived values (sun position,
    direct/diffuse radiation split) are computed automatically.

    Attributes:
        datetime: Date and time of measurement (end of interval).
        ta: Air temperature in °C.
        rh: Relative humidity in % (0-100).
        global_rad: Global solar radiation in W/m².
        ws: Wind speed in m/s. Default 1.0.
        pressure: Atmospheric pressure in hPa. Default 1013.25.
        timestep_minutes: Data timestep in minutes. Default 60.0.
            Sun position is computed at datetime - timestep/2 to represent
            the center of the measurement interval.
        measured_direct_rad: Optional measured direct beam radiation in W/m².
            If provided with measured_diffuse_rad, these override the computed values.
        measured_diffuse_rad: Optional measured diffuse radiation in W/m².
            If provided with measured_direct_rad, these override the computed values.

    Auto-computed (after calling compute_derived()):
        sun_altitude: Sun altitude angle in degrees. Initial: 0.0.
        sun_azimuth: Sun azimuth angle in degrees. Initial: 0.0.
        sun_zenith: Sun zenith angle in degrees. Initial: 90.0.
        direct_rad: Direct beam radiation in W/m². Initial: 0.0.
        diffuse_rad: Diffuse radiation in W/m². Initial: 0.0.
        clearness_index: Clearness index (0-1). Initial: 1.0.
        altmax: Maximum sun altitude for the day in degrees. Initial: 45.0.
    """

    datetime: dt
    ta: float
    rh: float
    global_rad: float
    ws: float = 1.0
    pressure: float = 1013.25
    timestep_minutes: float = 60.0  # Timestep in minutes (for half-timestep sun position offset)
    measured_direct_rad: float | None = None  # Optional measured direct beam radiation
    measured_diffuse_rad: float | None = None  # Optional measured diffuse radiation
    precomputed_sun_altitude: float | None = None  # Optional pre-computed sun altitude
    precomputed_sun_azimuth: float | None = None  # Optional pre-computed sun azimuth
    precomputed_altmax: float | None = None  # Optional pre-computed max sun altitude for day

    # Auto-computed values (set by compute_derived)
    sun_altitude: float = field(default=0.0, init=False)
    sun_azimuth: float = field(default=0.0, init=False)
    sun_zenith: float = field(default=90.0, init=False)
    direct_rad: float = field(default=0.0, init=False)
    diffuse_rad: float = field(default=0.0, init=False)
    clearness_index: float = field(default=1.0, init=False)
    altmax: float = field(default=45.0, init=False)  # Maximum sun altitude for the day

    _derived_computed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if not 0 <= self.rh <= 100:
            raise ValueError(f"Relative humidity must be in [0, 100], got {self.rh}")
        if self.global_rad < 0:
            raise ValueError(f"Global radiation must be >= 0, got {self.global_rad}")
        if self.ws < 0:
            raise ValueError(f"Wind speed must be >= 0, got {self.ws}")
        if not 300 <= self.pressure <= 1100:
            raise ValueError(f"Pressure must be in [300, 1100] hPa, got {self.pressure}")
        if self.timestep_minutes <= 0:
            raise ValueError(f"timestep_minutes must be > 0, got {self.timestep_minutes}")
        if self.measured_direct_rad is not None and self.measured_direct_rad < 0:
            raise ValueError(f"measured_direct_rad must be >= 0, got {self.measured_direct_rad}")
        if self.measured_diffuse_rad is not None and self.measured_diffuse_rad < 0:
            raise ValueError(f"measured_diffuse_rad must be >= 0, got {self.measured_diffuse_rad}")

    def compute_derived(self, location: Location) -> None:
        """
        Compute derived values: sun position and radiation split.

        Must be called before using sun_altitude, sun_azimuth, direct_rad,
        or diffuse_rad.

        Sun position is calculated at the center of the measurement interval
        (datetime - timestep/2), which is standard for meteorological data
        where measurements are averaged over the interval.

        Args:
            location: Geographic location for sun position calculation.
        """
        # Always create location_dict (needed for clearness index calculation)
        location_dict = location.to_sun_position_dict()

        # Use pre-computed sun position if provided, otherwise compute
        if self.precomputed_sun_altitude is not None and self.precomputed_sun_azimuth is not None:
            self.sun_altitude = self.precomputed_sun_altitude
            self.sun_azimuth = self.precomputed_sun_azimuth
            self.sun_zenith = 90.0 - self.sun_altitude
            self.altmax = self.precomputed_altmax if self.precomputed_altmax is not None else self.sun_altitude
        else:
            # Apply half-timestep offset for sun position
            # Meteorological data timestamps typically represent the end of an interval,
            # so we compute sun position at the center of the interval to match SOLWEIG runner
            from datetime import timedelta

            half_timestep = timedelta(minutes=self.timestep_minutes / 2.0)
            sun_time = self.datetime - half_timestep

            # Compute sun position using NREL algorithm
            time_dict = {
                "year": sun_time.year,
                "month": sun_time.month,
                "day": sun_time.day,
                "hour": sun_time.hour,
                "min": sun_time.minute,
                "sec": sun_time.second,
                "UTC": location.utc_offset,
            }
            location_dict = location.to_sun_position_dict()

            sun = sp.sun_position(time_dict, location_dict)

            # Extract scalar values (sun_position may return 0-d arrays)
            zenith = sun["zenith"]
            azimuth = sun["azimuth"]
            self.sun_zenith = float(np.asarray(zenith).flat[0]) if hasattr(zenith, "__iter__") else float(zenith)
            self.sun_azimuth = float(np.asarray(azimuth).flat[0]) if hasattr(azimuth, "__iter__") else float(azimuth)
            self.sun_altitude = 90.0 - self.sun_zenith

            # Use pre-computed altmax if available (avoids expensive 96-iteration loop)
            if self.precomputed_altmax is not None:
                self.altmax = self.precomputed_altmax
            else:
                # Calculate maximum sun altitude for the day (iterate in 15-min intervals).
                # Cache by (date, lat, lon, utc_offset) so the loop runs once per day.
                from datetime import timedelta

                ymd = self.datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                cache_key = (ymd.date(), location.latitude, location.longitude, location.utc_offset)

                if cache_key in _altmax_cache:
                    self.altmax = _altmax_cache[cache_key]
                else:
                    sunmaximum = -90.0
                    fifteen_min = 15.0 / 1440.0  # 15 minutes as fraction of day

                    for step in range(96):  # 24 hours * 4 (15-min intervals)
                        step_time = ymd + timedelta(days=step * fifteen_min)
                        time_dict_step = {
                            "year": step_time.year,
                            "month": step_time.month,
                            "day": step_time.day,
                            "hour": step_time.hour,
                            "min": step_time.minute,
                            "sec": 0,
                            "UTC": location.utc_offset,
                        }
                        sun_step = sp.sun_position(time_dict_step, location_dict)
                        zenith_step = sun_step["zenith"]
                        zenith_val = (
                            float(np.asarray(zenith_step).flat[0])
                            if hasattr(zenith_step, "__iter__")
                            else float(zenith_step)
                        )
                        alt_step = 90.0 - zenith_val
                        if alt_step > sunmaximum:
                            sunmaximum = alt_step

                    self.altmax = max(sunmaximum, 0.0)  # Ensure non-negative
                    _altmax_cache[cache_key] = self.altmax

        # Use measured radiation values if provided, otherwise compute
        if self.measured_direct_rad is not None and self.measured_diffuse_rad is not None:
            # Use pre-measured direct and diffuse radiation
            self.direct_rad = self.measured_direct_rad
            self.diffuse_rad = self.measured_diffuse_rad
            # Still compute clearness index for diagnostics/plotting
            if self.sun_altitude > 0 and self.global_rad > 0:
                zen_rad = self.sun_zenith * (np.pi / 180.0)
                result = clearnessindex_2013b(
                    zen_rad,
                    self.datetime.timetuple().tm_yday,
                    self.ta,
                    self.rh / 100.0,
                    self.global_rad,
                    location_dict,
                    self.pressure,
                )
                _, self.clearness_index, _, _, _ = result
            else:
                # Night/no-radiation: CI is undefined.  Use 1.0 ("clear")
                # so the Ldown cloud-correction branch (ci < 0.95) is NOT
                # triggered — otherwise esky is replaced by 1.0 (blackbody)
                # and nighttime Ldown becomes SBC·Ta⁴ regardless of SVF.
                self.clearness_index = 1.0
        elif self.sun_altitude > 0 and self.global_rad > 0:
            # Compute clearness index
            zen_rad = self.sun_zenith * (np.pi / 180.0)
            result = clearnessindex_2013b(
                zen_rad,
                self.datetime.timetuple().tm_yday,
                self.ta,
                self.rh / 100.0,
                self.global_rad,
                location_dict,
                self.pressure,
            )
            # clearnessindex_2013b returns: (I0, CI, Kt, I0_et, diff_et)
            _, self.clearness_index, kt, _, _ = result

            # Use Reindl model for diffuse fraction
            self.direct_rad, self.diffuse_rad = diffusefraction(
                self.global_rad, self.sun_altitude, kt, self.ta, self.rh
            )
        else:
            # Night or no radiation
            self.direct_rad = 0.0
            self.diffuse_rad = self.global_rad
            # CI is undefined at night.  Use 1.0 ("clear") so the Ldown
            # cloud-correction (ci < 0.95) is not triggered — see above.
            self.clearness_index = 1.0

        self._derived_computed = True

    @property
    def is_daytime(self) -> bool:
        """Check if sun is above horizon."""
        return self.sun_altitude > 0

    @classmethod
    def from_values(
        cls,
        ta: float,
        rh: float,
        global_rad: float,
        datetime: dt,
        ws: float = 1.0,
        **kwargs: Any,
    ) -> Weather:
        """
        Quick factory for creating Weather with minimal required values.

        Useful for testing and single-timestep calculations where you
        just need to specify the essential parameters.

        Args:
            ta: Air temperature in °C.
            rh: Relative humidity in % (0-100).
            global_rad: Global solar radiation in W/m².
            datetime: Date and time of measurement.
            ws: Wind speed in m/s. Default 1.0.
            **kwargs: Additional Weather parameters (pressure, etc.)

        Returns:
            Weather object ready for calculation.

        Example:
            weather = Weather.from_values(
                ta=30, rh=60, global_rad=900,
                datetime=datetime(2025, 7, 15, 14, 0)
            )
        """
        return cls(datetime=datetime, ta=ta, rh=rh, global_rad=global_rad, ws=ws, **kwargs)

    @classmethod
    def from_epw(
        cls,
        path: str | Path,
        start: str | dt | None = None,
        end: str | dt | None = None,
        hours: list[int] | None = None,
        year: int | None = None,
    ) -> list[Weather]:
        """
        Load weather data from an EnergyPlus Weather (EPW) file.

        Args:
            path: Path to the EPW file.
            start: Start date/datetime. Can be:
                   - ISO date string "YYYY-MM-DD" or "MM-DD" (for TMY with year=None)
                   - datetime object
                   If None, uses first date in file.
            end: End date/datetime (inclusive). Same format as start.
                 If None, uses same as start (single day).
            hours: List of hours to include (0-23). If None, includes all hours.
            year: Year override for TMY files. If None and start/end use MM-DD format,
                  matches any year in the file.

        Returns:
            List of Weather objects for each timestep in the requested range.

        Raises:
            FileNotFoundError: If the EPW file doesn't exist.
            ValueError: If requested dates are outside the EPW file's date range.

        Example:
            # Load a single day
            weather_list = Weather.from_epw("weather.epw", start="2023-07-15", end="2023-07-15")

            # Load with specific hours only (daylight hours)
            weather_list = Weather.from_epw(
                "weather.epw",
                start="2023-07-15",
                end="2023-07-16",
                hours=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            )

            # TMY file (year-agnostic)
            weather_list = Weather.from_epw("tmy.epw", start="07-15", end="07-15")
        """
        from .. import io as common

        # Parse EPW file
        df, epw_info = common.read_epw(path)

        # Parse start/end dates
        def parse_date(date_val, is_tmy: bool, default_year: int):
            if date_val is None:
                return None
            if isinstance(date_val, dt):
                return date_val
            # String parsing
            date_str = str(date_val)
            if "-" in date_str:
                parts = date_str.split("-")
                if len(parts) == 2:
                    # MM-DD format (TMY)
                    month, day = int(parts[0]), int(parts[1])
                    return dt(default_year, month, day)
                elif len(parts) == 3:
                    # YYYY-MM-DD format
                    return dt.fromisoformat(date_str)
            raise ValueError(f"Cannot parse date: {date_val}. Use 'YYYY-MM-DD' or 'MM-DD' format.")

        # Determine if using TMY mode (year-agnostic)
        is_tmy = year is None and start is not None and isinstance(start, str) and len(start.split("-")) == 2

        # Get default year from EPW data
        if df.index.empty:
            raise ValueError("EPW file contains no data")
        default_year = df.index[0].year if year is None else year

        # Parse dates
        start_dt = parse_date(start, is_tmy, default_year)
        end_dt = parse_date(end, is_tmy, default_year)

        if start_dt is None:
            # Default documented as "first date in file": normalize to midnight
            # so the end-of-day expansion below covers the whole first day
            # (EPW files start at hour 1, so the raw first timestamp is 01:00).
            start_dt = df.index[0].replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
        if end_dt is None:
            end_dt = start_dt

        # Make end_dt inclusive of the full day
        if end_dt.hour == 0 and end_dt.minute == 0:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        # Filter by date range
        # Remove timezone from index for comparison if needed
        df_idx = df.index.tz_localize(None) if df.index.tz is not None else df.index

        if is_tmy:
            # TMY mode: match month and day, ignore year
            # Build (month, day) tuples for comparison — handles year-crossing ranges
            # like Dec 15 → Jan 15 correctly.
            start_md = (start_dt.month, start_dt.day)
            end_md = (end_dt.month, end_dt.day)
            idx_md = list(zip(df_idx.month, df_idx.day, strict=False))

            if start_md <= end_md:
                # Normal range (e.g., Feb 7 → Feb 8)
                mask = [(start_md <= md <= end_md) for md in idx_md]
            else:
                # Year-crossing range (e.g., Dec 15 → Jan 15)
                mask = [(md >= start_md or md <= end_md) for md in idx_md]
        else:
            # Normal mode: match full datetime
            mask = (df_idx >= start_dt) & (df_idx <= end_dt)

        df_filtered = df[mask]

        if df_filtered.empty:
            # Build helpful error message
            avail_start = df_idx.min()
            avail_end = df_idx.max()
            raise ValueError(
                f"Requested dates {start_dt.date()} to {end_dt.date()} not found in EPW file.\n"
                f"EPW file '{path}' contains data for: {avail_start.date()} to {avail_end.date()}\n"
                "Suggestions:\n"
                "  - Use dates within the available range\n"
                "  - For TMY files, use 'MM-DD' format (e.g., '07-15') to match any year"
            )

        # Filter by hours if specified
        if hours is not None:
            hours_set = set(hours)
            hour_mask = df_filtered.index.hour.isin(hours_set)
            df_filtered = df_filtered[hour_mask]

        # Create Weather objects
        weather_list = []
        for timestamp, row in df_filtered.iterrows():
            # Create Weather object with available data
            # EPW has dni/dhi which we can use as measured values
            w = cls(
                datetime=timestamp.to_pydatetime().replace(tzinfo=None),
                ta=float(row["temp_air"]) if not np.isnan(row["temp_air"]) else 20.0,
                rh=float(row["relative_humidity"]) if not np.isnan(row["relative_humidity"]) else 50.0,
                global_rad=float(row["ghi"]) if not np.isnan(row["ghi"]) else 0.0,
                ws=float(row["wind_speed"]) if not np.isnan(row["wind_speed"]) else 1.0,
                pressure=(float(row["atmospheric_pressure"]) / 100.0)
                if not np.isnan(row["atmospheric_pressure"])
                else 1013.25,  # Convert Pa to hPa
                measured_direct_rad=float(row["dni"]) if not np.isnan(row["dni"]) else None,
                measured_diffuse_rad=float(row["dhi"]) if not np.isnan(row["dhi"]) else None,
            )
            weather_list.append(w)

        if weather_list:
            logger.info(
                f"Loaded {len(weather_list)} timesteps from EPW: "
                f"{weather_list[0].datetime.strftime('%Y-%m-%d %H:%M')} → "
                f"{weather_list[-1].datetime.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            logger.warning(f"No timesteps found in EPW file for date range {start_dt} to {end_dt}")

        return weather_list

    @classmethod
    def from_umep_met(
        cls,
        paths: str | Path | list[str | Path],
        resample_hourly: bool = True,
        start: str | dt | None = None,
        end: str | dt | None = None,
    ) -> list[Weather]:
        """
        Load weather data from UMEP/SUEWS meteorological forcing files.

        The UMEP met format is space-separated with columns:
        %iy, id, it, imin, Q*, QH, QE, Qs, Qf, Wind, RH, Td, press,
        rain, Kdn, snow, ldown, fcld, wuh, xsmd, lai_hr, Kdiff, Kdir, Wd

        Missing values are encoded as -999.

        Args:
            paths: Path(s) to UMEP met file(s). Multiple files are
                concatenated (e.g., one per month).
            resample_hourly: If True, keep only on-the-hour rows (imin=0).
                Default True since SOLWEIG works best with hourly data.
            start: Start date filter as "YYYY-MM-DD" or datetime. Optional.
            end: End date filter (inclusive) as "YYYY-MM-DD" or datetime. Optional.

        Returns:
            List of Weather objects sorted by datetime.

        Example:
            # Single file
            weather = Weather.from_umep_met("metdata_10min_july.txt")

            # Multiple monthly files
            weather = Weather.from_umep_met([
                "metdata_10min_may.txt",
                "metdata_10min_june.txt",
            ])
        """
        from datetime import timedelta

        if isinstance(paths, (str, Path)):
            paths = [paths]

        rows: list[dict[str, float]] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"UMEP met file not found: {path}")

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("%") or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 24:
                        continue
                    try:
                        rows.append(
                            {
                                "year": int(parts[0]),
                                "doy": int(parts[1]),
                                "hour": int(parts[2]),
                                "minute": int(parts[3]),
                                "wind": float(parts[9]),
                                "rh": float(parts[10]),
                                "ta": float(parts[11]),
                                "press_kpa": float(parts[12]),
                                "kdn": float(parts[14]),
                                "kdiff": float(parts[21]),
                                "kdir": float(parts[22]),
                            }
                        )
                    except (ValueError, IndexError):
                        continue

        if not rows:
            raise ValueError(f"No valid data rows found in UMEP met files: {paths}")

        # Filter hourly if requested
        if resample_hourly:
            rows = [r for r in rows if r["minute"] == 0]

        # Auto-detect timestep from consecutive rows (default 60 min).
        # When resample_hourly=False the data may still be hourly but with
        # imin ≠ 0 (e.g. imin=7).  A hard-coded 10 min would give the wrong
        # half-timestep offset for sun position, so we infer the actual gap.
        if len(rows) >= 2:
            t0 = dt(int(rows[0]["year"]), 1, 1) + timedelta(
                days=int(rows[0]["doy"]) - 1,
                hours=int(rows[0]["hour"]),
                minutes=int(rows[0]["minute"]),
            )
            t1 = dt(int(rows[1]["year"]), 1, 1) + timedelta(
                days=int(rows[1]["doy"]) - 1,
                hours=int(rows[1]["hour"]),
                minutes=int(rows[1]["minute"]),
            )
            detected_timestep = max((t1 - t0).total_seconds() / 60.0, 1.0)
        else:
            detected_timestep = 60.0

        # Convert to Weather objects
        weather_list = []
        for r in rows:
            timestamp = dt(int(r["year"]), 1, 1) + timedelta(
                days=int(r["doy"]) - 1, hours=int(r["hour"]), minutes=int(r["minute"])
            )

            # Skip rows with missing critical data
            if r["ta"] <= -998 or r["rh"] <= -998 or r["kdn"] <= -998:
                continue

            # Convert pressure from kPa to hPa (-999 means missing)
            pressure = r["press_kpa"] * 10.0 if r["press_kpa"] > -998 else 1013.25

            # Direct/diffuse radiation (-999 means missing)
            kdir = r["kdir"] if r["kdir"] > -998 else None
            kdiff = r["kdiff"] if r["kdiff"] > -998 else None

            # Wind speed (-999 means missing)
            ws = r["wind"] if r["wind"] > -998 else 1.0

            timestep = detected_timestep

            w = cls(
                datetime=timestamp,
                ta=r["ta"],
                rh=r["rh"],
                global_rad=max(r["kdn"], 0.0),
                ws=ws,
                pressure=pressure,
                timestep_minutes=timestep,
                measured_direct_rad=kdir,
                measured_diffuse_rad=kdiff,
            )
            weather_list.append(w)

        # Sort by datetime
        weather_list.sort(key=lambda w: w.datetime)

        # Apply date filters
        if start is not None:
            start_dt = dt.fromisoformat(start) if isinstance(start, str) else start
            weather_list = [w for w in weather_list if w.datetime >= start_dt]
        if end is not None:
            end_dt = dt.fromisoformat(end) if isinstance(end, str) else end
            if end_dt.hour == 0 and end_dt.minute == 0:
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
            weather_list = [w for w in weather_list if w.datetime <= end_dt]

        if weather_list:
            logger.info(
                f"Loaded {len(weather_list)} timesteps from UMEP met: "
                f"{weather_list[0].datetime.strftime('%Y-%m-%d %H:%M')} → "
                f"{weather_list[-1].datetime.strftime('%Y-%m-%d %H:%M')}"
            )
        elif start is not None or end is not None:
            # Warn when date filter produces no results — usually a year mismatch
            all_weather = []
            for r in rows:
                from datetime import timedelta as _td

                ts = dt(int(r["year"]), 1, 1) + _td(
                    days=int(r["doy"]) - 1, hours=int(r["hour"]), minutes=int(r["minute"])
                )
                all_weather.append(ts)
            if all_weather:
                avail_start = min(all_weather)
                avail_end = max(all_weather)
                logger.warning(
                    f"No timesteps found for requested range "
                    f"{start} to {end}.\n"
                    f"  File contains data for: "
                    f"{avail_start.strftime('%Y-%m-%d')} to {avail_end.strftime('%Y-%m-%d')}\n"
                    f"  Check that start/end dates match the year in the file."
                )

        return weather_list
