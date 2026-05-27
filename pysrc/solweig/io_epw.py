"""EPW (EnergyPlus Weather) file parsing and PVGIS TMY download.

Extracted from `io.py` to keep the raster I/O module under the
700-line hot-file threshold. The classes here are a pure-Python
DataFrame stand-in so SOLWEIG can read EPW files without a pandas
dependency. Callers should continue to import the public helpers
(`read_epw`, `download_epw`) from `solweig.io` — they are re-exported
there for backwards compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class _EpwDataIndex:
    """Lightweight index class mimicking pandas DatetimeIndex for EPW data."""

    def __init__(self, timestamps: list):
        self._timestamps = timestamps
        self.tz = None
        self.name = "datetime"

    def __len__(self):
        return len(self._timestamps)

    def __getitem__(self, idx):
        return self._timestamps[idx]

    def __iter__(self):
        return iter(self._timestamps)

    def __ge__(self, other):
        """Greater than or equal comparison, returns boolean array."""
        return _BooleanArray([t >= other for t in self._timestamps])

    def __le__(self, other):
        """Less than or equal comparison, returns boolean array."""
        return _BooleanArray([t <= other for t in self._timestamps])

    def __gt__(self, other):
        """Greater than comparison, returns boolean array."""
        return _BooleanArray([t > other for t in self._timestamps])

    def __lt__(self, other):
        """Less than comparison, returns boolean array."""
        return _BooleanArray([t < other for t in self._timestamps])

    @property
    def empty(self):
        return len(self._timestamps) == 0

    @property
    def year(self):
        return [t.year for t in self._timestamps]

    @property
    def month(self):
        return _IndexAccessor([t.month for t in self._timestamps])

    @property
    def day(self):
        return _IndexAccessor([t.day for t in self._timestamps])

    @property
    def hour(self):
        return _IndexAccessor([t.hour for t in self._timestamps])

    def min(self):
        return min(self._timestamps) if self._timestamps else None

    def max(self):
        return max(self._timestamps) if self._timestamps else None

    def tz_localize(self, tz):
        # Return self since we don't handle timezones in the fallback
        return self


class _IndexAccessor:
    """Helper for index property access like df.index.hour."""

    def __init__(self, values: list):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def __gt__(self, other):
        return _BooleanArray([v > other for v in self._values])

    def __ge__(self, other):
        return _BooleanArray([v >= other for v in self._values])

    def __lt__(self, other):
        return _BooleanArray([v < other for v in self._values])

    def __le__(self, other):
        return _BooleanArray([v <= other for v in self._values])

    def __eq__(self, other):
        return _BooleanArray([v == other for v in self._values])

    def isin(self, values_set):
        return [v in values_set for v in self._values]


class _BooleanArray:
    """Helper for boolean array operations (& and |)."""

    def __init__(self, values: list):
        self._values = values

    def __and__(self, other):
        if isinstance(other, _BooleanArray):
            return _BooleanArray([a and b for a, b in zip(self._values, other._values, strict=False)])
        return _BooleanArray([a and b for a, b in zip(self._values, other, strict=False)])

    def __or__(self, other):
        if isinstance(other, _BooleanArray):
            return _BooleanArray([a or b for a, b in zip(self._values, other._values, strict=False)])
        return _BooleanArray([a or b for a, b in zip(self._values, other, strict=False)])

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    def __len__(self):
        return len(self._values)

    def all(self):
        return all(self._values)

    def any(self):
        return any(self._values)

    def tolist(self):
        return self._values


class _EpwRow:
    """Lightweight row class mimicking pandas Series for EPW data."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data.get(key, float("nan"))

    def get(self, key, default=None):
        """Get value with default, like dict.get()."""
        val = self._data.get(key, default)
        if val is None or (isinstance(val, float) and val != val):  # NaN check
            return default
        return val


class _EpwColumn:
    """Lightweight column accessor mimicking a pandas Series for a single column."""

    def __init__(self, values: list):
        self._values = values

    def __getitem__(self, idx):
        return self._values[idx]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def min(self):
        return min(v for v in self._values if v == v)  # skip NaN

    def max(self):
        return max(v for v in self._values if v == v)  # skip NaN

    def __ge__(self, other):
        return _BooleanArray([v >= other for v in self._values])

    def __le__(self, other):
        return _BooleanArray([v <= other for v in self._values])

    def __gt__(self, other):
        return _BooleanArray([v > other for v in self._values])

    def __lt__(self, other):
        return _BooleanArray([v < other for v in self._values])

    def all(self):
        return all(self._values)


class _EpwIloc:
    """Positional indexing for _EpwDataFrame."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __getitem__(self, idx):
        return _EpwRow(self._rows[idx])


class _EpwDataFrame:
    """Lightweight DataFrame-like class for EPW data without pandas dependency."""

    def __init__(self, rows: list[dict], timestamps: list):
        self._rows = rows
        self._timestamps = timestamps
        self.index = _EpwDataIndex(timestamps)

    def __len__(self):
        return len(self._rows)

    @property
    def columns(self):
        """Column names from the first row."""
        if self._rows:
            return list(self._rows[0].keys())
        return []

    @property
    def iloc(self):
        """Positional indexing (returns _EpwRow objects)."""
        return _EpwIloc(self._rows)

    def __getitem__(self, key):
        """Access by column name (str) or filter by boolean mask."""
        if isinstance(key, str):
            return _EpwColumn([row.get(key, float("nan")) for row in self._rows])
        if isinstance(key, _BooleanArray):
            key = key._values
        if isinstance(key, list):
            filtered_rows = [r for r, m in zip(self._rows, key, strict=False) if m]
            filtered_ts = [t for t, m in zip(self._timestamps, key, strict=False) if m]
            return _EpwDataFrame(filtered_rows, filtered_ts)
        raise TypeError(f"Unsupported indexing type: {type(key)}")

    @property
    def empty(self):
        return len(self._rows) == 0

    def iterrows(self):
        """Iterate over (timestamp, row) pairs."""
        for ts, row_data in zip(self._timestamps, self._rows, strict=False):
            yield _EpwTimestamp(ts), _EpwRow(row_data)

    def to_dataframe(self):
        """Convert to pandas DataFrame if pandas is available.

        Returns:
            pd.DataFrame with DatetimeIndex, or self if pandas unavailable.
        """
        try:
            import pandas as pd

            df = pd.DataFrame(self._rows)
            df.index = pd.DatetimeIndex(self._timestamps, name="datetime")
            return df
        except ImportError:
            return self


class _EpwTimestamp:
    """Wrapper for datetime to provide pandas-like interface."""

    def __init__(self, dt_obj):
        self._dt = dt_obj

    def __getattr__(self, name):
        return getattr(self._dt, name)

    def to_pydatetime(self):
        return self._dt

    def replace(self, **kwargs):
        return self._dt.replace(**kwargs)


def _parse_epw_metadata(path: Path) -> dict:
    """Parse EPW header to extract metadata."""
    metadata = {}
    with open(path, encoding="utf-8") as f:
        location_line = f.readline().strip()
        if not location_line.startswith("LOCATION"):
            raise ValueError("Invalid EPW file: first line must start with 'LOCATION'")

        location_parts = location_line.split(",")
        if len(location_parts) < 10:
            raise ValueError(f"Invalid LOCATION line: expected at least 10 fields, got {len(location_parts)}")

        metadata["city"] = location_parts[1].strip()
        metadata["state"] = location_parts[2].strip()
        metadata["country"] = location_parts[3].strip()
        metadata["latitude"] = float(location_parts[6])
        metadata["longitude"] = float(location_parts[7])
        metadata["tz_offset"] = float(location_parts[8])
        metadata["elevation"] = float(location_parts[9])

    return metadata


def _read_epw_pure_python(path: Path) -> tuple:
    """Pure Python EPW parser without pandas dependency."""
    import csv
    from datetime import datetime as dt_class
    from datetime import timedelta

    metadata = _parse_epw_metadata(path)

    # Column indices for the fields we need
    # EPW format has 35 fields per line
    col_indices = {
        "year": 0,
        "month": 1,
        "day": 2,
        "hour": 3,
        "minute": 4,
        "temp_air": 6,
        "relative_humidity": 8,
        "atmospheric_pressure": 9,
        "ghi": 13,
        "dni": 14,
        "dhi": 15,
        "wind_direction": 20,
        "wind_speed": 21,
    }

    na_values = {"99", "999", "9999", "99999", "999999999", ""}

    rows = []
    timestamps = []

    with open(path, encoding="utf-8") as f:
        # Skip 8 header lines
        for _ in range(8):
            f.readline()

        reader = csv.reader(f)
        for line in reader:
            if len(line) < 22:
                continue

            try:
                year = int(line[col_indices["year"]])
                month = int(line[col_indices["month"]])
                day = int(line[col_indices["day"]])
                hour = int(line[col_indices["hour"]])
                minute = int(line[col_indices["minute"]])

                # EPW uses 1-24 hour format; hour 24 means midnight of next day
                if hour == 24:
                    timestamp = dt_class(year, month, day, 0, minute) + timedelta(days=1)
                else:
                    timestamp = dt_class(year, month, day, hour, minute)
                timestamps.append(timestamp)

                def parse_float(idx, row_data=line):
                    val = row_data[idx].strip()
                    if val in na_values:
                        return float("nan")
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return float("nan")

                row = {
                    "temp_air": parse_float(col_indices["temp_air"]),
                    "relative_humidity": parse_float(col_indices["relative_humidity"]),
                    "atmospheric_pressure": parse_float(col_indices["atmospheric_pressure"]),
                    "ghi": parse_float(col_indices["ghi"]),
                    "dni": parse_float(col_indices["dni"]),
                    "dhi": parse_float(col_indices["dhi"]),
                    "wind_speed": parse_float(col_indices["wind_speed"]),
                    "wind_direction": parse_float(col_indices["wind_direction"]),
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue

    if not rows:
        raise ValueError("EPW file contains no valid data rows")

    df = _EpwDataFrame(rows, timestamps)
    logger.info(f"Loaded EPW file: {metadata['city']}, {len(df)} timesteps (pure Python parser)")

    return df, metadata


def download_epw(
    latitude: float,
    longitude: float,
    output_path: str | Path,
    *,
    timeout: int = 60,
) -> Path:
    """
    Download a Typical Meteorological Year (TMY) EPW file from PVGIS.

    Uses the EU Joint Research Centre's PVGIS API (v5.3, no API key required).
    Coverage is near-global (all continents except polar regions),
    using ERA5 reanalysis data.

    **Important:** TMY files are *not* observations for a specific year.
    A TMY is a statistical composite — each calendar month is selected from
    the most "typical" month across a multi-year reference period (2005–2023
    for PVGIS v5.3).  The resulting file represents long-term average climate
    conditions, not a recent or continuously updated dataset.  Because each
    month is taken from a real historical year, the row timestamps in the
    returned file will legitimately span multiple years within the reference
    window (e.g. a January from 2022 alongside a February from 2014). This is
    the defining property of a TMY and not an artefact.  The reference period
    is fixed per PVGIS release; data freshness depends on the upstream PVGIS
    version, not on SOLWEIG.

    See the PVGIS TMY documentation for full methodology and data sources:
    https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en

    Ref: UMEP-dev/solweig#8

    The downloaded data contains modified Copernicus Climate Change Service
    information. Neither the European Commission nor ECMWF is responsible
    for any use that may be made of the Copernicus information or data it
    contains. See https://cds.climate.copernicus.eu/disclaimer for the full
    licence terms.

    Args:
        latitude: Latitude in decimal degrees (-90 to 90).
        longitude: Longitude in decimal degrees (-180 to 180).
        output_path: Path where the EPW file will be saved.
        timeout: HTTP request timeout in seconds (default 60).

    Returns:
        Path to the saved EPW file.

    Raises:
        ValueError: If coordinates are out of range.
        ConnectionError: If the PVGIS server is unreachable.
        RuntimeError: If the download fails (e.g. location over ocean).

    Example:
        >>> from solweig.io import download_epw
        >>> path = download_epw(37.98, 23.73, "athens.epw")
        >>> data, metadata = read_epw(path)
    """
    import urllib.error
    import urllib.request

    if not -90 <= latitude <= 90:
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    if not -180 <= longitude <= 180:
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")

    output_path = Path(output_path)

    url = f"https://re.jrc.ec.europa.eu/api/v5_3/tmy?lat={latitude}&lon={longitude}&outputformat=epw"

    logger.info(f"Downloading EPW from PVGIS for ({latitude:.4f}, {longitude:.4f})...")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 400:
            raise RuntimeError(
                f"PVGIS has no data for ({latitude}, {longitude}). The location may be over ocean or outside coverage."
            ) from e
        raise RuntimeError(f"PVGIS download failed (HTTP {e.code}): {e.reason}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach PVGIS server: {e.reason}") from e

    if len(data) < 1000:
        # PVGIS returns a short error message for invalid locations
        text = data.decode("utf-8", errors="replace")
        raise RuntimeError(f"PVGIS returned an error: {text.strip()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    lines = data.decode("utf-8", errors="replace").split("\n")
    n_data_lines = len(lines) - 8  # subtract header lines
    logger.info(f"Saved EPW file: {output_path} ({n_data_lines} hourly records)")

    return output_path


def read_epw(path: str | Path) -> tuple:
    """
    Read EnergyPlus Weather (EPW) file and return weather data with metadata.

    EPW files have 8 header lines followed by hourly weather data.
    Uses pure Python parser (no pandas/scipy dependencies).

    Args:
        path: Path to EPW file (string or Path)

    Returns:
        Tuple of (data, metadata_dict):
        - data: DataFrame-like object with datetime index and weather columns:
            - temp_air: Dry bulb temperature (°C)
            - relative_humidity: Relative humidity (%)
            - atmospheric_pressure: Atmospheric pressure (Pa)
            - ghi: Global horizontal irradiance (W/m²)
            - dni: Direct normal irradiance (W/m²)
            - dhi: Diffuse horizontal irradiance (W/m²)
            - wind_speed: Wind speed (m/s)
            - wind_direction: Wind direction (degrees)
        - metadata_dict: Dictionary with keys:
            - city: Location city name
            - latitude: Latitude (degrees)
            - longitude: Longitude (degrees)
            - elevation: Elevation (m)
            - tz_offset: Timezone offset (hours)

    Raises:
        FileNotFoundError: If EPW file doesn't exist
        ValueError: If EPW file is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")

    return _read_epw_pure_python(path)
