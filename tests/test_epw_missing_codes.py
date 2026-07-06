"""Tests for per-field EPW missing-value handling.

Regression gate: the parser previously used one shared set of magic strings
({"99", "999", "9999", ...}) for every column, which nulled legitimate data
(RH = 99 %, GHI/DNI/DHI = 999 W/m2) and missed real EPW missing markers
(dry-bulb 99.9, station pressure 999999 Pa). EPW missing codes are defined
per field in the EnergyPlus specification.
"""

from __future__ import annotations

import math

import pytest

from solweig.io_epw import read_epw
from solweig.models.weather import Weather

HEADER = (
    "LOCATION,TestCity,TestState,TST,SRC,000000,38.00,23.75,2.0,175.0\n"
    "DESIGN CONDITIONS,0\n"
    "TYPICAL/EXTREME PERIODS,0\n"
    "GROUND TEMPERATURES,0\n"
    "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n"
    "COMMENTS 1,generated for tests\n"
    "COMMENTS 2,\n"
    "DATA PERIODS,1,1,Data,Sunday,1/1,12/31\n"
)


def _row(
    hour: int,
    ta: str = "25.0",
    rh: str = "50",
    pressure: str = "101325",
    ghi: str = "500",
    dni: str = "600",
    dhi: str = "100",
    wind_dir: str = "180",
    ws: str = "3.0",
) -> str:
    """One EPW data line (35 fields) with the columns the parser reads."""
    fields = ["2023", "7", "1", str(hour), "0", "SRC"] + ["0"] * 29
    fields[6] = ta
    fields[7] = "15.0"  # dew point
    fields[8] = rh
    fields[9] = pressure
    fields[13] = ghi
    fields[14] = dni
    fields[15] = dhi
    fields[20] = wind_dir
    fields[21] = ws
    return ",".join(fields) + "\n"


def _write_epw(path, rows: list[str]):
    path.write_text(HEADER + "".join(rows))
    return path


def test_legitimate_boundary_values_are_kept(tmp_path):
    """RH=99 %, GHI/DNI/DHI=999 W/m2, and Ta=-9.9 are real data, not missing."""
    path = _write_epw(
        tmp_path / "boundary.epw",
        [_row(1, ta="-9.9", rh="99", ghi="999", dni="999", dhi="999")],
    )
    df, _ = read_epw(path)
    row = df.iloc[0]
    assert row["temp_air"] == pytest.approx(-9.9)
    assert row["relative_humidity"] == pytest.approx(99.0)
    assert row["ghi"] == pytest.approx(999.0)
    assert row["dni"] == pytest.approx(999.0)
    assert row["dhi"] == pytest.approx(999.0)


def test_spec_missing_codes_become_nan(tmp_path):
    """Per-field EPW missing markers must parse as NaN."""
    path = _write_epw(
        tmp_path / "missing.epw",
        [
            _row(
                1,
                ta="99.9",
                rh="999",
                pressure="999999",
                ghi="9999",
                dni="9999",
                dhi="9999",
                wind_dir="999",
                ws="999",
            )
        ],
    )
    df, _ = read_epw(path)
    row = df.iloc[0]
    for field in (
        "temp_air",
        "relative_humidity",
        "atmospheric_pressure",
        "ghi",
        "dni",
        "dhi",
        "wind_direction",
        "wind_speed",
    ):
        assert math.isnan(row[field]), f"{field} missing code not detected"


def test_missing_pressure_falls_back_to_standard_atmosphere(tmp_path):
    """Weather.from_epw substitutes 1013.25 hPa when station pressure is missing."""
    path = _write_epw(
        tmp_path / "pressure.epw",
        [_row(1, pressure="999999"), _row(2, pressure="98500")],
    )
    weather = Weather.from_epw(str(path))
    assert weather[0].pressure == pytest.approx(1013.25)
    # Real pressure is converted from Pa to hPa
    assert weather[1].pressure == pytest.approx(985.0)


def test_normal_values_roundtrip(tmp_path):
    """A plain row parses to the values written."""
    path = _write_epw(tmp_path / "plain.epw", [_row(1)])
    df, meta = read_epw(path)
    row = df.iloc[0]
    assert row["temp_air"] == pytest.approx(25.0)
    assert row["atmospheric_pressure"] == pytest.approx(101325.0)
    assert meta["latitude"] == pytest.approx(38.0)
