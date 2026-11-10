"""The summary's hour totals assume a contiguous series; say so when it is not one.

Sun hours, shade hours and UTCI exceedance hours add the gap between the first
two weather records for every timestep. Keeping one record per day therefore
counts 24 h per step. The totals are left as computed; a warning names the
problem.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from solweig import Weather, timeseries


def _series(hours_from_start, **kw):
    base = datetime(2024, 7, 15, 12, 0)
    return [
        Weather(datetime=base + timedelta(hours=h), ta=25.0, rh=50.0, global_rad=700.0, **kw) for h in hours_from_start
    ]


@pytest.fixture
def warnings(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(timeseries.logger, "warning", messages.append)
    return messages


def test_contiguous_hourly_series_is_silent(warnings):
    timeseries._warn_if_summary_hours_overcount(_series([0, 1, 2, 3]), 1.0)
    assert warnings == []


def test_contiguous_ten_minute_series_is_silent(warnings):
    series = _series([0, 1 / 6, 2 / 6], timestep_minutes=10.0)
    timeseries._warn_if_summary_hours_overcount(series, 1 / 6)
    assert warnings == []


def test_one_record_per_day_warns_with_the_factor(warnings):
    timeseries._warn_if_summary_hours_overcount(_series([0, 24, 48, 72]), 24.0)
    assert len(warnings) == 1
    assert "24 h apart" in warnings[0]
    assert "factor of 24" in warnings[0]


def test_uneven_gaps_warn(warnings):
    timeseries._warn_if_summary_hours_overcount(_series([0, 1, 2, 26, 27]), 1.0)
    assert len(warnings) == 1
    assert "not evenly spaced" in warnings[0]
