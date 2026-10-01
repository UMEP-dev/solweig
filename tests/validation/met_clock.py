"""Clock correction for the Gothenburg validation met files.

The three validation sites compare modelled and measured values hour by
hour. For that pairing to be meaningful, three clocks must agree: the
labels on the measured radiation and Tmrt at the point of interest, the
labels on the met forcing that drives the model, and the time at which
the model places the sun for each row.

The model follows the UMEP convention: a row labelled ``h`` is the hourly
mean of the interval ending at ``h`` in local standard time, so the sun is
evaluated at ``h`` minus half a timestep with ``utc_offset = 1`` (CET).
The measurement CSVs follow the same convention: their Kdown curves peak
one hour after the model's sun on every clear day, and lag-fitting the
measured Kdown against the model's sine of solar altitude gives a best
lag of zero at Gustav Adolfs torg and GVC.

The prepared met files do not. Their Kdn at label ``h`` matches the
model's sun at label ``h + 1`` on every clear day (Gustav Adolfs
2005-10-11, Kronenhuset 2005-10-07, GVC 2010-07-12), and the measured
Kdown at label ``h`` matches met Kdn at label ``h - 1``. The met files are
therefore labelled one hour early relative to the hour-ending CET
convention: either hour-starting CET or hour-ending UTC. Left as is, every
timestep is driven by forcing recorded one hour later than its sun
position, which puts the modelled Kdown peak an hour before the measured
one (the symptom Lindberg noticed, 2026-06 and 2026-08).

:func:`load_site_met` loads a prepared met file and adds one hour to each
timestamp so that forcing, sun and measurements share one clock. The
measurements are left untouched. The library itself is not changed; the
correction belongs to these data files.

Diagnostic: lag fits of met Kdn, measured Kdown and the model's sine of
solar altitude at lags of -2 to +2 hours, 2026-10-01.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solweig import Weather

#: Hours to add to the prepared met-file timestamps (see module docstring).
MET_CLOCK_SHIFT_HOURS = 1


def load_site_met(met_file: str | Path) -> list[Weather]:
    """Load a prepared validation met file with its clock corrected.

    The prepared files carry ``imin`` of 1, 7 or 10 for every row, so the
    default hourly resampling (which keeps only ``imin == 0``) is disabled.
    Each timestamp is then shifted by :data:`MET_CLOCK_SHIFT_HOURS`.

    Args:
        met_file: Path to the UMEP-format met file.

    Returns:
        Weather rows labelled hour-ending CET, matching the measurement CSVs.
    """
    from solweig import Weather

    weather = Weather.from_umep_met(met_file, resample_hourly=False)
    shift = timedelta(hours=MET_CLOCK_SHIFT_HOURS)
    for w in weather:
        w.datetime = w.datetime + shift
    return weather
