from __future__ import annotations

import math

import numpy as np

from ..constants import SOLAR_CONSTANT
from . import sun_distance


def clearnessindex_2013b(
    zen: float, jday: int, Ta: float, RH: float, radG: float, location: dict[str, float], P: float
) -> tuple[float, float, float, float, float]:
    """Clearness Index at the Earth's surface calculated from Crawford and Duchon 1999

    :param zen: zenith angle in radians
    :param jday: day of year
    :param Ta: air temperature in degrees C
    :param RH: relative humidity as a fraction (0-1)
    :param radG: global shortwave radiation
    :param location: dictionary including lat, lon and alt
    :param P: station pressure in hPa, or -999.0 for the 1013 mb standard atmosphere
    :return: (I0, CI, Kt, I0et, CIuncorr)

    Unit note: the Crawford & Duchon transmission formula takes pressure in
    millibars, and 1 hPa == 1 mb, so ``P`` is used directly. Classic UMEP
    passes ``P`` in kPa from its met format and scales by 10 inside this
    function; ``Weather.pressure`` here is hPa, so applying UMEP's x10 would
    inflate p to ~10000 mb and underestimate clear-sky irradiance by ~20-25%
    (the refactored umep-core package inherits exactly that bug by feeding
    hPa to the kPa-convention function).
    """

    p = 1013.0 if P == -999.0 else P  # Pressure in millibars (1 hPa == 1 mb)

    if RH <= 0:
        RH = 0.01  # Guard against log(0); physically RH > 0

    # Sun at or below the horizon: clearness index is undefined. This is the
    # same guard applied further down (log_arg < 0.01), hoisted before the
    # transmission math so night calls don't compute NaN powers of a negative
    # optical air mass (RuntimeWarning noise). Return values are identical.
    zen_deg_early = zen / np.pi * 180
    if 90 - zen_deg_early < 0.01:
        return 0.0, float("Inf"), 0.0, 0.0, 0.0

    Itoa = SOLAR_CONSTANT
    D = sun_distance.sun_distance(jday)  # irradiance differences due to Sun-Earth distances
    m = 35.0 * np.cos(zen) * ((1224.0 * (np.cos(zen) ** 2) + 1) ** (-1 / 2.0))  # optical air mass at p=1013
    Trpg = (
        1.021 - 0.084 * (m * (0.000949 * p + 0.051)) ** 0.5
    )  # Transmission coefficient for Rayliegh scattering and permanent gases

    # empirical constant depending on latitude
    abs_latitude = abs(location["latitude"])
    if abs_latitude < 10.0:
        G_coeffs = [3.37, 2.85, 2.80, 2.64]
    elif abs_latitude < 20.0:
        G_coeffs = [2.99, 3.02, 2.70, 2.93]
    elif abs_latitude < 30.0:
        G_coeffs = [3.60, 3.00, 2.98, 2.93]
    elif abs_latitude < 40.0:
        G_coeffs = [3.04, 3.11, 2.92, 2.94]
    elif abs_latitude < 50.0:
        G_coeffs = [2.70, 2.95, 2.77, 2.71]
    elif abs_latitude < 60.0:
        G_coeffs = [2.52, 3.07, 2.67, 2.93]
    elif abs_latitude < 70.0:
        G_coeffs = [1.76, 2.69, 2.61, 2.61]
    elif abs_latitude < 80.0:
        G_coeffs = [1.60, 1.67, 2.24, 2.63]
    else:  # abs_latitude >= 80.0
        G_coeffs = [1.11, 1.44, 1.94, 2.02]

    if jday > 335 or jday <= 60:
        G: float = G_coeffs[0]
    elif jday > 60 and jday <= 152:
        G = G_coeffs[1]
    elif jday > 152 and jday <= 244:
        G = G_coeffs[2]
    else:  # jday > 244 and jday <= 335
        G = G_coeffs[3]

    # dewpoint calculation
    a2 = 17.27
    b2 = 237.7
    Td = (b2 * (((a2 * Ta) / (b2 + Ta)) + np.log(RH))) / (a2 - (((a2 * Ta) / (b2 + Ta)) + np.log(RH)))
    Td = (Td * 1.8) + 32  # Dewpoint (F)
    u = np.exp(0.1133 - np.log(G + 1) + 0.0393 * Td)  # Precipitable water
    Tw = 1 - 0.077 * ((u * m) ** 0.3)  # Transmission coefficient for water vapor
    Tar = 0.935**m  # Transmission coefficient for aerosols

    I0 = Itoa * np.cos(zen) * Trpg * Tw * D * Tar
    if abs(zen) > np.pi / 2:
        I0 = 0
    # b=I0==abs(zen)>np.pi/2
    # I0(b==1)=0
    # clear b;
    if not (np.isreal(I0)):
        I0 = 0

    zen_deg = zen / np.pi * 180
    log_arg = 90 - zen_deg
    if log_arg < 0.01:
        # Sun at or below horizon — clearness index undefined
        return 0.0, float("Inf"), 0.0, 0.0, 0.0

    corr = 0.1473 * np.log(log_arg) + 0.3454  # 20070329

    if I0 == 0:
        return 0.0, float("Inf"), 0.0, 0.0, 0.0

    CIuncorr = radG / I0
    CI = CIuncorr + (1 - corr)
    I0et = Itoa * np.cos(zen) * D  # extra terrestial solar radiation
    Kt = radG / I0et if I0et != 0 else 0.0
    if math.isnan(CI):
        CI = float("Inf")

    return I0, CI, Kt, I0et, CIuncorr
