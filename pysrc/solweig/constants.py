"""
Physical constants and default parameters for SOLWEIG.

This module consolidates all physical constants and default parameter values
to eliminate duplication across the codebase and provide clear documentation
with proper references.
"""

# =============================================================================
# Physical Constants
# =============================================================================

# Stefan-Boltzmann constant (W/m²/K⁴)
# Used for blackbody radiation calculations: E = σ × T⁴
# Legacy UMEP value. CODATA 2018 exact: 5.670374419e-8. Kept for upstream parity.
SBC = 5.67051e-8

# Kelvin to Celsius conversion offset
# Used for temperature unit conversions
KELVIN_OFFSET = 273.15

# Minimum sun elevation for shadow calculations (degrees)
# Below this threshold, shadows are not computed (negligible solar radiation)
MIN_SUN_ELEVATION_DEG = 3.0

# Effective solar constant (W/m²)
# Reference: Crawford and Duchon (1999)
SOLAR_CONSTANT = 1370.0


# =============================================================================
# View Factor Constants
# =============================================================================
# View factors represent the fraction of radiation leaving one surface
# that is intercepted by another surface. For a human body modeled as
# a cylinder or cube, these factors depend on posture.
#
# Reference: Höppe (1992) - "The physiological equivalent temperature"
# =============================================================================

# Standing posture view factors (cylindrical model)
# Human body modeled as a standing cylinder
F_UP_STANDING = 0.06  # View factor to sky/ground from top/bottom
F_SIDE_STANDING = 0.22  # View factor from each of 4 cardinal directions (N, E, S, W)
F_CYL_STANDING = 0.28  # Cylindrical projection factor for direct beam radiation

# Sitting posture view factors (cubic model)
# Human body modeled as a sitting cube with equal area on all 6 sides
F_UP_SITTING = 1.0 / 6.0  # View factor to sky/ground (1/6 per side)
F_SIDE_SITTING = 1.0 / 6.0  # View factor from each of 4 cardinal directions (1/6 per side)
F_CYL_SITTING = 0.20  # Cylindric projection factor for sitting posture

# Height of the body's centre of gravity (m), by posture. This is the height
# the ground view factor is evaluated at (UMEP `Posture.*.height`); it is not
# the person's stature, which only PET uses.
POSTURE_HEIGHT_STANDING = 1.1
POSTURE_HEIGHT_SITTING = 0.75


# =============================================================================
# Default Physical Parameters
# =============================================================================
# These are common default values used when not specified by the user
# or when materials/physics config is not provided.
# =============================================================================

# Default wall properties
DEFAULT_ALBEDO_WALL = 0.20  # Wall albedo (reflectance)
DEFAULT_EMIS_WALL = 0.90  # Wall emissivity (longwave radiation)
DEFAULT_TG_WALL = 0.0  # Wall temperature deviation from air temperature (K)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Physical constants
    "SBC",
    "KELVIN_OFFSET",
    "MIN_SUN_ELEVATION_DEG",
    # View factors
    "F_UP_STANDING",
    "F_SIDE_STANDING",
    "F_CYL_STANDING",
    "F_UP_SITTING",
    "F_SIDE_SITTING",
    "F_CYL_SITTING",
    # Defaults
    "DEFAULT_ALBEDO_WALL",
    "DEFAULT_EMIS_WALL",
    "DEFAULT_TG_WALL",
    "SOLAR_CONSTANT",
]
