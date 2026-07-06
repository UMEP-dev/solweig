"""
Shadow computation helpers.

Provides ``compute_transmissivity()`` — the seasonal leaf-on/off
transmissivity look-up used by ``calculate_core_fused()``.
"""

from __future__ import annotations

from types import SimpleNamespace


def compute_transmissivity(
    doy: int,
    physics: SimpleNamespace | None = None,
    conifer: bool = False,
) -> float:
    """
    Compute vegetation transmissivity based on day of year and leaf status.

    Implements seasonal leaf on/off logic for deciduous vegetation.
    During leaf-on season, vegetation transmits less light (low psi ~0.03).
    During leaf-off season (winter), bare branches transmit more light (psi ~0.5).

    Args:
        doy: Day of year (1-366)
        physics: Physics params from load_physics() containing Tree_settings.
            If provided, reads Transmissivity, First_day_leaf, Last_day_leaf.
        conifer: Override to treat vegetation as conifer (always leaf-on).

    Returns:
        Transmissivity value:
        - 0.03 (default) during leaf-on period
        - 0.5 during leaf-off period (deciduous trees in winter)

    Reference:
        Lindberg et al. (2008) - SOLWEIG vegetation transmissivity model
    """
    # Default values for deciduous vegetation, matching the shipped
    # default_params.json Tree_settings (First_day_leaf 97 / Last_day_leaf 300)
    # so param-less calls agree with param-loaded runs.
    transmissivity = 0.03
    transmissivity_leafoff = 0.5
    first_day = 97  # ~April 7
    last_day = 300  # ~October 27
    is_conifer = conifer

    # Override from physics params if provided
    if physics is not None and hasattr(physics, "Tree_settings"):
        ts = physics.Tree_settings.Value
        transmissivity = getattr(ts, "Transmissivity", 0.03)
        transmissivity_leafoff = getattr(ts, "Transmissivity_leafoff", 0.5)
        first_day = int(getattr(ts, "First_day_leaf", 97))
        last_day = int(getattr(ts, "Last_day_leaf", 300))
        # Note: Conifer flag may not be in all params files
        is_conifer = conifer or getattr(ts, "Conifer", False)

    # Determine leaf on/off
    if is_conifer:
        leaf_on = True
    elif first_day > last_day:
        # Wraps around year end (southern hemisphere or unusual dates)
        leaf_on = doy > first_day or doy < last_day
    else:
        # Normal case: leaves on between first_day and last_day
        leaf_on = first_day < doy < last_day

    return transmissivity if leaf_on else transmissivity_leafoff
