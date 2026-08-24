"""SOLWEIG error types for actionable error messages.

These exceptions provide structured information about what went wrong
and how to fix it, rather than generic error messages.

Example:
    try:
        summary = solweig.calculate(surface, weather, location, output_dir="out/")
    except solweig.GridShapeMismatch as e:
        print(f"Grid '{e.field}' has wrong shape: expected {e.expected}, got {e.got}")
    except solweig.MissingPrecomputedData as e:
        print(f"Missing data: {e}")
"""

from __future__ import annotations


class SolweigError(Exception):
    """Base class for all SOLWEIG errors."""

    pass


class InvalidSurfaceData(SolweigError):
    """Raised when surface data is invalid or inconsistent.

    Attributes:
        message: Human-readable error description.
        field: Name of the problematic field (e.g., "cdsm", "dem").
        expected: What was expected (optional).
        got: What was actually provided (optional).
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        expected: str | None = None,
        got: str | None = None,
    ):
        self.field = field
        self.expected = expected
        self.got = got
        super().__init__(message)


class GridShapeMismatch(InvalidSurfaceData):
    """Raised when grid shapes don't match the DSM.

    All surface grids (CDSM, DEM, TDSM, land_cover, etc.) must have
    the same shape as the DSM.

    Example:
        >>> surface = SurfaceData(dsm=np.ones((100, 100)), cdsm=np.ones((50, 50)))
        GridShapeMismatch: Grid shape mismatch for 'cdsm':
          Expected: (100, 100) (matching DSM)
          Got: (50, 50)
    """

    def __init__(self, field: str, expected_shape: tuple, actual_shape: tuple):
        message = (
            f"Grid shape mismatch for '{field}':\n"
            f"  Expected: {expected_shape} (matching DSM)\n"
            f"  Got: {actual_shape}\n"
            "Ensure all surface grids have the same dimensions as the DSM."
        )
        super().__init__(message, field=field, expected=str(expected_shape), got=str(actual_shape))
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class MissingPrecomputedData(SolweigError):
    """Raised when required precomputed data is not available.

    Some features require precomputed data (e.g., anisotropic sky needs
    shadow matrices). This error explains what's missing and how to fix it.

    Attributes:
        what: Description of the missing data.
        suggestion: How to fix the issue (optional).
    """

    def __init__(self, what: str, suggestion: str | None = None):
        self.what = what
        self.suggestion = suggestion
        message = f"Missing precomputed data: {what}"
        if suggestion:
            message += f"\n{suggestion}"
        super().__init__(message)


class StalePrecomputedData(SolweigError):
    """Raised when a cached SVF/shadow dataset was computed for a different raster.

    A prepared surface directory can accumulate SVF and shadow-matrix caches
    from earlier runs (a different extent, resolution, or study area). Loading
    such a cache against the current DSM would fail later with a confusing
    grid-shape error, so the mismatch is reported at load time with the
    offending cache path.

    Attributes:
        what: Which cached dataset mismatched (e.g., "SVF").
        cache_dir: Directory the stale cache was loaded from.
        expected_shape: Grid shape of the current DSM.
        actual_shape: Grid shape the cache was computed for.
    """

    def __init__(self, what: str, cache_dir: str, expected_shape: tuple, actual_shape: tuple):
        self.what = what
        self.cache_dir = cache_dir
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        message = (
            f"Cached {what} in {cache_dir} was computed for a different raster:\n"
            f"  Expected: {expected_shape} (matching this surface's DSM)\n"
            f"  Got: {actual_shape}\n"
            "The cache is stale (from a different extent, resolution, or study area). "
            "Delete that cache directory (or the 'svf' folder inside the prepared "
            "surface directory) and re-run SurfaceData.prepare() to regenerate it."
        )
        super().__init__(message)


class WeatherDataError(SolweigError):
    """Raised when weather data is invalid.

    Attributes:
        field: The problematic weather field (e.g., "ta", "rh").
        value: The invalid value.
        reason: Why the value is invalid.
    """

    def __init__(self, field: str, value: float | str, reason: str | None = None):
        self.field = field
        self.value = value
        self.reason = reason
        message = f"Invalid weather data for '{field}': {value}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class ComputationCancelled(SolweigError):
    """Raised when a long-running computation is cancelled by the user.

    Raised on cancellation (e.g. QGIS feedback ``isCanceled()``) so partial
    results are never persisted as valid caches. Callers that treat
    cancellation as a graceful stop should catch this and return.

    Attributes:
        stage: The computation stage that was cancelled (e.g., "SVF").
    """

    def __init__(self, stage: str):
        self.stage = stage
        super().__init__(f"Computation cancelled by user during {stage}")


class ConfigurationError(SolweigError):
    """Raised when configuration is invalid or inconsistent.

    Attributes:
        parameter: The problematic parameter name.
        reason: Why the configuration is invalid.
    """

    def __init__(self, parameter: str, reason: str):
        self.parameter = parameter
        self.reason = reason
        message = f"Invalid configuration for '{parameter}': {reason}"
        super().__init__(message)
