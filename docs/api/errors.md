# Error Handling

SOLWEIG uses structured exceptions for diagnostic error reporting.

## Exception Hierarchy

```
SolweigError (base)
├── InvalidSurfaceData
│   └── GridShapeMismatch
├── MissingPrecomputedData
├── WeatherDataError
├── ComputationCancelled
└── ConfigurationError
```

## Catching Errors

```python
import solweig
from solweig.errors import GridShapeMismatch, MissingPrecomputedData, SolweigError

try:
    result = solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")
except GridShapeMismatch as e:
    print(f"Grid mismatch: {e.field}")
    print(f"  Expected: {e.expected}")
    print(f"  Got: {e.got}")
except MissingPrecomputedData as e:
    print(f"Missing data: {e}")
    print(f"  Suggestion: {e.suggestion}")
except SolweigError as e:
    print(f"Error: {e}")
```

---

## SolweigError

::: solweig.errors.SolweigError
    options:
      show_source: false
      heading_level: 3

---

## GridShapeMismatch

::: solweig.errors.GridShapeMismatch
    options:
      show_source: false
      heading_level: 3

---

## MissingPrecomputedData

::: solweig.errors.MissingPrecomputedData
    options:
      show_source: false
      heading_level: 3

---

## InvalidSurfaceData

::: solweig.errors.InvalidSurfaceData
    options:
      show_source: false
      heading_level: 3

---

## WeatherDataError

::: solweig.errors.WeatherDataError
    options:
      show_source: false
      heading_level: 3

---

## ComputationCancelled

::: solweig.errors.ComputationCancelled
    options:
      show_source: false
      heading_level: 3

---

## ConfigurationError

::: solweig.errors.ConfigurationError
    options:
      show_source: false
      heading_level: 3

---

## Pre-Flight Validation

Use `validate_inputs()` to identify errors before computation:

```python
from solweig.errors import GridShapeMismatch, MissingPrecomputedData

try:
    warnings = solweig.validate_inputs(surface, location, weather)
    for w in warnings:
        print(f"Warning: {w}")

    result = solweig.calculate(surface, weather=[weather], location=location, output_dir="output/")

except GridShapeMismatch as e:
    print(f"Grid shape mismatch: {e.field}")
except MissingPrecomputedData as e:
    print(f"Missing required data: {e}")
```

This catches shape mismatches, missing data, and other issues prior to SVF computation.
