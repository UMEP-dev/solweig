# Core Functions

## calculate

::: solweig.calculate
    options:
      show_source: false
      heading_level: 3

---

## validate_inputs

::: solweig.api.validate_inputs
    options:
      show_source: false
      heading_level: 3

---

## SurfaceData.prepare

The most common entry point for building a [`SurfaceData`](dataclasses.md#surfacedata)
from rasters or in-memory arrays — see the class page for the full method
list. Documented here because every quick-start uses it before `calculate()`.

::: solweig.SurfaceData.prepare
    options:
      show_source: false
      heading_level: 3

---

## GPU helpers

Runtime toggles for the wgpu compute path. The GPU is enabled by default
when available; these helpers let scripts inspect or override that.

### is_gpu_available

::: solweig.is_gpu_available
    options:
      show_source: false
      heading_level: 4

### get_compute_backend

::: solweig.get_compute_backend
    options:
      show_source: false
      heading_level: 4

### disable_gpu

::: solweig.disable_gpu
    options:
      show_source: false
      heading_level: 4

### get_gpu_limits

::: solweig.get_gpu_limits
    options:
      show_source: false
      heading_level: 4
