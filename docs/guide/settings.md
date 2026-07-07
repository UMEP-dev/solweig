# Settings: how `calculate()` resolves configuration

`solweig.calculate()` accepts configuration from three sources:

1. **Per-call keyword arguments** — `use_anisotropic_sky=True`,
   `wall_material="brick"`, `conifer=False`, etc.
2. **A `ModelConfig` object** passed as the `config=` keyword.
3. **Defaults** baked into the library (e.g. anisotropic sky on,
   ``max_shadow_distance_m=1000.0``).

When the same setting is specified at more than one level, the
**precedence is: per-call kwarg > `ModelConfig` > default**. A
``None`` at any level means "inherit from the next level down."

This is implemented by
[`solweig.models.settings.Settings.resolve()`](https://github.com/UMEP-dev/solweig/blob/main/pysrc/solweig/models/settings.py).
You usually don't construct `Settings` directly — `calculate()` does
it internally — but knowing the rules helps reason about which value
"wins" when sources disagree.

## Example

```python
import solweig
from solweig.models.config import HumanParams, ModelConfig

# A reusable config: anisotropic off, conservative shadow distance.
cfg = ModelConfig(
    use_anisotropic_sky=False,
    max_shadow_distance_m=500.0,
    human=HumanParams(abs_k=0.6),  # custom shortwave absorption
)

# Call A — pure config:
solweig.calculate(surface, weather, location, config=cfg, output_dir="a/")
# Effective: aniso=False, max_shadow=500, abs_k=0.6.

# Call B — kwargs override one field:
solweig.calculate(
    surface, weather, location,
    config=cfg,
    use_anisotropic_sky=True,   # overrides cfg.use_anisotropic_sky
    output_dir="b/",
)
# Effective: aniso=True (kwarg), max_shadow=500 (config), abs_k=0.6 (config).

# Call C — kwarg explicitly None falls through to config:
solweig.calculate(
    surface, weather, location,
    config=cfg,
    use_anisotropic_sky=None,   # no override, defer to config
    output_dir="c/",
)
# Effective: same as Call A — aniso=False.
```

## The fields `Settings` carries

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| ``use_anisotropic_sky`` | bool | True | Perez sky model |
| ``conifer`` | bool | False | Treat all vegetation as evergreen (leaf-on year-round) |
| ``wall_material`` | str \| None | None | "brick", "concrete", "wood", "cobblestone", or None to use bundled wall defaults |
| ``max_shadow_distance_m`` | float | 1000.0 | Caps horizontal shadow ray reach + tile overlap |
| ``human`` | HumanParams | HumanParams() | Body geometry + absorption coefficients |
| ``physics`` | SimpleNamespace \| None | None (lazy-loaded from bundled JSON) | Tree settings, posture |
| ``materials`` | SimpleNamespace \| None | None (lazy-loaded from bundled JSON) | Land-cover albedo / emissivity / thermal coefficients |
| ``use_ground_scheme`` | bool | False | UMEP 2026a force-restore/OHM ground surface temperature (opt-in) |
| ``use_outgoing_longwave`` | bool | False | UMEP 2026a solid-angle outgoing longwave march (opt-in) |

``physics`` and ``materials`` are intentionally kept as ``None`` after
``Settings.resolve()`` — the underlying JSON files are only loaded
when downstream code actually needs them. Call
``.with_loaded_defaults()`` to force the load explicitly.

## The 2026a ground-surface scheme (opt-in)

``use_ground_scheme`` and ``use_outgoing_longwave`` activate the UMEP 2026a
ground-surface formulation: a prognostic force-restore/OHM surface
temperature and a solid-angle outgoing longwave model (see
[the ground temperature spec](../physics/ground_temperature.md) for the
physics). Rules:

- Both flags must currently be enabled together; enabling one without the
  other raises ``ConfigurationError``.
- The surface must have a land-cover grid (ground classes 0/1/2/5/6/7).
- Tiled processing is not yet supported with the scheme; rasters must fit
  in a single tile.
- The defaults are off, and stay off until the scheme has its own
  validation entry — the validated baseline physics is unchanged unless
  you opt in.

```python
summary = solweig.calculate(
    surface, weather, location,
    use_ground_scheme=True,
    use_outgoing_longwave=True,
    output_dir="out/",
)
```

## Why a separate `Settings` exists

Before 0.1.0b85, the configuration merge happened inline at the top of
``api._calculate_single`` — a 50-line block with one `if effective_X is
None` chain per field. It was correct but hard to test in isolation and
hostile to extension.

`Settings.resolve()` packages the merge into a single typed function
with explicit semantics, comprehensive tests
([`tests/test_settings_merge.py`](https://github.com/UMEP-dev/solweig/blob/main/tests/test_settings_merge.py)),
and a frozen dataclass for the resolved result. Adding a new setting
is now a one-line dataclass field + one line in `resolve()`.

If you need to inspect the effective settings for a call, you can
build them yourself:

```python
from solweig.models.settings import Settings

s = Settings.resolve(config=my_config, use_anisotropic_sky=True)
print(s.max_shadow_distance_m)  # 1000.0 (default)
print(s.use_anisotropic_sky)     # True (kwarg)
```
