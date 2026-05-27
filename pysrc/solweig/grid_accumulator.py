"""Per-pixel summary grid accumulator for the timeseries loop.

Extracted from `summary.py` so that the public-facing
:class:`~solweig.summary.TimeseriesSummary` (the dataclass returned by
:func:`solweig.calculate`) lives in its own file and the much larger
:class:`GridAccumulator` (an internal helper consumed only by
:mod:`solweig.timeseries` and :mod:`solweig.tiling`) does not push the
summary module over the 700-line hot-file threshold.

The class is re-exported from :mod:`solweig.summary` for backwards
compatibility — existing callers do not need to change their imports.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .summary import Timeseries, TimeseriesSummary

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .models.results import SolweigResult
    from .models.weather import Weather


class GridAccumulator:
    """Accumulates per-pixel summary grids during the timeseries loop.

    Used identically by both ``timeseries.py`` and ``tiling.py``.
    All internal accumulators use float64 for numerical stability.

    For very large rasters, pass *memmap_dir* to back arrays with
    memory-mapped files so the OS can page them to disk transparently.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        heat_thresholds_day: list[float],
        heat_thresholds_night: list[float],
        timestep_hours: float,
        memmap_dir: Path | None = None,
        track_scalars: bool = True,
    ) -> None:
        self.shape = shape
        self._track_scalars = track_scalars
        self.heat_thresholds_day = list(heat_thresholds_day)
        self.heat_thresholds_night = list(heat_thresholds_night)
        self.timestep_hours = timestep_hours
        self._memmap_dir = memmap_dir

        def _zeros(name: str, dtype=np.float64) -> NDArray:
            if memmap_dir is not None:
                fp = memmap_dir / f"acc_{name}.dat"
                arr = np.memmap(fp, dtype=dtype, mode="w+", shape=shape)
                arr[:] = 0
                return arr
            return np.zeros(shape, dtype=dtype)

        def _full(name: str, fill: float, dtype=np.float64) -> NDArray:
            if memmap_dir is not None:
                fp = memmap_dir / f"acc_{name}.dat"
                arr = np.memmap(fp, dtype=dtype, mode="w+", shape=shape)
                arr[:] = fill
                return arr
            return np.full(shape, fill, dtype=dtype)

        # Tmrt accumulators
        self._tmrt_sum = _zeros("tmrt_sum")
        self._tmrt_count = _zeros("tmrt_count", dtype=np.int32)
        self._tmrt_max = _full("tmrt_max", -np.inf)
        self._tmrt_min = _full("tmrt_min", np.inf)
        self._tmrt_day_sum = _zeros("tmrt_day_sum")
        self._tmrt_day_count = _zeros("tmrt_day_count", dtype=np.int32)
        self._tmrt_night_sum = _zeros("tmrt_night_sum")
        self._tmrt_night_count = _zeros("tmrt_night_count", dtype=np.int32)

        # UTCI accumulators
        self._utci_sum = _zeros("utci_sum")
        self._utci_count = _zeros("utci_count", dtype=np.int32)
        self._utci_max = _full("utci_max", -np.inf)
        self._utci_min = _full("utci_min", np.inf)
        self._utci_day_sum = _zeros("utci_day_sum")
        self._utci_day_count = _zeros("utci_day_count", dtype=np.int32)
        self._utci_night_sum = _zeros("utci_night_sum")
        self._utci_night_count = _zeros("utci_night_count", dtype=np.int32)

        # Sun/shade
        self._sun_hours = _zeros("sun_hours")
        self._shade_hours = _zeros("shade_hours")
        self._shadow_seen = False

        # UTCI threshold exceedance — combine all unique thresholds
        all_thresholds = sorted(set(heat_thresholds_day) | set(heat_thresholds_night))
        self._utci_hours_above: dict[float, NDArray] = {t: _zeros(f"utci_above_{t}") for t in all_thresholds}
        self._day_thresholds_set = set(heat_thresholds_day)
        self._night_thresholds_set = set(heat_thresholds_night)

        # Pre-allocated scratch buffers (avoids per-call allocation in update).
        # Only needed for the non-tiled update() path; memmap-mode uses
        # update_tile() which operates directly on memmap slices.
        if memmap_dir is None:
            self._scratch_valid = np.empty(shape, dtype=np.bool_)
            self._scratch_utci_valid = np.empty(shape, dtype=np.bool_)
            self._scratch_threshold_mask: NDArray[np.bool_] | None = np.empty(shape, dtype=np.bool_)
        else:
            self._scratch_valid = None
            self._scratch_utci_valid = None
            self._scratch_threshold_mask = None

        # Per-timestep tile accumulation state (used by begin_timestep/commit_timestep)
        self._tile_tmrt_sum = 0.0
        self._tile_tmrt_count = 0
        self._tile_utci_sum = 0.0
        self._tile_utci_count = 0
        self._tile_shadow_sunlit_sum = 0.0
        self._tile_shadow_valid_count = 0

        # Counters
        self._n_timesteps = 0
        self._n_daytime = 0
        self._n_nighttime = 0

        # Per-timestep scalar accumulators (lists, finalized to arrays)
        self._ts_datetime: list[_dt.datetime] = []
        self._ts_ta: list[float] = []
        self._ts_rh: list[float] = []
        self._ts_ws: list[float] = []
        self._ts_global_rad: list[float] = []
        self._ts_direct_rad: list[float] = []
        self._ts_diffuse_rad: list[float] = []
        self._ts_sun_altitude: list[float] = []
        self._ts_tmrt_mean: list[float] = []
        self._ts_utci_mean: list[float] = []
        self._ts_sun_fraction: list[float] = []
        self._ts_diffuse_fraction: list[float] = []
        self._ts_clearness_index: list[float] = []
        self._ts_is_daytime: list[bool] = []

    def update(
        self,
        result: SolweigResult,
        weather: Weather,
        compute_utci_fn: Callable,
    ) -> NDArray[np.floating]:
        """Ingest one timestep. Must be called BEFORE arrays are freed.

        Not available when *memmap_dir* was set — use :meth:`update_tile` instead.

        Returns:
            The computed UTCI grid (full tile shape, float32). Callers can
            slice this to the core region for per-timestep output without
            recomputing.
        """
        assert self._scratch_valid is not None, (
            "update() requires scratch buffers (memmap_dir must be None); "
            "use update_tile() for memmap-backed accumulators"
        )
        tmrt = result.tmrt
        np.isfinite(tmrt, out=self._scratch_valid)
        valid = self._scratch_valid
        is_day = weather.is_daytime
        dt = self.timestep_hours

        # --- Tmrt stats ---
        # In-place ufuncs with `where=` avoid allocating a full-grid
        # temporary per call.
        np.add(self._tmrt_sum, tmrt, out=self._tmrt_sum, where=valid)
        self._tmrt_count += valid
        np.fmax(self._tmrt_max, tmrt, out=self._tmrt_max, where=valid)
        np.fmin(self._tmrt_min, tmrt, out=self._tmrt_min, where=valid)

        if is_day:
            np.add(self._tmrt_day_sum, tmrt, out=self._tmrt_day_sum, where=valid)
            self._tmrt_day_count += valid
        else:
            np.add(self._tmrt_night_sum, tmrt, out=self._tmrt_night_sum, where=valid)
            self._tmrt_night_count += valid

        # --- UTCI ---
        utci = compute_utci_fn(tmrt, weather.ta, weather.rh, weather.ws)
        np.isfinite(utci, out=self._scratch_utci_valid)
        self._scratch_utci_valid &= valid
        utci_valid = self._scratch_utci_valid

        np.add(self._utci_sum, utci, out=self._utci_sum, where=utci_valid)
        self._utci_count += utci_valid
        np.fmax(self._utci_max, utci, out=self._utci_max, where=utci_valid)
        np.fmin(self._utci_min, utci, out=self._utci_min, where=utci_valid)

        if is_day:
            np.add(self._utci_day_sum, utci, out=self._utci_day_sum, where=utci_valid)
            self._utci_day_count += utci_valid
        else:
            np.add(self._utci_night_sum, utci, out=self._utci_night_sum, where=utci_valid)
            self._utci_night_count += utci_valid

        # --- Sun/shade hours ---
        # `(1.0 - shadow) * dt` kept in allocation form to preserve bit-exact
        # parity with :meth:`update_tile` (test_grid_accumulators_match_exactly).
        sun_fraction = np.nan
        if result.shadow is not None:
            self._shadow_seen = True
            if is_day:
                self._sun_hours += np.where(valid, result.shadow * dt, 0.0)
                self._shade_hours += np.where(valid, (1.0 - result.shadow) * dt, 0.0)
                n_valid = valid.sum()
                sun_fraction = float(result.shadow[valid].sum() / n_valid) if n_valid > 0 else np.nan
            else:
                # At night the shadow grid is all-sunlit (no shadows cast) which is
                # physically meaningless — skip accumulation and report 0 sun fraction.
                self._shade_hours += np.where(valid, dt, 0.0)
                sun_fraction = 0.0

        # --- UTCI threshold exceedance ---
        # Reuse the scratch mask to hold `utci_valid & (utci > threshold)`
        # per threshold without allocating two new bool grids per iteration.
        active_thresholds = self._day_thresholds_set if is_day else self._night_thresholds_set
        scratch_mask = self._scratch_threshold_mask
        if scratch_mask is None:
            raise RuntimeError(
                "GridAccumulator.update() requires _scratch_threshold_mask; "
                "it is only None in memmap-backed mode, which must use "
                "update_tile() instead"
            )
        for threshold in active_thresholds:
            acc = self._utci_hours_above[threshold]
            np.greater(utci, threshold, out=scratch_mask)
            np.logical_and(scratch_mask, utci_valid, out=scratch_mask)
            np.add(acc, dt, out=acc, where=scratch_mask)

        self._n_timesteps += 1
        if is_day:
            self._n_daytime += 1
        else:
            self._n_nighttime += 1

        # --- Per-timestep scalar tracking (skipped in tile-outer mode) ---
        if self._track_scalars:
            self._ts_datetime.append(weather.datetime)
            self._ts_ta.append(weather.ta)
            self._ts_rh.append(weather.rh)
            self._ts_ws.append(weather.ws)
            self._ts_global_rad.append(weather.global_rad)
            self._ts_direct_rad.append(weather.direct_rad)
            self._ts_diffuse_rad.append(weather.diffuse_rad)
            self._ts_sun_altitude.append(weather.sun_altitude)
            self._ts_is_daytime.append(is_day)

            n_valid_tmrt = valid.sum()
            self._ts_tmrt_mean.append(float(tmrt[valid].mean()) if n_valid_tmrt > 0 else np.nan)
            n_valid_utci = utci_valid.sum()
            self._ts_utci_mean.append(float(utci[utci_valid].mean()) if n_valid_utci > 0 else np.nan)
            self._ts_sun_fraction.append(sun_fraction)
            if weather.global_rad > 0:
                self._ts_diffuse_fraction.append(weather.diffuse_rad / weather.global_rad)
            else:
                self._ts_diffuse_fraction.append(np.nan)
            self._ts_clearness_index.append(weather.clearness_index)

        return utci

    # ------------------------------------------------------------------
    # Tile-aware accumulation
    # ------------------------------------------------------------------
    # These three methods replace update() in the tiled timeseries path.
    # They avoid assembling a full-raster intermediary: each tile's core
    # result is written directly into the accumulator slices.

    def begin_timestep(self) -> None:
        """Reset per-timestep partial-sum accumulators before processing tiles."""
        self._tile_tmrt_sum = 0.0
        self._tile_tmrt_count = 0
        self._tile_utci_sum = 0.0
        self._tile_utci_count = 0
        self._tile_shadow_sunlit_sum = 0.0
        self._tile_shadow_valid_count = 0

    def update_tile(
        self,
        tile_tmrt: NDArray[np.floating],
        tile_shadow: NDArray[np.floating] | None,
        write_slice: tuple[slice, slice],
        core_slice: tuple[slice, slice],
        weather: Weather,
        compute_utci_fn: Callable,
    ) -> NDArray[np.floating] | None:
        """Accumulate one tile's core results into the full-raster accumulators.

        This performs identical arithmetic to :meth:`update` but operates only
        on the tile's core region, writing into *write_slice* of the internal
        arrays.  No full-raster intermediary is needed.

        Args:
            tile_tmrt: Tile-sized Tmrt array (includes overlap).
            tile_shadow: Tile-sized shadow array, or None.
            write_slice: ``(row_slice, col_slice)`` into the full-raster accumulators.
            core_slice: ``(row_slice, col_slice)`` to extract the non-overlap core
                from the tile-sized arrays.
            weather: Weather for this timestep.
            compute_utci_fn: ``(tmrt, ta, rh, ws) -> utci`` callable.

        Returns:
            UTCI core array (float32) if UTCI was computed, else None.
        """
        ws = write_slice
        tmrt = tile_tmrt[core_slice]
        valid = np.isfinite(tmrt)
        is_day = weather.is_daytime

        # --- Tmrt grid stats ---
        self._tmrt_sum[ws] += np.where(valid, tmrt, 0.0)
        self._tmrt_count[ws] += valid
        np.fmax(self._tmrt_max[ws], np.where(valid, tmrt, -np.inf), out=self._tmrt_max[ws])
        np.fmin(self._tmrt_min[ws], np.where(valid, tmrt, np.inf), out=self._tmrt_min[ws])

        if is_day:
            self._tmrt_day_sum[ws] += np.where(valid, tmrt, 0.0)
            self._tmrt_day_count[ws] += valid
        else:
            self._tmrt_night_sum[ws] += np.where(valid, tmrt, 0.0)
            self._tmrt_night_count[ws] += valid

        # --- UTCI ---
        utci = compute_utci_fn(tmrt, weather.ta, weather.rh, weather.ws)
        utci_valid = np.isfinite(utci) & valid

        self._utci_sum[ws] += np.where(utci_valid, utci, 0.0)
        self._utci_count[ws] += utci_valid
        np.fmax(self._utci_max[ws], np.where(utci_valid, utci, -np.inf), out=self._utci_max[ws])
        np.fmin(self._utci_min[ws], np.where(utci_valid, utci, np.inf), out=self._utci_min[ws])

        if is_day:
            self._utci_day_sum[ws] += np.where(utci_valid, utci, 0.0)
            self._utci_day_count[ws] += utci_valid
        else:
            self._utci_night_sum[ws] += np.where(utci_valid, utci, 0.0)
            self._utci_night_count[ws] += utci_valid

        # --- Sun/shade hours ---
        if tile_shadow is not None:
            self._shadow_seen = True
            shadow = tile_shadow[core_slice]
            if is_day:
                self._sun_hours[ws] += np.where(valid, shadow * self.timestep_hours, 0.0)
                self._shade_hours[ws] += np.where(valid, (1.0 - shadow) * self.timestep_hours, 0.0)
                n_v = int(valid.sum())
                self._tile_shadow_sunlit_sum += float(shadow[valid].sum()) if n_v > 0 else 0.0
                self._tile_shadow_valid_count += n_v
            else:
                self._shade_hours[ws] += np.where(valid, self.timestep_hours, 0.0)

        # --- UTCI threshold exceedance ---
        active_thresholds = self._day_thresholds_set if is_day else self._night_thresholds_set
        for threshold in active_thresholds:
            acc = self._utci_hours_above[threshold]
            acc[ws] += np.where(utci_valid & (utci > threshold), self.timestep_hours, 0.0)

        # --- Partial sums for per-timestep scalar means ---
        n_valid_tmrt = int(valid.sum())
        self._tile_tmrt_sum += float(tmrt[valid].sum()) if n_valid_tmrt > 0 else 0.0
        self._tile_tmrt_count += n_valid_tmrt
        n_valid_utci = int(utci_valid.sum())
        self._tile_utci_sum += float(utci[utci_valid].sum()) if n_valid_utci > 0 else 0.0
        self._tile_utci_count += n_valid_utci

        return utci

    def commit_timestep(self, weather: Weather) -> None:
        """Finalise per-timestep scalar tracking after all tiles are processed.

        Must be called once per timestep, after all :meth:`update_tile` calls.
        """
        is_day = weather.is_daytime

        self._n_timesteps += 1
        if is_day:
            self._n_daytime += 1
        else:
            self._n_nighttime += 1

        self._ts_datetime.append(weather.datetime)
        self._ts_ta.append(weather.ta)
        self._ts_rh.append(weather.rh)
        self._ts_ws.append(weather.ws)
        self._ts_global_rad.append(weather.global_rad)
        self._ts_direct_rad.append(weather.direct_rad)
        self._ts_diffuse_rad.append(weather.diffuse_rad)
        self._ts_sun_altitude.append(weather.sun_altitude)
        self._ts_is_daytime.append(is_day)

        # Spatial means aggregated from tile partial sums
        self._ts_tmrt_mean.append(self._tile_tmrt_sum / self._tile_tmrt_count if self._tile_tmrt_count > 0 else np.nan)
        self._ts_utci_mean.append(self._tile_utci_sum / self._tile_utci_count if self._tile_utci_count > 0 else np.nan)
        if self._tile_shadow_valid_count > 0:
            sun_fraction = self._tile_shadow_sunlit_sum / self._tile_shadow_valid_count
        elif not is_day and self._shadow_seen:
            sun_fraction = 0.0
        else:
            sun_fraction = np.nan
        self._ts_sun_fraction.append(sun_fraction)

        if weather.global_rad > 0:
            self._ts_diffuse_fraction.append(weather.diffuse_rad / weather.global_rad)
        else:
            self._ts_diffuse_fraction.append(np.nan)
        self._ts_clearness_index.append(weather.clearness_index)

    def finalize(self) -> TimeseriesSummary:
        """Compute final summary grids from accumulated state."""

        def _safe_mean(total: NDArray, count: NDArray) -> NDArray[np.floating]:
            with np.errstate(invalid="ignore"):
                out = np.where(count > 0, total / count, np.nan)
            return out.astype(np.float32)

        def _safe_extrema(arr: NDArray, count: NDArray) -> NDArray[np.floating]:
            out = np.where(count > 0, arr, np.nan)
            return out.astype(np.float32)

        sun_hours = (
            self._sun_hours.astype(np.float32) if self._shadow_seen else np.full(self.shape, np.nan, dtype=np.float32)
        )
        shade_hours = (
            self._shade_hours.astype(np.float32) if self._shadow_seen else np.full(self.shape, np.nan, dtype=np.float32)
        )

        utci_hours = {t: arr.astype(np.float32) for t, arr in sorted(self._utci_hours_above.items())}

        # Build per-timestep timeseries
        timeseries = (
            Timeseries(
                datetime=list(self._ts_datetime),
                ta=np.array(self._ts_ta, dtype=np.float32),
                rh=np.array(self._ts_rh, dtype=np.float32),
                ws=np.array(self._ts_ws, dtype=np.float32),
                global_rad=np.array(self._ts_global_rad, dtype=np.float32),
                direct_rad=np.array(self._ts_direct_rad, dtype=np.float32),
                diffuse_rad=np.array(self._ts_diffuse_rad, dtype=np.float32),
                sun_altitude=np.array(self._ts_sun_altitude, dtype=np.float32),
                tmrt_mean=np.array(self._ts_tmrt_mean, dtype=np.float32),
                utci_mean=np.array(self._ts_utci_mean, dtype=np.float32),
                sun_fraction=np.array(self._ts_sun_fraction, dtype=np.float32),
                diffuse_fraction=np.array(self._ts_diffuse_fraction, dtype=np.float32),
                clearness_index=np.array(self._ts_clearness_index, dtype=np.float32),
                is_daytime=np.array(self._ts_is_daytime, dtype=np.bool_),
            )
            if self._n_timesteps > 0
            else None
        )

        return TimeseriesSummary(
            tmrt_mean=_safe_mean(self._tmrt_sum, self._tmrt_count),
            tmrt_max=_safe_extrema(self._tmrt_max, self._tmrt_count),
            tmrt_min=_safe_extrema(self._tmrt_min, self._tmrt_count),
            tmrt_day_mean=_safe_mean(self._tmrt_day_sum, self._tmrt_day_count),
            tmrt_night_mean=_safe_mean(self._tmrt_night_sum, self._tmrt_night_count),
            utci_mean=_safe_mean(self._utci_sum, self._utci_count),
            utci_max=_safe_extrema(self._utci_max, self._utci_count),
            utci_min=_safe_extrema(self._utci_min, self._utci_count),
            utci_day_mean=_safe_mean(self._utci_day_sum, self._utci_day_count),
            utci_night_mean=_safe_mean(self._utci_night_sum, self._utci_night_count),
            sun_hours=sun_hours,
            shade_hours=shade_hours,
            utci_hours_above=utci_hours,
            n_timesteps=self._n_timesteps,
            n_daytime=self._n_daytime,
            n_nighttime=self._n_nighttime,
            shadow_available=self._shadow_seen,
            heat_thresholds_day=self.heat_thresholds_day,
            heat_thresholds_night=self.heat_thresholds_night,
            timeseries=timeseries,
        )
