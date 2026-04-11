"""Timeseries summary and grid accumulation.

Defines :class:`TimeseriesSummary` (the default return type of
:func:`calculate`) and :class:`GridAccumulator` (the
internal helper that builds it incrementally during the loop).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .models.results import SolweigResult
    from .models.surface import SurfaceData
    from .models.weather import Weather


@dataclass
class Timeseries:
    """Per-timestep scalar timeseries extracted during the calculation loop.

    Each field is a 1-D array of length ``n_timesteps``, holding the spatial
    mean (or fraction) for that metric at each timestep.  Useful for plotting
    how conditions evolve over the simulation period.

    Attributes:
        datetime: Timestamp per step.
        ta: Air temperature per step (°C) — from weather input.
        rh: Relative humidity per step (%) — from weather input.
        ws: Wind speed per step (m/s) — from weather input.
        global_rad: Global solar radiation per step (W/m²) — from weather input.
        direct_rad: Direct beam radiation per step (W/m²).
        diffuse_rad: Diffuse radiation per step (W/m²).
        sun_altitude: Sun altitude angle per step (°).
        tmrt_mean: Spatial mean Tmrt per step (°C).
        utci_mean: Spatial mean UTCI per step (°C).
        sun_fraction: Fraction of sunlit pixels per step (0–1). NaN when shadow unavailable.
        diffuse_fraction: Diffuse fraction per step (0–1). 0 = clear sky, 1 = fully overcast.
        clearness_index: Clearness index per step. Higher = clearer sky. 0 at night.
        is_daytime: Day/night flag per step.
    """

    datetime: list[_dt.datetime]
    ta: NDArray[np.floating]
    rh: NDArray[np.floating]
    ws: NDArray[np.floating]
    global_rad: NDArray[np.floating]
    direct_rad: NDArray[np.floating]
    diffuse_rad: NDArray[np.floating]
    sun_altitude: NDArray[np.floating]
    tmrt_mean: NDArray[np.floating]
    utci_mean: NDArray[np.floating]
    sun_fraction: NDArray[np.floating]
    diffuse_fraction: NDArray[np.floating]
    clearness_index: NDArray[np.floating]
    is_daytime: NDArray[np.bool_]


@dataclass
class TimeseriesSummary:
    """Aggregated summary from a SOLWEIG timeseries calculation.

    All grids have the same shape as the input DSM (rows, cols).

    Attributes:
        tmrt_mean: Mean Tmrt across all timesteps (°C).
        tmrt_max: Per-pixel maximum Tmrt (°C).
        tmrt_min: Per-pixel minimum Tmrt (°C).
        tmrt_day_mean: Mean Tmrt during daytime (°C). NaN where no daytime data.
        tmrt_night_mean: Mean Tmrt during nighttime (°C). NaN where no nighttime data.
        utci_mean: Mean UTCI across all timesteps (°C).
        utci_max: Per-pixel maximum UTCI (°C).
        utci_min: Per-pixel minimum UTCI (°C).
        utci_day_mean: Mean UTCI during daytime (°C). NaN where no daytime data.
        utci_night_mean: Mean UTCI during nighttime (°C). NaN where no nighttime data.
        sun_hours: Hours of direct sun per pixel.
        shade_hours: Hours of shade per pixel.
        utci_hours_above: Threshold (°C) → grid of hours exceeding that UTCI value.
        n_timesteps: Total number of timesteps processed.
        n_daytime: Number of daytime timesteps.
        n_nighttime: Number of nighttime timesteps.
        shadow_available: Whether shadow data was available for sun/shade hours.
        heat_thresholds_day: Daytime UTCI thresholds used.
        heat_thresholds_night: Nighttime UTCI thresholds used.
        timeseries: Per-timestep scalar timeseries (spatial means over time).
    """

    # Tmrt summary grids
    tmrt_mean: NDArray[np.floating]
    tmrt_max: NDArray[np.floating]
    tmrt_min: NDArray[np.floating]
    tmrt_day_mean: NDArray[np.floating]
    tmrt_night_mean: NDArray[np.floating]

    # UTCI summary grids
    utci_mean: NDArray[np.floating]
    utci_max: NDArray[np.floating]
    utci_min: NDArray[np.floating]
    utci_day_mean: NDArray[np.floating]
    utci_night_mean: NDArray[np.floating]

    # Sun/shade
    sun_hours: NDArray[np.floating]
    shade_hours: NDArray[np.floating]

    # UTCI threshold exceedance
    utci_hours_above: dict[float, NDArray[np.floating]]

    # Metadata
    n_timesteps: int
    n_daytime: int
    n_nighttime: int
    shadow_available: bool
    heat_thresholds_day: list[float]
    heat_thresholds_night: list[float]

    # Per-timestep scalar timeseries
    timeseries: Timeseries | None = None

    # Surface reference for GeoTIFF output (not shown in repr)
    _surface: SurfaceData | None = field(default=None, repr=False)

    # Output directory where summary GeoTIFFs were saved (not shown in repr)
    _output_dir: Path | None = field(default=None, repr=False)

    def __len__(self) -> int:
        """Return number of timesteps processed."""
        return self.n_timesteps

    def report(self) -> str:
        """Return a human-readable summary report.

        Includes spatial statistics, threshold exceedance, timeseries ranges,
        and links to saved GeoTIFF files when available.

        Returns:
            Multi-line report string.
        """
        if self.n_timesteps == 0:
            return "TimeseriesSummary: 0 timesteps (empty)"

        lines: list[str] = []

        # Period header from timeseries datetimes
        if self.timeseries is not None and self.timeseries.datetime:
            dt0 = self.timeseries.datetime[0]
            dt1 = self.timeseries.datetime[-1]
            lines.append(
                f"SOLWEIG Summary: {self.n_timesteps} timesteps ({self.n_daytime} day, {self.n_nighttime} night)"
            )
            lines.append(f"  Period: {dt0:%Y-%m-%d %H:%M} — {dt1:%Y-%m-%d %H:%M}")
        else:
            lines.append(
                f"SOLWEIG Summary: {self.n_timesteps} timesteps ({self.n_daytime} day, {self.n_nighttime} night)",
            )

        # Tmrt stats
        with np.errstate(invalid="ignore"):
            tmrt_vals = self.tmrt_mean[np.isfinite(self.tmrt_mean)]
            if tmrt_vals.size > 0:
                lines.append(
                    f"  Tmrt  — mean: {tmrt_vals.mean():.1f}°C, "
                    f"range: {np.nanmin(self.tmrt_min):.1f} – {np.nanmax(self.tmrt_max):.1f}°C"
                )

            # UTCI stats
            utci_vals = self.utci_mean[np.isfinite(self.utci_mean)]
            if utci_vals.size > 0:
                lines.append(
                    f"  UTCI  — mean: {utci_vals.mean():.1f}°C, "
                    f"range: {np.nanmin(self.utci_min):.1f} – {np.nanmax(self.utci_max):.1f}°C"
                )

            # Sun/shade
            if self.shadow_available:
                sun_valid = self.sun_hours[np.isfinite(self.sun_hours)]
                if sun_valid.size > 0:
                    lines.append(f"  Sun   — {sun_valid.min():.1f} – {sun_valid.max():.1f} hours")

            # UTCI threshold exceedance (labelled day/night)
            day_set = set(self.heat_thresholds_day)
            night_set = set(self.heat_thresholds_night)
            for threshold, grid in sorted(self.utci_hours_above.items()):
                valid = grid[np.isfinite(grid)]
                if valid.size > 0 and valid.max() > 0:
                    label = ""
                    if threshold in day_set and threshold not in night_set:
                        label = " (day)"
                    elif threshold in night_set and threshold not in day_set:
                        label = " (night)"
                    lines.append(f"  UTCI > {threshold:g}°C{label} — max {valid.max():.1f}h")

        # Timeseries summary
        if self.timeseries is not None:
            ts = self.timeseries
            ta_range = f"{np.nanmin(ts.ta):.1f} – {np.nanmax(ts.ta):.1f}°C"
            lines.append(f"  Ta    — range: {ta_range}")

        # Output file links
        if self._output_dir is not None:
            summary_dir = self._output_dir / "summary"
            if summary_dir.exists():
                tifs = sorted(summary_dir.glob("*.tif"))
                if tifs:
                    lines.append(f"  Summary GeoTIFFs: {summary_dir}/")
                    for tif in tifs:
                        lines.append(f"    {tif.name}")

        # Access hint
        lines.append("")
        lines.append("Tip: per-timestep arrays are in summary.timeseries (e.g. .ta, .tmrt_mean, .utci_mean).")
        lines.append("     Spatial grids are on the summary itself (e.g. .tmrt_mean, .utci_max).")
        if self._output_dir is not None:
            lines.append("     Summary grids are saved as GeoTIFFs above; timeseries arrays are in memory only.")
        else:
            lines.append("     Call summary.to_geotiff(output_dir) to save spatial grids.")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML rendering for Jupyter notebooks."""
        return "<pre>" + self.report().replace("&", "&amp;").replace("<", "&lt;") + "</pre>"

    @classmethod
    def empty(cls) -> TimeseriesSummary:
        """Create an empty summary for zero-timestep runs."""
        z = np.empty((0, 0), dtype=np.float32)
        return cls(
            tmrt_mean=z,
            tmrt_max=z,
            tmrt_min=z,
            tmrt_day_mean=z,
            tmrt_night_mean=z,
            utci_mean=z,
            utci_max=z,
            utci_min=z,
            utci_day_mean=z,
            utci_night_mean=z,
            sun_hours=z,
            shade_hours=z,
            utci_hours_above={},
            n_timesteps=0,
            n_daytime=0,
            n_nighttime=0,
            shadow_available=False,
            heat_thresholds_day=[],
            heat_thresholds_night=[],
        )

    def plot(
        self,
        save_path: str | Path | None = None,
        figsize: tuple[float, float] = (14, 10),
        max_days: int = 5,
    ) -> None:
        """Plot the per-timestep timeseries as a multi-panel figure.

        Requires ``matplotlib``.  If ``save_path`` is provided, the figure
        is saved to that path instead of being shown interactively.

        For long simulations the plot is truncated to the first ``max_days``
        days so that individual diurnal cycles remain readable.

        Args:
            save_path: File path to save the figure (e.g. ``"summary.png"``).
                If None, calls ``plt.show()``.
            figsize: Figure size as ``(width, height)`` in inches.
            max_days: Maximum number of days to display.  If the timeseries
                spans more than this many days, only the first ``max_days``
                are plotted.  Set to ``0`` to plot all data.

        Raises:
            RuntimeError: If no timeseries data is available.
        """
        if self.timeseries is None or self.n_timesteps == 0:
            raise RuntimeError("No timeseries data to plot. Timeseries data is populated automatically by calculate().")

        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib") from None

        ts = self.timeseries
        dates = ts.datetime

        # Truncate to max_days if the timeseries is long
        n = len(dates)
        truncated = False
        if max_days > 0 and n > 1:
            import datetime as _dtmod

            span = dates[-1] - dates[0]
            if span > _dtmod.timedelta(days=max_days):
                cutoff = dates[0] + _dtmod.timedelta(days=max_days)
                # Find the first index past the cutoff
                n = next((i for i, d in enumerate(dates) if d > cutoff), len(dates))
                truncated = True

        if truncated:
            dates = dates[:n]
            ts = Timeseries(
                datetime=dates,
                ta=ts.ta[:n],
                rh=ts.rh[:n],
                ws=ts.ws[:n],
                global_rad=ts.global_rad[:n],
                direct_rad=ts.direct_rad[:n],
                diffuse_rad=ts.diffuse_rad[:n],
                sun_altitude=ts.sun_altitude[:n],
                tmrt_mean=ts.tmrt_mean[:n],
                utci_mean=ts.utci_mean[:n],
                sun_fraction=ts.sun_fraction[:n],
                diffuse_fraction=ts.diffuse_fraction[:n],
                clearness_index=ts.clearness_index[:n],
                is_daytime=ts.is_daytime[:n],
            )

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Helper: shade nighttime regions on any axis
        def _shade_night(ax):
            """Add light grey background for nighttime periods."""
            in_night = False
            night_start = None
            for i, d in enumerate(dates):
                if not ts.is_daytime[i]:
                    if not in_night:
                        night_start = d
                        in_night = True
                else:
                    if in_night:
                        ax.axvspan(night_start, d, alpha=0.08, color="grey", linewidth=0)
                        in_night = False
            # Close final night span
            if in_night and night_start is not None:
                ax.axvspan(night_start, dates[-1], alpha=0.08, color="grey", linewidth=0)

        # Panel 1: Temperature (Ta + Tmrt spatial mean + UTCI spatial mean)
        ax = axes[0]
        _shade_night(ax)
        ax.plot(dates, ts.ta, label="Ta", color="#2196F3", linewidth=1)
        ax.plot(dates, ts.tmrt_mean, label="Tmrt (spatial mean)", color="#F44336", linewidth=1)
        ax.plot(dates, ts.utci_mean, label="UTCI (spatial mean)", color="#FF9800", linewidth=1)
        ax.set_ylabel("Temperature (°C)")
        ax.legend(loc="upper right", fontsize=8)
        title = "Temperature and Thermal Comfort"
        if truncated:
            title += f"  (showing first {max_days} of {self.n_timesteps} timesteps)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Panel 2: Solar radiation
        ax = axes[1]
        _shade_night(ax)
        ax.plot(dates, ts.global_rad, label="Global", color="#FFC107", linewidth=1)
        ax.plot(dates, ts.direct_rad, label="Direct", color="#FF5722", linewidth=1)
        ax.plot(dates, ts.diffuse_rad, label="Diffuse", color="#03A9F4", linewidth=1)
        ax.set_ylabel("Radiation (W/m²)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Solar Radiation")
        ax.grid(True, alpha=0.3)

        # Panel 3: Sun fraction + sun altitude + clearness index
        ax = axes[2]
        _shade_night(ax)
        ax2 = ax.twinx()
        if self.shadow_available:
            ax.fill_between(dates, ts.sun_fraction, alpha=0.3, color="#FFEB3B", label="Sun fraction")
        ax.plot(dates, ts.clearness_index, label="Clearness index", color="#FF9800", linewidth=1, alpha=0.8)
        ax.set_ylabel("Fraction / Index")
        ax.set_ylim(0, max(1.05, float(np.nanmax(ts.clearness_index)) * 1.1))
        ax2.plot(dates, ts.sun_altitude, label="Sun altitude", color="#9C27B0", linewidth=1, alpha=0.7)
        ax2.set_ylabel("Sun altitude (°)")
        ax.set_title("Sun Exposure, Clearness and Solar Geometry")
        # Combine legends
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 4: Weather inputs (RH + wind speed)
        ax = axes[3]
        _shade_night(ax)
        ax.plot(dates, ts.rh, label="RH", color="#4CAF50", linewidth=1)
        ax.set_ylabel("Relative Humidity (%)")
        ax.set_ylim(0, 105)
        ax3 = ax.twinx()
        ax3.plot(dates, ts.ws, label="Wind speed", color="#607D8B", linewidth=1)
        ax3.set_ylabel("Wind speed (m/s)")
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax3.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)
        ax.set_title("Meteorological Inputs")
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=30, ha="right")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def to_geotiff(
        self,
        output_dir: str | Path,
        surface: SurfaceData | None = None,
    ) -> None:
        """Save all summary grids to GeoTIFF files in ``output_dir/summary/``.

        Args:
            output_dir: Base output directory.
            surface: SurfaceData for CRS/transform metadata. Falls back to
                the internal ``_surface`` reference if not provided.
        """
        from . import io

        surface = surface or self._surface

        summary_dir = Path(output_dir) / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Resolve geo-referencing from surface
        transform: list[float] | None = None
        crs_wkt: str = ""
        if surface is not None:
            if surface._geotransform is not None:
                transform = surface._geotransform
            if surface._crs_wkt is not None:
                crs_wkt = surface._crs_wkt
        if transform is None:
            transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]

        def _save(name: str, arr: NDArray[np.floating]) -> None:
            if arr.size == 0:
                return
            io.save_raster(
                out_path_str=str(summary_dir / f"{name}.tif"),
                data_arr=arr,
                trf_arr=transform,
                crs_wkt=crs_wkt,
                no_data_val=np.nan,
            )

        # Tmrt grids
        _save("tmrt_mean", self.tmrt_mean)
        _save("tmrt_max", self.tmrt_max)
        _save("tmrt_min", self.tmrt_min)
        _save("tmrt_day_mean", self.tmrt_day_mean)
        _save("tmrt_night_mean", self.tmrt_night_mean)

        # UTCI grids
        _save("utci_mean", self.utci_mean)
        _save("utci_max", self.utci_max)
        _save("utci_min", self.utci_min)
        _save("utci_day_mean", self.utci_day_mean)
        _save("utci_night_mean", self.utci_night_mean)

        # Sun/shade
        _save("sun_hours", self.sun_hours)
        _save("shade_hours", self.shade_hours)

        # UTCI threshold exceedance (labelled day/night in filename)
        day_set = set(self.heat_thresholds_day)
        night_set = set(self.heat_thresholds_night)
        for threshold, arr in sorted(self.utci_hours_above.items()):
            suffix = ""
            if threshold in day_set and threshold not in night_set:
                suffix = "_day"
            elif threshold in night_set and threshold not in day_set:
                suffix = "_night"
            _save(f"utci_hours_above_{threshold:g}{suffix}", arr)


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
