"""Thermal state and tile specification models.

:class:`ThermalState` carries surface temperature history between
timesteps (ground and wall thermal inertia via TsWaveDelay).
:class:`TileSpec` describes the geometry of a single tile used by the
large-raster tiling engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ThermalState:
    """
    Thermal state for multi-timestep calculations.

    Carries forward surface temperature history between timesteps
    to model thermal inertia of ground and walls (TsWaveDelay_2015a).

    This enables accurate time-series simulations where surface temperatures
    depend on accumulated heating throughout the day.

    Attributes:
        tgmap1: Upwelling longwave history (center view).
        tgmap1_e: Upwelling longwave history (east view).
        tgmap1_s: Upwelling longwave history (south view).
        tgmap1_w: Upwelling longwave history (west view).
        tgmap1_n: Upwelling longwave history (north view).
        tgout1: Ground temperature output history.
        firstdaytime: Flag for first morning timestep (1.0=first, 0.0=subsequent).
        timeadd: Accumulated time for thermal delay function.
        timestep_dec: Decimal time between steps (fraction of day).

    Note:
        State chaining is handled internally: :func:`solweig.calculate`
        carries thermal state across timesteps automatically within a
        single call (via the internal ``_calculate_single``). ThermalState
        is exported for advanced use, e.g. constructing an initial state
        with :meth:`initial` when driving the internal API directly.
    """

    tgmap1: NDArray[np.floating]
    tgmap1_e: NDArray[np.floating]
    tgmap1_s: NDArray[np.floating]
    tgmap1_w: NDArray[np.floating]
    tgmap1_n: NDArray[np.floating]
    tgout1: NDArray[np.floating]
    firstdaytime: float = 1.0
    timeadd: float = 0.0
    timestep_dec: float = 0.0

    @classmethod
    def initial(cls, shape: tuple[int, int]) -> ThermalState:
        """
        Create initial state for first timestep.

        Args:
            shape: Grid shape (rows, cols) matching the DSM.

        Returns:
            ThermalState with zero-initialized arrays.
        """
        zeros = np.zeros(shape, dtype=np.float32)
        return cls(
            tgmap1=zeros.copy(),
            tgmap1_e=zeros.copy(),
            tgmap1_s=zeros.copy(),
            tgmap1_w=zeros.copy(),
            tgmap1_n=zeros.copy(),
            tgout1=zeros.copy(),
            firstdaytime=1.0,
            timeadd=0.0,
            timestep_dec=0.0,
        )

    @classmethod
    def initial_memmap(cls, shape: tuple[int, int], memmap_dir: Path) -> ThermalState:
        """Create initial state backed by memory-mapped files.

        Identical to :meth:`initial` but arrays are ``np.memmap`` objects,
        allowing the OS to page them to disk transparently.  This is used
        for very large rasters where 6 float32 arrays would exceed RAM.

        Args:
            shape: Grid shape (rows, cols) matching the DSM.
            memmap_dir: Directory for the memmap backing files.
        """
        names = ["tgmap1", "tgmap1_e", "tgmap1_s", "tgmap1_w", "tgmap1_n", "tgout1"]
        arrays = {}
        for name in names:
            fp = memmap_dir / f"state_{name}.dat"
            arr = np.memmap(fp, dtype=np.float32, mode="w+", shape=shape)
            arr[:] = 0
            arrays[name] = arr
        return cls(**arrays, firstdaytime=1.0, timeadd=0.0, timestep_dec=0.0)

    def copy(self) -> ThermalState:
        """Create a deep copy of this state."""
        return ThermalState(
            tgmap1=self.tgmap1.copy(),
            tgmap1_e=self.tgmap1_e.copy(),
            tgmap1_s=self.tgmap1_s.copy(),
            tgmap1_w=self.tgmap1_w.copy(),
            tgmap1_n=self.tgmap1_n.copy(),
            tgout1=self.tgout1.copy(),
            firstdaytime=self.firstdaytime,
            timeadd=self.timeadd,
            timestep_dec=self.timestep_dec,
        )


@dataclass
class TileSpec:
    """
    Specification for a single tile with overlap regions.

    Attributes:
        row_start, row_end: Core tile row bounds (without overlap).
        col_start, col_end: Core tile column bounds (without overlap).
        row_start_full, row_end_full: Full tile row bounds (with overlap).
        col_start_full, col_end_full: Full tile column bounds (with overlap).
        overlap_top, overlap_bottom: Vertical overlap in pixels.
        overlap_left, overlap_right: Horizontal overlap in pixels.
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    row_start_full: int
    row_end_full: int
    col_start_full: int
    col_end_full: int
    overlap_top: int
    overlap_bottom: int
    overlap_left: int
    overlap_right: int

    @property
    def core_shape(self) -> tuple[int, int]:
        """Shape of core tile (without overlap)."""
        return (self.row_end - self.row_start, self.col_end - self.col_start)

    @property
    def full_shape(self) -> tuple[int, int]:
        """Shape of full tile (with overlap)."""
        return (self.row_end_full - self.row_start_full, self.col_end_full - self.col_start_full)

    @property
    def core_slice(self) -> tuple[slice, slice]:
        """Slices for extracting core from full tile result."""
        return (
            slice(self.overlap_top, self.overlap_top + self.core_shape[0]),
            slice(self.overlap_left, self.overlap_left + self.core_shape[1]),
        )

    @property
    def write_slice(self) -> tuple[slice, slice]:
        """Slices for writing core to global output."""
        return (
            slice(self.row_start, self.row_end),
            slice(self.col_start, self.col_end),
        )

    @property
    def read_slice(self) -> tuple[slice, slice]:
        """Slices for reading full tile from global input."""
        return (
            slice(self.row_start_full, self.row_end_full),
            slice(self.col_start_full, self.col_end_full),
        )
