"""Shadow matrices for the anisotropic sky model.

Extracted from `models/precomputed.py` so that the precomputed
container stays under the 700-line hot-file threshold. The
:class:`ShadowArrays` dataclass plus the private bitpacking helpers
:func:`_pack_u8_to_bitpacked` and :func:`_unpack_bitpacked_to_float32`
are re-exported from :mod:`solweig.models.precomputed` for
backwards compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _unpack_bitpacked_to_float32(packed: NDArray[np.uint8], patch_count: int) -> NDArray[np.floating]:
    """Unpack bitpacked shadow matrix to float32 (0.0 or 1.0).

    Args:
        packed: Bitpacked array, shape (rows, cols, n_pack) where n_pack = ceil(patch_count/8).
        patch_count: Number of actual patches.

    Returns:
        Float32 array, shape (rows, cols, patch_count) with values 0.0 or 1.0.
    """
    rows, cols, _ = packed.shape
    result = np.zeros((rows, cols, patch_count), dtype=np.float32)
    for p in range(patch_count):
        byte_idx = p >> 3
        bit_mask = np.uint8(1 << (p & 7))
        result[:, :, p] = ((packed[:, :, byte_idx] & bit_mask) != 0).astype(np.float32)
    return result


def _pack_u8_to_bitpacked(
    u8_data: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Pack u8 shadow matrix (0 or 255 per patch) to bitpacked format.

    Args:
        u8_data: Array shape (rows, cols, patch_count) with values 0 or 255.

    Returns:
        Bitpacked array, shape (rows, cols, n_pack) where n_pack = ceil(patch_count/8).
    """
    rows, cols, patch_count = u8_data.shape
    n_pack = (patch_count + 7) // 8
    packed = np.zeros((rows, cols, n_pack), dtype=np.uint8)
    for p in range(patch_count):
        byte_idx = p >> 3
        bit_mask = np.uint8(1 << (p & 7))
        packed[:, :, byte_idx] |= np.where(u8_data[:, :, p] >= 128, bit_mask, np.uint8(0))
    return packed


@dataclass
class ShadowArrays:
    """
    Shadow matrices for the anisotropic sky model.

    Record which sky patches are visible or blocked (by buildings /
    vegetation) at each pixel. Used to distribute diffuse radiation
    realistically across the sky dome instead of treating it as uniform.

    Normally created by :meth:`SurfaceData.prepare`. Can also be loaded
    from ``shadowmats.npz`` via :meth:`from_npz`.

    Stored internally as bitpacked uint8 for memory efficiency.
    """

    _shmat_u8: NDArray[np.uint8]
    _vegshmat_u8: NDArray[np.uint8]
    _vbshmat_u8: NDArray[np.uint8]
    _n_patches: int = 153
    patch_count: int = field(init=False)
    # Cache for converted float32 arrays (allocated on first access)
    _shmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vegshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _vbshmat_f32: NDArray[np.floating] | None = field(init=False, default=None, repr=False)
    _steradians: NDArray[np.float32] | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # Ensure uint8 dtype
        if self._shmat_u8.dtype != np.uint8:
            self._shmat_u8 = self._shmat_u8.astype(np.uint8)
        if self._vegshmat_u8.dtype != np.uint8:
            self._vegshmat_u8 = self._vegshmat_u8.astype(np.uint8)
        if self._vbshmat_u8.dtype != np.uint8:
            self._vbshmat_u8 = self._vbshmat_u8.astype(np.uint8)

        self.patch_count = self._n_patches
        # Initialize cache as None (lazy allocation)
        self._shmat_f32 = None
        self._vegshmat_f32 = None
        self._vbshmat_f32 = None
        self._steradians = None

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Grid shape (rows, cols) the shadow matrices were computed for."""
        return (int(self._shmat_u8.shape[0]), int(self._shmat_u8.shape[1]))

    @property
    def shmat(self) -> NDArray[np.floating]:
        """Building shadow matrix as float32 (0.0-1.0). Unpacked from bitpacked on demand."""
        if self._shmat_f32 is None:
            self._shmat_f32 = _unpack_bitpacked_to_float32(self._shmat_u8, self.patch_count)
        return self._shmat_f32

    @property
    def vegshmat(self) -> NDArray[np.floating]:
        """Vegetation shadow matrix as float32 (0.0-1.0). Unpacked from bitpacked on demand."""
        if self._vegshmat_f32 is None:
            self._vegshmat_f32 = _unpack_bitpacked_to_float32(self._vegshmat_u8, self.patch_count)
        return self._vegshmat_f32

    @property
    def vbshmat(self) -> NDArray[np.floating]:
        """Combined shadow matrix as float32 (0.0-1.0). Unpacked from bitpacked on demand."""
        if self._vbshmat_f32 is None:
            self._vbshmat_f32 = _unpack_bitpacked_to_float32(self._vbshmat_u8, self.patch_count)
        return self._vbshmat_f32

    @property
    def patch_option(self) -> int:
        """Patch option code (1=145, 2=153, 3=305, 4=609 patches)."""
        patch_map = {145: 1, 153: 2, 305: 3, 609: 4}
        return patch_map.get(self.patch_count, 2)

    @property
    def steradians(self) -> NDArray[np.float32]:
        """Patch steradians (cached, depends only on patch layout)."""
        if self._steradians is None:
            from ..physics.create_patches import create_patches
            from ..physics.patch_radiation import patch_steradians

            skyvaultalt, skyvaultazi, *_ = create_patches(self.patch_option)
            # patch_steradians only uses column 0 (altitudes)
            lv_stub = np.column_stack([skyvaultalt.ravel(), skyvaultazi.ravel(), np.zeros(skyvaultalt.size)])
            self._steradians, _, _ = patch_steradians(lv_stub)
        return self._steradians

    def diffsh(self, transmissivity: float = 0.03, use_vegetation: bool = True) -> NDArray[np.floating]:
        """
        Compute diffuse shadow matrix.

        Args:
            transmissivity: Vegetation transmissivity (default 0.03).
            use_vegetation: Whether to account for vegetation.

        Returns:
            Diffuse shadow matrix as float32.
        """
        shmat = self.shmat
        if use_vegetation:
            vegshmat = self.vegshmat
            return (shmat - (1 - vegshmat) * (1 - transmissivity)).astype(np.float32)
        return shmat

    def release_float32_cache(self) -> None:
        """Release cached float32 shadow matrices to free memory.

        The bitpacked originals remain available. Future property access will
        re-unpack as needed.
        """
        self._shmat_f32 = None
        self._vegshmat_f32 = None
        self._vbshmat_f32 = None

    def crop(self, r0: int, r1: int, c0: int, c1: int) -> ShadowArrays:
        """Crop all shadow matrices to [r0:r1, c0:c1] (3D: rows, cols, n_pack)."""
        return ShadowArrays(
            _shmat_u8=self._shmat_u8[r0:r1, c0:c1, :].copy(),
            _vegshmat_u8=self._vegshmat_u8[r0:r1, c0:c1, :].copy(),
            _vbshmat_u8=self._vbshmat_u8[r0:r1, c0:c1, :].copy(),
            _n_patches=self.patch_count,
        )

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> ShadowArrays:
        """
        Load shadow matrices from SOLWEIG shadowmats.npz format.

        Handles both legacy u8-per-patch format and new bitpacked format.
        Legacy files have shape[2] matching patch count (145/153/305/609).
        New files include a 'patch_count' metadata key.
        """
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Shadow matrices file not found: {npz_path}")

        data = np.load(str(npz_path))

        shmat = data["shadowmat"]
        vegshmat = data["vegshadowmat"]
        vbshmat = data["vbshmat"]

        # Detect format: new bitpacked files include 'patch_count' key
        if "patch_count" in data:
            patch_count = int(data["patch_count"])
            # Data is already bitpacked uint8
            return cls(
                _shmat_u8=shmat.astype(np.uint8),
                _vegshmat_u8=vegshmat.astype(np.uint8),
                _vbshmat_u8=vbshmat.astype(np.uint8),
                _n_patches=patch_count,
            )

        # Legacy format: shape[2] == patch_count, values are 0/255 uint8 or 0.0/1.0 float32
        # Convert float32 → uint8 first if needed
        if shmat.dtype != np.uint8:
            shmat = (np.clip(shmat, 0, 1) * 255).astype(np.uint8)
        if vegshmat.dtype != np.uint8:
            vegshmat = (np.clip(vegshmat, 0, 1) * 255).astype(np.uint8)
        if vbshmat.dtype != np.uint8:
            vbshmat = (np.clip(vbshmat, 0, 1) * 255).astype(np.uint8)

        patch_count = shmat.shape[2]

        # Pack u8 → bitpacked
        return cls(
            _shmat_u8=_pack_u8_to_bitpacked(shmat),
            _vegshmat_u8=_pack_u8_to_bitpacked(vegshmat),
            _vbshmat_u8=_pack_u8_to_bitpacked(vbshmat),
            _n_patches=patch_count,
        )

    @classmethod
    def from_memmap(cls, directory: str | Path, mode: Literal["r", "r+", "c"] = "r") -> ShadowArrays:
        """
        Load bitpacked shadow matrices from a memmap directory.

        Expected files:
            - metadata.json (shape, patch_count, file names)
            - shmat.dat
            - vegshmat.dat
            - vbshmat.dat
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Shadow memmap directory not found: {directory}")

        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Shadow memmap metadata not found: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        shape_raw = meta.get("shape")
        if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 3:
            raise ValueError(f"Invalid shadow memmap shape metadata in {metadata_path}: {shape_raw}")
        shape = tuple(int(v) for v in shape_raw)
        patch_count = int(meta.get("patch_count", 153))

        sh_file = meta.get("shadowmat_file", "shmat.dat")
        veg_file = meta.get("vegshadowmat_file", "vegshmat.dat")
        vb_file = meta.get("vbshmat_file", "vbshmat.dat")

        sh_path = directory / sh_file
        veg_path = directory / veg_file
        vb_path = directory / vb_file
        for path in (sh_path, veg_path, vb_path):
            if not path.exists():
                raise FileNotFoundError(f"Expected shadow memmap file not found: {path}")

        return cls(
            _shmat_u8=np.memmap(sh_path, dtype=np.uint8, mode=mode, shape=shape),
            _vegshmat_u8=np.memmap(veg_path, dtype=np.uint8, mode=mode, shape=shape),
            _vbshmat_u8=np.memmap(vb_path, dtype=np.uint8, mode=mode, shape=shape),
            _n_patches=patch_count,
        )
