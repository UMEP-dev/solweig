"""Pre-computed SVF arrays and shadow matrices.

Normally you don't need to touch these directly —
:meth:`SurfaceData.prepare` computes and stores them automatically.
These classes are only needed if you want to load or supply your own
pre-computed data (e.g. from ``svfs.zip`` / ``shadowmats.npz``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..cache import CacheMetadata, pixel_size_tag
from ..solweig_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


@dataclass
class SvfArrays:
    """
    Sky View Factor arrays (15 directional grids).

    SVF measures how much open sky each pixel can see (0 = fully
    enclosed, 1 = fully open). Directional components (N/E/S/W) split
    this by compass quadrant. Vegetation variants account for tree
    canopy obstruction.

    Normally created by :meth:`SurfaceData.prepare`. Can also be loaded
    from a ``svfs.zip`` via :meth:`from_zip`.
    """

    svf: NDArray[np.floating]
    svf_north: NDArray[np.floating]
    svf_east: NDArray[np.floating]
    svf_south: NDArray[np.floating]
    svf_west: NDArray[np.floating]
    svf_veg: NDArray[np.floating]
    svf_veg_north: NDArray[np.floating]
    svf_veg_east: NDArray[np.floating]
    svf_veg_south: NDArray[np.floating]
    svf_veg_west: NDArray[np.floating]
    svf_aveg: NDArray[np.floating]
    svf_aveg_north: NDArray[np.floating]
    svf_aveg_east: NDArray[np.floating]
    svf_aveg_south: NDArray[np.floating]
    svf_aveg_west: NDArray[np.floating]

    def __post_init__(self):
        import dataclasses

        # Ensure all arrays are float32 for memory efficiency.
        # np.asarray preserves memmap arrays (doesn't copy unless dtype changes).
        for f in dataclasses.fields(self):
            arr = getattr(self, f.name)
            if isinstance(arr, np.memmap):
                if arr.dtype != np.float32:
                    logger.warning(f"Memmap array {f.name} has wrong dtype, loading into memory")
                    setattr(self, f.name, np.asarray(arr, dtype=np.float32))
            else:
                setattr(self, f.name, np.asarray(arr, dtype=np.float32))

    @property
    def svfalfa(self) -> NDArray[np.floating]:
        """Compute SVF alpha (angle) from SVF values. Computed on-demand."""
        tmp = self.svf + self.svf_veg - 1.0
        tmp = np.clip(tmp, 0.0, 1.0)
        eps = np.finfo(np.float32).tiny
        safe_term = np.clip(1.0 - tmp, eps, 1.0)
        return np.arcsin(np.exp(np.log(safe_term) / 2.0))

    @property
    def svfbuveg(self) -> NDArray[np.floating]:
        """Combined building + vegetation SVF. Computed on-demand."""
        return np.clip(self.svf + self.svf_veg - 1.0, 0.0, 1.0)

    def crop(self, r0: int, r1: int, c0: int, c1: int) -> SvfArrays:
        """Crop all SVF arrays to [r0:r1, c0:c1]."""
        return SvfArrays(
            svf=self.svf[r0:r1, c0:c1].copy(),
            svf_north=self.svf_north[r0:r1, c0:c1].copy(),
            svf_east=self.svf_east[r0:r1, c0:c1].copy(),
            svf_south=self.svf_south[r0:r1, c0:c1].copy(),
            svf_west=self.svf_west[r0:r1, c0:c1].copy(),
            svf_veg=self.svf_veg[r0:r1, c0:c1].copy(),
            svf_veg_north=self.svf_veg_north[r0:r1, c0:c1].copy(),
            svf_veg_east=self.svf_veg_east[r0:r1, c0:c1].copy(),
            svf_veg_south=self.svf_veg_south[r0:r1, c0:c1].copy(),
            svf_veg_west=self.svf_veg_west[r0:r1, c0:c1].copy(),
            svf_aveg=self.svf_aveg[r0:r1, c0:c1].copy(),
            svf_aveg_north=self.svf_aveg_north[r0:r1, c0:c1].copy(),
            svf_aveg_east=self.svf_aveg_east[r0:r1, c0:c1].copy(),
            svf_aveg_south=self.svf_aveg_south[r0:r1, c0:c1].copy(),
            svf_aveg_west=self.svf_aveg_west[r0:r1, c0:c1].copy(),
        )

    @classmethod
    def from_rust_result(
        cls,
        svf_result,
        use_veg: bool = True,
        ones: NDArray[np.floating] | None = None,
    ) -> SvfArrays:
        """
        Create SvfArrays from a raw Rust SVF computation result.

        Wraps each attribute with ``np.array()`` and substitutes ``ones``
        arrays for vegetation fields when ``use_veg`` is False.

        Args:
            svf_result: Rust SvfResult object with per-direction SVF attributes.
            use_veg: Whether vegetation SVF was computed. When False, vegetation
                and aveg fields are filled with ``ones``.
            ones: Array of ones matching the grid shape (used when ``use_veg``
                is False). If None and ``use_veg`` is False, a ones array is
                created from the SVF shape.

        Returns:
            SvfArrays instance.
        """
        svf_arr = np.array(svf_result.svf)
        if not use_veg:
            if ones is None:
                ones = np.ones_like(svf_arr, dtype=np.float32)
        else:
            # ones not needed when use_veg is True, but satisfy type checker
            ones = np.ones_like(svf_arr, dtype=np.float32)
        return cls(
            svf=svf_arr,
            svf_north=np.array(svf_result.svf_north),
            svf_east=np.array(svf_result.svf_east),
            svf_south=np.array(svf_result.svf_south),
            svf_west=np.array(svf_result.svf_west),
            svf_veg=np.array(svf_result.svf_veg) if use_veg else ones.copy(),
            svf_veg_north=np.array(svf_result.svf_veg_north) if use_veg else ones.copy(),
            svf_veg_east=np.array(svf_result.svf_veg_east) if use_veg else ones.copy(),
            svf_veg_south=np.array(svf_result.svf_veg_south) if use_veg else ones.copy(),
            svf_veg_west=np.array(svf_result.svf_veg_west) if use_veg else ones.copy(),
            svf_aveg=np.array(svf_result.svf_veg_blocks_bldg_sh) if use_veg else ones.copy(),
            svf_aveg_north=np.array(svf_result.svf_veg_blocks_bldg_sh_north) if use_veg else ones.copy(),
            svf_aveg_east=np.array(svf_result.svf_veg_blocks_bldg_sh_east) if use_veg else ones.copy(),
            svf_aveg_south=np.array(svf_result.svf_veg_blocks_bldg_sh_south) if use_veg else ones.copy(),
            svf_aveg_west=np.array(svf_result.svf_veg_blocks_bldg_sh_west) if use_veg else ones.copy(),
        )

    @classmethod
    def from_bundle(cls, bundle) -> SvfArrays:
        """
        Create SvfArrays from a SvfBundle (computation result).

        This enables caching fresh-computed SVF back to surface.svf for reuse.

        Args:
            bundle: SvfBundle from resolve_svf() or skyview.calculate_svf()

        Returns:
            SvfArrays instance suitable for caching on SurfaceData.svf
        """
        return cls(
            svf=bundle.svf,
            svf_north=bundle.svf_directional.north,
            svf_east=bundle.svf_directional.east,
            svf_south=bundle.svf_directional.south,
            svf_west=bundle.svf_directional.west,
            svf_veg=bundle.svf_veg,
            svf_veg_north=bundle.svf_veg_directional.north,
            svf_veg_east=bundle.svf_veg_directional.east,
            svf_veg_south=bundle.svf_veg_directional.south,
            svf_veg_west=bundle.svf_veg_directional.west,
            svf_aveg=bundle.svf_aveg,
            svf_aveg_north=bundle.svf_aveg_directional.north,
            svf_aveg_east=bundle.svf_aveg_directional.east,
            svf_aveg_south=bundle.svf_aveg_directional.south,
            svf_aveg_west=bundle.svf_aveg_directional.west,
        )

    @classmethod
    def from_zip(cls, zip_path: str | Path, use_vegetation: bool = True) -> SvfArrays:
        """
        Load SVF arrays from SOLWEIG svfs.zip format.

        Args:
            zip_path: Path to svfs.zip file.
            use_vegetation: Whether to load vegetation SVF arrays. Default True.

        Returns:
            SvfArrays instance with loaded data.

        Memory note:
            Files are extracted temporarily and loaded as float32 arrays.
            The zip file contains GeoTIFF rasters.
        """
        import tempfile
        import zipfile

        from .. import io as common

        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"SVF zip file not found: {zip_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(tmpdir)

            tmppath = Path(tmpdir)

            def load(filename: str) -> NDArray[np.floating]:
                filepath = tmppath / filename
                if not filepath.exists():
                    raise FileNotFoundError(f"Expected SVF file not found in zip: {filename}")
                data, _, _, _ = common.load_raster(str(filepath), ensure_float32=True)
                return data

            # Load basic SVF arrays
            svf = load("svf.tif")
            svf_n = load("svfN.tif")
            svf_e = load("svfE.tif")
            svf_s = load("svfS.tif")
            svf_w = load("svfW.tif")

            # Load vegetation arrays or create defaults
            if use_vegetation:
                svf_veg = load("svfveg.tif")
                svf_veg_n = load("svfNveg.tif")
                svf_veg_e = load("svfEveg.tif")
                svf_veg_s = load("svfSveg.tif")
                svf_veg_w = load("svfWveg.tif")
                svf_aveg = load("svfaveg.tif")
                svf_aveg_n = load("svfNaveg.tif")
                svf_aveg_e = load("svfEaveg.tif")
                svf_aveg_s = load("svfSaveg.tif")
                svf_aveg_w = load("svfWaveg.tif")
            else:
                ones = np.ones_like(svf)
                svf_veg = ones
                svf_veg_n = ones
                svf_veg_e = ones
                svf_veg_s = ones
                svf_veg_w = ones
                svf_aveg = ones
                svf_aveg_n = ones
                svf_aveg_e = ones
                svf_aveg_s = ones
                svf_aveg_w = ones

        return cls(
            svf=svf,
            svf_north=svf_n,
            svf_east=svf_e,
            svf_south=svf_s,
            svf_west=svf_w,
            svf_veg=svf_veg,
            svf_veg_north=svf_veg_n,
            svf_veg_east=svf_veg_e,
            svf_veg_south=svf_veg_s,
            svf_veg_west=svf_veg_w,
            svf_aveg=svf_aveg,
            svf_aveg_north=svf_aveg_n,
            svf_aveg_east=svf_aveg_e,
            svf_aveg_south=svf_aveg_s,
            svf_aveg_west=svf_aveg_w,
        )

    def to_memmap(self, directory: str | Path, metadata: CacheMetadata | None = None) -> Path:
        """
        Save SVF arrays as memory-mapped .npy files for efficient large-raster processing.

        This enables processing of 10k×10k+ rasters without loading all SVF data into RAM.
        The OS handles paging, loading only the needed regions into physical memory.

        Args:
            directory: Directory to save memmap files. Created if doesn't exist.
            metadata: Optional cache metadata for validation on reload.
                When provided, enables automatic cache invalidation if inputs change.

        Returns:
            Path to the directory containing memmap files.

        Memory note:
            For a 10k×10k grid with 15 arrays: ~6 GB on disk, but only accessed
            regions are loaded into RAM. Typical usage loads <100 MB.

        Example:
            svf = SvfArrays.from_zip("svfs.zip")
            svf.to_memmap("cache/svf_memmap")
            # Later:
            svf = SvfArrays.from_memmap("cache/svf_memmap")
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save each array as .npy file
        array_names = [
            "svf",
            "svf_north",
            "svf_east",
            "svf_south",
            "svf_west",
            "svf_veg",
            "svf_veg_north",
            "svf_veg_east",
            "svf_veg_south",
            "svf_veg_west",
            "svf_aveg",
            "svf_aveg_north",
            "svf_aveg_east",
            "svf_aveg_south",
            "svf_aveg_west",
        ]

        for name in array_names:
            arr = getattr(self, name)
            np.save(directory / f"{name}.npy", arr)

        # Save metadata for cache validation
        if metadata is not None:
            metadata.save(directory)

        logger.info(f"Saved SVF memmap cache to {directory} ({len(array_names)} arrays)")
        return directory

    @classmethod
    def from_memmap(cls, directory: str | Path, mode: Literal["r", "r+", "c"] = "r") -> SvfArrays:
        """
        Load SVF arrays as memory-mapped files for efficient large-raster processing.

        Memory-mapped arrays are not loaded into RAM until accessed. The OS handles
        paging, making this suitable for rasters larger than available RAM.

        Args:
            directory: Directory containing memmap .npy files (from to_memmap()).
            mode: Memory-map mode. Default "r" (read-only).
                - "r": Read-only (safest, allows OS caching)
                - "r+": Read-write (modifications saved to disk)
                - "c": Copy-on-write (modifications not saved)

        Returns:
            SvfArrays with memory-mapped backing.

        Memory note:
            Only accessed regions are loaded into physical RAM. For tiled processing,
            this dramatically reduces memory usage compared to loading full arrays.

        Example:
            svf = SvfArrays.from_memmap("cache/svf_memmap")
            # Arrays are loaded on-demand as tiles access them
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"SVF memmap directory not found: {directory}")

        def load_memmap(name: str) -> np.ndarray:
            path = directory / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"SVF memmap file not found: {path}")
            return np.load(path, mmap_mode=mode)

        return cls(
            svf=load_memmap("svf"),
            svf_north=load_memmap("svf_north"),
            svf_east=load_memmap("svf_east"),
            svf_south=load_memmap("svf_south"),
            svf_west=load_memmap("svf_west"),
            svf_veg=load_memmap("svf_veg"),
            svf_veg_north=load_memmap("svf_veg_north"),
            svf_veg_east=load_memmap("svf_veg_east"),
            svf_veg_south=load_memmap("svf_veg_south"),
            svf_veg_west=load_memmap("svf_veg_west"),
            svf_aveg=load_memmap("svf_aveg"),
            svf_aveg_north=load_memmap("svf_aveg_north"),
            svf_aveg_east=load_memmap("svf_aveg_east"),
            svf_aveg_south=load_memmap("svf_aveg_south"),
            svf_aveg_west=load_memmap("svf_aveg_west"),
        )


# ShadowArrays and bitpacking helpers were extracted to shadow_arrays.py to
# keep this module under the 700-line hot-file threshold. Re-export so existing
# imports (e.g. ``from .precomputed import ShadowArrays``) keep working.
from .shadow_arrays import (  # noqa: E402, F401
    ShadowArrays,
    _pack_u8_to_bitpacked,
    _unpack_bitpacked_to_float32,
)


@dataclass
class PrecomputedData:
    """
    Optional container for externally supplied SVF/shadow data.

    Most users don't need this — :meth:`SurfaceData.prepare` computes
    everything. Use ``PrecomputedData`` only when loading SVF or shadow
    matrices from files produced by a previous run or an external tool.

    Example::

        svf = SvfArrays.from_zip("cache/svfs.zip")
        shadows = ShadowArrays.from_npz("cache/shadowmats.npz")
        precomputed = PrecomputedData(svf=svf, shadow_matrices=shadows)

        result = calculate(surface=surface, weather=weather, precomputed=precomputed)
    """

    wall_height: NDArray[np.floating] | None = None
    wall_aspect: NDArray[np.floating] | None = None
    svf: SvfArrays | None = None
    shadow_matrices: ShadowArrays | None = None

    @classmethod
    def prepare(
        cls,
        walls_dir: str | Path | None = None,
        svf_dir: str | Path | None = None,
    ) -> PrecomputedData:
        """
        Prepare preprocessing data from directories.

        Loads preprocessing files if they exist. If files don't exist,
        the corresponding data will be None.

        All parameters are optional.

        Args:
            walls_dir: Directory containing wall preprocessing files:
                - wall_hts.tif: Wall heights (meters)
                - wall_aspects.tif: Wall aspects (degrees, 0=N)
            svf_dir: Directory containing SVF preprocessing files:
                - svfs.zip: SVF arrays (required if svf_dir provided)
                - shadowmats.npz: Shadow matrices for anisotropic sky (optional)

        Returns:
            PrecomputedData with loaded arrays. Missing data is set to None.

        Example:
            # Prepare all preprocessing
            precomputed = PrecomputedData.prepare(
                walls_dir="preprocessed/walls",
                svf_dir="preprocessed/svf",
            )

            # Prepare only SVF
            precomputed = PrecomputedData.prepare(svf_dir="preprocessed/svf")

            # Nothing prepared (SVF must be provided before calculate())
            precomputed = PrecomputedData.prepare()
        """
        from .. import io

        wall_height_arr = None
        wall_aspect_arr = None
        svf_arrays = None
        shadow_arrays = None

        def _load_svf_from_dir(base: Path) -> SvfArrays | None:
            memmap_dir = base / "memmap"
            svf_zip = base / "svfs.zip"
            if memmap_dir.exists() and (memmap_dir / "svf.npy").exists():
                logger.info(f"  Loaded SVF memmap cache from {memmap_dir}")
                return SvfArrays.from_memmap(memmap_dir)
            if svf_zip.exists():
                logger.info(f"  Loaded SVF zip from {svf_zip}")
                return SvfArrays.from_zip(str(svf_zip))
            return None

        def _load_shadow_from_dir(base: Path) -> ShadowArrays | None:
            shadow_npz = base / "shadowmats.npz"
            if shadow_npz.exists():
                logger.info(f"  Loaded shadow matrices from {shadow_npz}")
                return ShadowArrays.from_npz(str(shadow_npz))

            shadow_memmap_dir = base / "shadow_memmaps"
            metadata = shadow_memmap_dir / "metadata.json"
            if shadow_memmap_dir.exists() and metadata.exists():
                logger.info(f"  Loaded shadow memmaps from {shadow_memmap_dir}")
                return ShadowArrays.from_memmap(shadow_memmap_dir)
            return None

        # Load walls if directory provided
        if walls_dir is not None:
            walls_path = Path(walls_dir)
            wall_height_path = walls_path / "wall_hts.tif"
            wall_aspect_path = walls_path / "wall_aspects.tif"

            if wall_height_path.exists():
                wall_height_arr, _, _, _ = io.load_raster(str(wall_height_path))
                logger.info(f"  Loaded wall heights from {walls_dir}")
            else:
                logger.debug(f"  Wall heights not found: {wall_height_path}")

            if wall_aspect_path.exists():
                wall_aspect_arr, _, _, _ = io.load_raster(str(wall_aspect_path))
                logger.info(f"  Loaded wall aspects from {walls_dir}")
            else:
                logger.debug(f"  Wall aspects not found: {wall_aspect_path}")

        # Load SVF if directory provided
        if svf_dir is not None:
            svf_path = Path(svf_dir)
            svf_arrays = _load_svf_from_dir(svf_path)
            shadow_arrays = _load_shadow_from_dir(svf_path)

            # Fallback: look for pixel-size-keyed cache under svf/<tag>/ when
            # caller points at a prepared surface directory root.
            if svf_arrays is None or shadow_arrays is None:
                candidate_dirs: list[Path] = []
                meta_path = svf_path / "metadata.json"
                if meta_path.exists():
                    try:
                        with meta_path.open("r", encoding="utf-8") as f:
                            meta = json.load(f)
                        px = meta.get("pixel_size")
                        if px is not None:
                            candidate_dirs.append(svf_path / "svf" / pixel_size_tag(float(px)))
                    except Exception:
                        pass

                svf_root = svf_path / "svf"
                if svf_root.exists():
                    for child in svf_root.iterdir():
                        if child.is_dir():
                            candidate_dirs.append(child)

                seen: set[Path] = set()
                for candidate in candidate_dirs:
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    if svf_arrays is None:
                        svf_arrays = _load_svf_from_dir(candidate)
                    if shadow_arrays is None:
                        shadow_arrays = _load_shadow_from_dir(candidate)
                    if svf_arrays is not None and shadow_arrays is not None:
                        break

            if svf_arrays is None:
                logger.debug(f"  SVF not found in {svf_path}")
            else:
                logger.info(f"  Loaded SVF data: {svf_arrays.svf.shape}")

            if shadow_arrays is None:
                logger.debug("  No shadow matrices found (anisotropic sky will be slower)")
            else:
                logger.info("  Loaded shadow matrices for anisotropic sky")

        return cls(
            wall_height=wall_height_arr,
            wall_aspect=wall_aspect_arr,
            svf=svf_arrays,
            shadow_matrices=shadow_arrays,
        )
