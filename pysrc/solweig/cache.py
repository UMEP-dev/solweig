"""
Cache validation utilities for SVF and wall data.

Provides hash-based validation to detect stale caches when input data changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Cache metadata filename
CACHE_METADATA_FILE = "cache_meta.json"

# Fingerprint format version — bump when the schema of prepare-fingerprint
# changes (e.g. adding a new kwarg to the comparison, switching from
# mtime+size to content hashing). Old fingerprints with a different version
# are treated as an immediate mismatch so unknown formats fail safe.
PREPARE_FINGERPRINT_VERSION = "1"


def pixel_size_tag(pixel_size: float) -> str:
    """Return a directory-safe tag encoding the pixel size, e.g. ``'px1.000'``."""
    return f"px{pixel_size:.3f}"


def compute_array_hash(arr: np.ndarray, *, sample_size: int = 10000) -> str:
    """
    Compute a fast hash of a numpy array.

    Uses a combination of shape, dtype, and sampled values for speed.
    For large arrays, samples evenly spaced values rather than hashing everything.

    Args:
        arr: Numpy array to hash.
        sample_size: Maximum number of values to sample for hashing.

    Returns:
        Hex string hash.
    """
    hasher = hashlib.sha256()

    # Include shape and dtype
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())

    # For small arrays, hash everything
    flat = arr.ravel()
    if len(flat) <= sample_size:
        hasher.update(flat.tobytes())
    else:
        # Sample evenly spaced values for large arrays
        indices = np.linspace(0, len(flat) - 1, sample_size, dtype=np.int64)
        hasher.update(flat[indices].tobytes())

    return hasher.hexdigest()[:16]  # First 16 chars is enough


@dataclass
class CacheMetadata:
    """Metadata for cache validation."""

    dsm_hash: str
    dsm_shape: tuple[int, int]
    pixel_size: float
    cdsm_hash: str | None = None
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "dsm_hash": self.dsm_hash,
            "dsm_shape": list(self.dsm_shape),
            "pixel_size": self.pixel_size,
            "cdsm_hash": self.cdsm_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheMetadata:
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            dsm_hash=data["dsm_hash"],
            dsm_shape=tuple(data["dsm_shape"]),
            pixel_size=data["pixel_size"],
            cdsm_hash=data.get("cdsm_hash"),
        )

    @classmethod
    def from_arrays(
        cls,
        dsm: np.ndarray,
        pixel_size: float,
        cdsm: np.ndarray | None = None,
    ) -> CacheMetadata:
        """Create metadata from input arrays."""
        return cls(
            dsm_hash=compute_array_hash(dsm),
            dsm_shape=(dsm.shape[0], dsm.shape[1]),
            pixel_size=pixel_size,
            cdsm_hash=compute_array_hash(cdsm) if cdsm is not None else None,
        )

    def matches(self, other: CacheMetadata) -> bool:
        """Check if this metadata matches another."""
        return (
            self.dsm_hash == other.dsm_hash
            and self.dsm_shape == other.dsm_shape
            and abs(self.pixel_size - other.pixel_size) < 0.001
            and self.cdsm_hash == other.cdsm_hash
        )

    def save(self, directory: Path) -> None:
        """Save metadata to cache directory."""
        meta_path = directory / CACHE_METADATA_FILE
        with open(meta_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, directory: Path) -> CacheMetadata | None:
        """Load metadata from cache directory. Returns None if not found."""
        meta_path = directory / CACHE_METADATA_FILE
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None


def validate_cache(
    cache_dir: Path,
    dsm: np.ndarray,
    pixel_size: float,
    cdsm: np.ndarray | None = None,
) -> bool:
    """
    Validate that cached data matches current inputs.

    Args:
        cache_dir: Directory containing cached data.
        dsm: Current DSM array.
        pixel_size: Current pixel size.
        cdsm: Current CDSM array (optional).

    Returns:
        True if cache is valid, False if stale or missing.
    """
    stored = CacheMetadata.load(cache_dir)
    if stored is None:
        logger.debug(f"No cache metadata found in {cache_dir}")
        return False

    current = CacheMetadata.from_arrays(dsm, pixel_size, cdsm)

    if stored.matches(current):
        logger.debug(f"Cache validated: {cache_dir}")
        return True
    else:
        logger.info(f"Cache stale (input changed): {cache_dir}")
        logger.debug(f"  Stored: dsm_hash={stored.dsm_hash}, shape={stored.dsm_shape}")
        logger.debug(f"  Current: dsm_hash={current.dsm_hash}, shape={current.dsm_shape}")
        return False


def clear_stale_cache(cache_dir: Path) -> None:
    """
    Remove stale cache files from a directory.

    Deletes all .npy files and the metadata file.
    """
    if not cache_dir.exists():
        return

    import shutil

    for item in cache_dir.iterdir():
        if item.is_file() and (item.suffix == ".npy" or item.name == CACHE_METADATA_FILE):
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    logger.info(f"Cleared stale cache: {cache_dir}")


# ---------------------------------------------------------------------------
# prepare() fast-path fingerprint
# ---------------------------------------------------------------------------


def _stat_source_file(path: str | Path | None) -> dict[str, Any] | None:
    """
    Build a fingerprint entry for a single source raster file.

    Returns ``None`` if ``path`` is ``None`` or the file doesn't exist.
    The returned dict has an absolute path plus mtime/size — cheap to
    compute, robust enough to detect in-place edits and file swaps.

    Uses ``os.path.abspath`` rather than ``Path.resolve`` to avoid the
    ``lstat()`` syscall chain that ``resolve`` triggers; the fingerprint
    is called on every warm ``prepare()`` call and ``abspath`` normalises
    paths without touching the filesystem.
    """
    if path is None:
        return None
    p_str = os.fspath(path)
    try:
        st = os.stat(p_str)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return None
    if not os.path.isfile(p_str):
        return None
    return {
        "path": os.path.abspath(p_str),
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
    }


def compute_prepare_fingerprint(
    sources: dict[str, str | Path | None],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute a fingerprint for ``SurfaceData.prepare()`` inputs.

    The fingerprint is stored in ``working_dir/metadata.json`` so warm runs
    can short-circuit the full load → resample → preprocess pipeline when the
    inputs and kwargs are unchanged.

    Args:
        sources: Mapping from layer name to source file path (or ``None`` if
            that layer was not provided). Typical keys: ``dsm``, ``dem``,
            ``cdsm``, ``tdsm``, ``land_cover``, ``wall_height``, ``wall_aspect``.
        kwargs: Preparation kwargs that affect the produced surface. Values
            must be JSON-serializable. Typical keys: ``pixel_size``, ``bbox``,
            ``trunk_ratio``, ``dsm_relative``, ``cdsm_relative``,
            ``tdsm_relative``, ``min_object_height``, ``smooth_quantized_dem``,
            ``dem_smooth_sigma``.

    Returns:
        JSON-serializable dict with a ``version`` tag, ``sources`` mapping of
        stat-based file fingerprints, and a ``kwargs`` snapshot.
    """
    return {
        "version": PREPARE_FINGERPRINT_VERSION,
        "sources": {name: _stat_source_file(path) for name, path in sorted(sources.items())},
        "kwargs": {k: kwargs[k] for k in sorted(kwargs.keys())},
    }


def compare_prepare_fingerprints(
    stored: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    """
    Compare two ``compute_prepare_fingerprint()`` results.

    Returns a list of human-readable mismatch descriptions. An empty list
    means the fingerprints match and the cache can be reused. Callers should
    log the returned descriptions so users know *why* their cache was
    invalidated (e.g. "kwarg 'dem_smooth_sigma' changed (1.5 → 3.0)" or
    "source 'dem' mtime changed").
    """
    mismatches: list[str] = []

    # Format version: if they differ we can't safely compare anything else.
    s_version = stored.get("version")
    c_version = current.get("version")
    if s_version != c_version:
        mismatches.append(f"fingerprint version changed ({s_version!r} → {c_version!r}) — full rebuild")
        return mismatches

    # Source-file entries.
    s_sources = stored.get("sources", {}) or {}
    c_sources = current.get("sources", {}) or {}
    for name in sorted(set(s_sources.keys()) | set(c_sources.keys())):
        s_entry = s_sources.get(name)
        c_entry = c_sources.get(name)
        # Presence change: one has a file, the other doesn't.
        if s_entry is None and c_entry is None:
            continue
        if s_entry is None or c_entry is None:
            was = "absent" if s_entry is None else "present"
            now = "absent" if c_entry is None else "present"
            mismatches.append(f"source '{name}' presence changed (was {was}, now {now})")
            continue
        # Both present: compare path, size, mtime.
        if s_entry.get("path") != c_entry.get("path"):
            mismatches.append(f"source '{name}' path changed ({s_entry.get('path')!r} → {c_entry.get('path')!r})")
        elif s_entry.get("size") != c_entry.get("size"):
            mismatches.append(f"source '{name}' size changed ({s_entry.get('size')} → {c_entry.get('size')} bytes)")
        elif s_entry.get("mtime_ns") != c_entry.get("mtime_ns"):
            # Don't dump raw ns in the message — format as seconds.
            s_mtime = int(s_entry.get("mtime_ns") or 0) / 1e9
            c_mtime = int(c_entry.get("mtime_ns") or 0) / 1e9
            mismatches.append(f"source '{name}' mtime changed ({s_mtime:.0f} → {c_mtime:.0f})")

    # Kwarg snapshot.
    s_kwargs = stored.get("kwargs", {}) or {}
    c_kwargs = current.get("kwargs", {}) or {}
    for k in sorted(set(s_kwargs.keys()) | set(c_kwargs.keys())):
        s_val = s_kwargs.get(k)
        c_val = c_kwargs.get(k)
        if s_val != c_val:
            mismatches.append(f"kwarg {k!r} changed ({s_val!r} → {c_val!r})")

    return mismatches
