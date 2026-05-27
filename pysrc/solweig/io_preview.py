"""Preview PNG generation for raster outputs.

Extracted from `io.py` to keep the raster I/O module below the
700-line hot-file threshold. These helpers are only used by
:func:`solweig.io.save_raster` when ``preview=True``. PIL and
matplotlib are imported lazily so that headless / minimal
deployments stay free of the optional dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Default color scale ranges for preview images (ensures consistency across timesteps)
# Format: prefix -> (vmin, vmax)
_PREVIEW_RANGES: dict[str, tuple[float, float]] = {
    "tmrt": (0, 80),  # Mean radiant temperature (°C)
    "utci": (-40, 50),  # Universal Thermal Climate Index (°C)
    "pet": (-40, 50),  # Physiological Equivalent Temperature (°C)
    "shadow": (0, 1),  # Shadow fraction (1=sunlit, 0=shaded)
    "kdown": (0, 1200),  # Downwelling shortwave radiation (W/m²)
    "kup": (0, 800),  # Upwelling shortwave radiation (W/m²)
    "ldown": (150, 550),  # Downwelling longwave radiation (W/m²)
    "lup": (250, 650),  # Upwelling longwave radiation (W/m²)
    "svf": (0, 1),  # Sky view factor
    "gvf": (0, 1),  # Ground view factor
}


def _get_preview_range(filename: str) -> tuple[float, float] | None:
    """Get the color scale range for a variable based on filename prefix."""
    name = filename.lower()
    for prefix, range_vals in _PREVIEW_RANGES.items():
        if name.startswith(prefix):
            return range_vals
    return None


def _generate_preview_png(data_arr: np.ndarray, out_path: Path, max_size: int = 512, colormap: str = "turbo") -> None:
    """
    Generate a color PNG preview image from raster data.

    Uses consistent color scales for known variable types (tmrt, utci, shadow, etc.)
    to enable visual comparison across timesteps. Falls back to percentile-based
    scaling for unknown variables.

    Args:
        data_arr: 2D numpy array to visualize
        out_path: Output file path (preview will be saved as .preview.png)
        max_size: Maximum dimension for the preview image (maintains aspect ratio)
        colormap: Matplotlib colormap name (default: 'turbo'). Falls back to grayscale if unavailable.
                  Common options: 'turbo', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm'
    """
    try:
        from PIL import Image

        # Handle NaN values
        valid_mask = ~np.isnan(data_arr)
        if not np.any(valid_mask):
            return  # All NaN, skip preview

        # Use variable-specific range if available, otherwise fall back to percentiles
        preset_range = _get_preview_range(out_path.stem)
        if preset_range is not None:
            vmin, vmax = preset_range
        else:
            # Fallback: use percentiles for unknown variables
            valid_data = data_arr[valid_mask]
            vmin, vmax = np.nanpercentile(valid_data, [2, 98])

        if vmax <= vmin:
            vmax = vmin + 1  # Avoid division by zero

        # Normalize to 0-1
        normalized = np.clip((data_arr - vmin) / (vmax - vmin), 0, 1)
        normalized = np.nan_to_num(normalized, nan=0)

        # Try to apply matplotlib colormap for color output
        try:
            import matplotlib.pyplot as plt

            # Get colormap and apply
            cmap = plt.get_cmap(colormap)
            colored = cmap(normalized)  # Returns RGBA in [0, 1]

            # Convert to RGB uint8 (drop alpha channel)
            rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(rgb, mode="RGB")
        except (ImportError, ValueError):
            # Fallback to grayscale if matplotlib not available or colormap invalid
            grayscale = (normalized * 255).astype(np.uint8)
            img = Image.fromarray(grayscale, mode="L")

        # Resize to max_size while maintaining aspect ratio
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save preview
        preview_path = out_path.with_suffix(".preview.png")
        img.save(preview_path, "PNG")
        logger.debug(f"Saved preview: {preview_path}")
    except ImportError:
        logger.debug("PIL not available, skipping preview generation")
    except Exception as e:
        logger.warning(f"Failed to generate preview: {e}")
