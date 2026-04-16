"""
Pure numpy implementations of morphological operations — **reference implementation only**.

Not called by the production ``calculate()`` API. The fused Rust pipeline
uses ``crate::morphology`` internally.

Retained for readability, tests, and validation against UMEP.
Originally replaced scipy.ndimage functions to eliminate the scipy dependency,
making the package lighter for QGIS plugin distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def binary_dilation(
    input_array: NDArray[np.bool_],
    structure: NDArray[np.bool_] | None = None,
    iterations: int = 1,
) -> NDArray[np.bool_]:
    """
    Perform binary dilation on a 2D boolean array.

    Pure numpy implementation replacing scipy.ndimage.binary_dilation.

    Args:
        input_array: 2D boolean array to dilate.
        structure: Structuring element (3x3 boolean array).
                   If None, uses 8-connectivity (all neighbors).
        iterations: Number of times to apply dilation.

    Returns:
        Dilated boolean array.
    """
    if structure is None:
        # Default: 8-connectivity (3x3 all ones)
        structure = np.ones((3, 3), dtype=bool)

    result = input_array.copy()

    for _ in range(iterations):
        # Pad the array
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        new_result = np.zeros_like(result)

        # Apply structuring element
        rows, cols = result.shape
        struct_rows, struct_cols = structure.shape
        offset_r = struct_rows // 2
        offset_c = struct_cols // 2

        for dr in range(struct_rows):
            for dc in range(struct_cols):
                if structure[dr, dc]:
                    shifted = padded[
                        1 + dr - offset_r : 1 + rows + dr - offset_r,
                        1 + dc - offset_c : 1 + cols + dc - offset_c,
                    ]
                    new_result |= shifted

        result = new_result

    return result


def generate_binary_structure(rank: int, connectivity: int) -> NDArray[np.bool_]:
    """
    Generate a binary structuring element for morphological operations.

    Pure numpy implementation replacing scipy.ndimage.generate_binary_structure.

    Args:
        rank: Number of dimensions (must be 2).
        connectivity: 1 for 4-connectivity (cross), 2 for 8-connectivity (square).

    Returns:
        3x3 boolean structuring element.
    """
    if rank != 2:
        raise ValueError(f"Only rank=2 supported, got {rank}")

    if connectivity == 1:
        # 4-connectivity (cross pattern)
        return np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
            dtype=bool,
        )
    elif connectivity == 2:
        # 8-connectivity (all neighbors)
        return np.ones((3, 3), dtype=bool)
    else:
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
