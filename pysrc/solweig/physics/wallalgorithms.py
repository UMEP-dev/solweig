from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray


def findwalls(a: NDArray[np.floating], walllimit: float) -> NDArray[np.float32]:
    # This function identifies walls based on a DSM and a wall-height limit
    # Walls are represented by outer pixels within building footprints
    #
    # Fredrik Lindberg, Goteborg Urban Climate Group
    # fredrikl@gvc.gu.se
    # 20150625
    #
    # For each pixel, find the max of its 4 cardinal neighbors (cross kernel).
    # Wall height = max_neighbor - self, clipped to walllimit.

    walls = np.zeros_like(a, dtype=np.float32)

    # Max of 4 cardinal neighbors for all interior pixels
    max_neighbors = np.maximum.reduce(
        [
            a[:-2, 1:-1],  # north
            a[2:, 1:-1],  # south
            a[1:-1, :-2],  # west
            a[1:-1, 2:],  # east
        ]
    )
    walls[1:-1, 1:-1] = max_neighbors

    walls = walls - a
    walls[walls < walllimit] = 0

    # Zero borders
    walls[0, :] = 0
    walls[-1, :] = 0
    walls[:, 0] = 0
    walls[:, -1] = 0

    return walls


def filter1Goodwin_as_aspect_v3(
    walls: NDArray[np.floating],
    scale: float,
    a: NDArray[np.floating],
    feedback: Any = None,
    progress_range: tuple[float, float] | None = None,
) -> NDArray[np.float32]:
    """
    tThis function applies the filter processing presented in Goodwin et al (2010) but instead for removing
    linear fetures it calculates wall aspect based on a wall pixels grid, a dsm (a) and a scale factor

    Fredrik Lindberg, 2012-02-14
    fredrikl@gvc.gu.se

    Translated: 2015-09-15

    :param walls:
    :param scale:
    :param a:
    :return: dirwalls
    """
    # Single code path: the Rust ``WallAspectRunner`` kernel. Keeping a
    # Python fallback alongside would give solweig two numerically-
    # divergent behaviors (Rust f32 rotation vs pure-numpy f64, and vs
    # scipy — three sets of results), which breaks reproducibility.
    # If the Rust extension is unavailable, the whole solweig package
    # is non-functional (shadows, SVF, Tmrt all require it), so there
    # is no scenario where a Python fallback here would usefully help.
    from ..progress import ProgressReporter
    from ..rustalgos import wall_aspect as _wa_rust

    walls_f32 = np.asarray(walls, dtype=np.float32)
    dsm_f32 = np.asarray(a, dtype=np.float32)

    runner = _wa_rust.WallAspectRunner()
    result: list[Any] = [None]
    error: list[Exception | None] = [None]

    def _run() -> None:
        try:
            result[0] = runner.compute(walls_f32, float(scale), dsm_f32)
        except Exception as e:  # noqa: BLE001 - propagate any exception from the Rust call to the caller
            error[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    total = 180
    pbar = ProgressReporter(
        total=total, desc="Computing wall aspects", feedback=feedback, progress_range=progress_range
    )
    last = 0
    while thread.is_alive():
        thread.join(timeout=0.05)
        done = runner.progress()
        if done > last:
            pbar.update(done - last)
            last = done
        if feedback is not None and hasattr(feedback, "isCanceled") and feedback.isCanceled():
            runner.cancel()
            thread.join(timeout=5.0)
            pbar.close()
            return np.zeros_like(walls_f32)
    if last < total:
        pbar.update(total - last)
    pbar.close()

    thread.join()
    if error[0] is not None:
        raise error[0]
    return np.asarray(result[0])
