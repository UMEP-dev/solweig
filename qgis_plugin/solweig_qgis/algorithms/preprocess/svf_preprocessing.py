"""
SVF Preprocessing Algorithm

Pre-computes Sky View Factor (SVF) arrays for reuse across timesteps.
"""

from __future__ import annotations

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
)

from ...utils.converters import load_prepared_surface
from ..base import SolweigAlgorithmBase


class SvfPreprocessingAlgorithm(SolweigAlgorithmBase):
    """
    Pre-compute Sky View Factor (SVF) arrays.

    SVF computation is expensive (~30-120s for 1000x1000 grid).
    Pre-computing allows reuse across multiple timesteps, providing
    significant speedup (from 60s to 0.3s per timestep).
    """

    TRUNK_RATIO = "TRUNK_RATIO"
    OUTPUT_DIR = "OUTPUT_DIR"

    def name(self) -> str:
        return "svf_preprocessing"

    def displayName(self) -> str:
        return self.tr("2b. Recompute Sky View Factor (advanced)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Recompute Sky View Factor (SVF) with custom parameters.

<b>Note:</b> SVF is already computed during "Prepare Surface Data" (step 2).
Use this tool only if you need to recompute SVF with different parameters
(e.g., different trunk ratio) without re-running the full surface preparation.

<b>Input:</b>
Provide the <b>prepared surface directory</b> from "Prepare Surface Data".
DSM, CDSM, and TDSM are loaded automatically.

<b>Output:</b>
SVF arrays are saved into the prepared surface directory (`svfs.zip`).
Shadow matrices are saved as `shadowmats.npz` when manageable, or kept
as `svf/&lt;pixel&gt;/shadow_memmaps/` for very large grids. The SOLWEIG
Calculation algorithm loads these automatically.

<b>Typical runtime:</b>
- 1000x1000 grid: 30-120 seconds

<b>Documentation:</b>
<ul>
<li><a href="https://umep-dev.github.io/solweig/guide/qgis-plugin/">QGIS Plugin Guide</a></li>
<li><a href="https://umep-dev.github.io/solweig/physics/svf/">Sky View Factor</a></li>
<li><a href="https://umep-dev.github.io/solweig/">SOLWEIG Documentation</a></li>
</ul>"""
        )

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        self.addParameter(
            QgsProcessingParameterFile(
                "PREPARED_SURFACE_DIR",
                self.tr("Prepared surface directory (from 'Prepare Surface Data')"),
                behavior=QgsProcessingParameterFile.Behavior.Folder,
            )
        )

        trunk_ratio = QgsProcessingParameterNumber(
            self.TRUNK_RATIO,
            self.tr("Trunk ratio (fraction of canopy height, used when no TDSM provided)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.25,
            minValue=0.0,
            maxValue=1.0,
        )
        trunk_ratio.setFlags(trunk_ratio.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(trunk_ratio)

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                self.tr("Output directory for SVF arrays (defaults to prepared surface directory)"),
                optional=True,
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                "SVF_DIR",
                self.tr("SVF output directory"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "COMPUTATION_TIME",
                self.tr("Computation time (seconds)"),
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Execute the algorithm."""
        import time

        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SOLWEIG SVF Preprocessing")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()

        # Load surface data from prepared directory
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        prepared_dir = self.parameterAsFile(parameters, "PREPARED_SURFACE_DIR", context)
        surface = load_prepared_surface(prepared_dir, feedback)
        dsm = surface.dsm
        cdsm = surface.cdsm
        tdsm = surface.tdsm
        geotransform = surface._geotransform
        crs_wkt = surface._crs_wkt
        pixel_size = surface.pixel_size

        feedback.pushInfo(f"DSM: {dsm.shape[1]}x{dsm.shape[0]} pixels")
        feedback.pushInfo(f"Pixel size: {pixel_size:.2f} m")

        if feedback.isCanceled():
            return {}

        trunk_ratio = self.parameterAsDouble(parameters, self.TRUNK_RATIO, context)

        import os

        import numpy as np

        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        # QGIS auto-generates a temp path ending in the parameter name when left blank
        if not output_dir or output_dir.rstrip("/").endswith("OUTPUT_DIR"):
            output_dir = prepared_dir
            feedback.pushInfo(f"SVF output will be saved to prepared surface directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Compute SVF using the same Python API as SurfaceData.prepare() —
        # automatically tiles large grids to stay within GPU buffer limits.
        feedback.setProgressText("Computing Sky View Factor...")
        feedback.setProgress(20)

        from pathlib import Path

        from solweig import SurfaceData as SD

        use_veg = cdsm is not None
        dsm_f32 = dsm.astype(np.float32)

        aligned_rasters = {
            "dsm_arr": dsm_f32,
            "cdsm_arr": cdsm.astype(np.float32) if use_veg else None,
            "tdsm_arr": (
                tdsm.astype(np.float32)
                if tdsm is not None
                else (cdsm * trunk_ratio).astype(np.float32)
                if use_veg
                else None
            ),
            "pixel_size": pixel_size,
            "dsm_transform": geotransform,
            "dsm_crs": crs_wkt,
        }

        rows, cols = dsm_f32.shape
        from solweig import compute_max_tile_pixels

        _max_px = compute_max_tile_pixels(context="svf")
        n_pixels = rows * cols
        if n_pixels > _max_px:
            feedback.pushInfo(
                f"Large grid ({rows}x{cols} = {n_pixels:,} px, limit {_max_px:,}) — using tiled computation"
            )
        else:
            feedback.pushInfo(f"Grid {rows}x{cols} = {n_pixels:,} px — single-pass computation")

        try:
            SD._compute_and_cache_svf(
                surface,
                aligned_rasters,
                Path(output_dir),
                trunk_ratio=trunk_ratio,
                feedback=feedback,
                progress_range=(20.0, 90.0),
            )
        except solweig.SolweigError as e:
            from ..base import format_solweig_error

            raise QgsProcessingException(format_solweig_error("SVF computation failed", e)) from e
        except Exception as e:
            raise QgsProcessingException(f"SVF computation failed: {e}") from e
        if surface.svf is None:
            raise QgsProcessingException(
                "SVF computation completed without producing SVF arrays "
                "(surface.svf is None). Check that the active solweig build "
                "matches the current plugin code."
            )

        if feedback.isCanceled():
            return {}

        feedback.setProgress(90)

        # SVF and shadow matrices are already saved by _compute_and_cache_svf
        # in svf/px{size}/.  No root-level copies needed — SurfaceData.load()
        # finds them via PrecomputedData.prepare() fallback logic.
        from solweig import pixel_size_tag

        cache_dir = Path(output_dir) / "svf" / pixel_size_tag(pixel_size)
        feedback.pushInfo(f"SVF cached in {cache_dir}")

        computation_time = time.time() - start_time
        feedback.setProgress(100)

        # Report results
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SVF preprocessing complete!")
        feedback.pushInfo(f"  Computation time: {computation_time:.1f} seconds")
        feedback.pushInfo(f"  Output directory: {output_dir}")
        feedback.pushInfo("")
        feedback.pushInfo(
            "Use this directory as 'Pre-computed SVF directory' in calculation algorithms for ~200x speedup."
        )
        feedback.pushInfo("=" * 60)

        return {
            "SVF_DIR": output_dir,
            "COMPUTATION_TIME": computation_time,
        }
