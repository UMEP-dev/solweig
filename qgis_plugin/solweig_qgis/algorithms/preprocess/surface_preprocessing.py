"""
Surface Data Preprocessing Algorithm

Aligns rasters, computes valid mask, crops, computes walls and SVF,
and saves cleaned GeoTIFFs to a prepared surface directory. This
directory can then be loaded directly by the SOLWEIG Calculation
algorithm, avoiding repeated preprocessing and making intermediate
results transparent.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
)

from ...utils.parameters import (
    add_land_cover_mapping_parameters,
    add_surface_parameters,
    add_vegetation_parameters,
)
from ..base import SolweigAlgorithmBase


def _needs_relative_retry(
    surface: object,
    *,
    cdsm_path: str | None,
    tdsm_path: str | None,
    cdsm_relative: bool,
    tdsm_relative: bool,
) -> tuple[bool, bool]:
    """Detect likely normalized vegetation/trunk rasters marked as absolute."""
    # Deferred import: the plugin must load (and offer its install prompt)
    # even when solweig is not installed, so no solweig import may run at
    # module level. See provider.loadAlgorithms().
    from solweig.geospatial import looks_like_relative

    dem = getattr(surface, "dem", None)
    dsm = getattr(surface, "dsm", None)
    base_surface = dem if dem is not None else dsm

    retry_cdsm = bool(
        cdsm_path and not cdsm_relative and looks_like_relative(getattr(surface, "cdsm", None), base_surface)
    )
    retry_tdsm = bool(
        tdsm_path and not tdsm_relative and looks_like_relative(getattr(surface, "tdsm", None), base_surface)
    )
    return retry_cdsm, retry_tdsm


class SurfacePreprocessingAlgorithm(SolweigAlgorithmBase):
    """
    Prepare surface data for SOLWEIG calculation.

    Loads all surface rasters, aligns them to a common grid, computes
    a unified valid mask, crops to the valid bounding box, computes
    wall heights and aspects, and saves cleaned GeoTIFFs.

    The output directory can be loaded directly by the SOLWEIG Calculation
    algorithm, skipping all preprocessing steps.
    """

    def name(self) -> str:
        return "surface_preprocessing"

    def displayName(self) -> str:
        return self.tr("2. Prepare Surface Data (align, walls, SVF)")

    def shortHelpString(self) -> str:
        return self.tr(
            """Prepare surface data for SOLWEIG calculation.

Aligns all rasters, computes walls and Sky View Factor, and saves
everything needed to run SOLWEIG Calculation directly.

<b>What this does:</b>
<ol>
<li>Loads all surface rasters (DSM, CDSM, DEM, TDSM, Land cover)</li>
<li>Aligns all rasters to a common grid (intersection of extents)</li>
<li>Converts vegetation heights from relative to absolute (if needed)</li>
<li>Computes a unified valid mask (removes NaN borders)</li>
<li>Crops all rasters to the valid bounding box</li>
<li><b>Computes wall heights and wall aspects</b> from the DSM</li>
<li><b>Computes Sky View Factor (SVF)</b> and shadow matrices</li>
<li>Saves all cleaned rasters as GeoTIFFs</li>
</ol>

<b>Outputs:</b>
<pre>
  output_dir/
    dsm.tif
    wall_height.tif
    wall_aspect.tif
    svfs.zip          (Sky View Factor arrays)
    shadowmats.npz    (shadow matrices for anisotropic sky, when export size is manageable)
    svf/&lt;pixel&gt;/shadow_memmaps/ (large-grid shadow cache fallback)
    cdsm.tif         (if CDSM provided)
    dem.tif           (if DEM provided)
    tdsm.tif          (if TDSM provided)
    land_cover.tif    (if land cover provided)
    parametersforsolweig.json  (vegetation & material settings)
    metadata.json     (pixel size, CRS, prepare fingerprint for warm-run cache)
</pre>

<b>Next step:</b>
Run "SOLWEIG Calculation" with the prepared surface directory.

<b>Documentation:</b>
<ul>
<li><a href="https://umep-dev.github.io/solweig/guide/qgis-plugin/">QGIS Plugin Guide</a></li>
<li><a href="https://umep-dev.github.io/solweig/guide/geotiffs/">Working with GeoTIFFs</a></li>
<li><a href="https://umep-dev.github.io/solweig/">SOLWEIG Documentation</a></li>
</ul>"""
        )

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        # Surface inputs (DSM, CDSM, DEM, TDSM, Land cover + per-layer height modes)
        add_surface_parameters(self)

        # Processing extent (optional)
        self.addParameter(
            QgsProcessingParameterExtent(
                "EXTENT",
                self.tr("Processing extent (leave empty to use intersection of inputs)"),
                optional=True,
            )
        )

        # Output pixel size (optional — coarser than native for faster processing)
        pixel_size_param = QgsProcessingParameterNumber(
            "PIXEL_SIZE",
            self.tr("Output pixel size (m) — leave 0 to use native DSM resolution"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=0.0,
            minValue=0.0,
            maxValue=100.0,
            optional=True,
        )
        self.addParameter(pixel_size_param)

        # Wall limit (advanced)
        wall_limit = QgsProcessingParameterNumber(
            "WALL_LIMIT",
            self.tr("Minimum wall height (m)"),
            type=QgsProcessingParameterNumber.Type.Double,
            defaultValue=1.0,
            minValue=0.0,
            maxValue=10.0,
        )
        from qgis.core import QgsProcessingParameterDefinition

        wall_limit.setFlags(wall_limit.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)
        self.addParameter(wall_limit)

        # --- Vegetation (advanced) ---
        add_vegetation_parameters(self)

        # --- Land cover mapping (advanced) ---
        add_land_cover_mapping_parameters(self)

        # Output directory
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                "OUTPUT_DIR",
                self.tr("Output directory for prepared surface"),
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                "SURFACE_DIR",
                self.tr("Prepared surface directory"),
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
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("SOLWEIG Surface Preprocessing")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()

        # Resolve raster sources from QGIS layers, then delegate preparation to
        # the core SOLWEIG API instead of re-implementing alignment/masking here.
        def _layer_source(param_name: str, *, required: bool = False) -> str | None:
            layer = self.parameterAsRasterLayer(parameters, param_name, context)
            if layer is None:
                if required:
                    raise QgsProcessingException(f"{param_name} layer is required")
                return None
            source = layer.source()
            if required and not source:
                raise QgsProcessingException(f"{param_name} layer has no readable source path")
            return source or None

        feedback.setProgressText("Preparing surface via core SOLWEIG API...")
        feedback.setProgress(10)

        dsm_path = _layer_source("DSM", required=True)
        cdsm_path = _layer_source("CDSM")
        dem_path = _layer_source("DEM")
        tdsm_path = _layer_source("TDSM")
        land_cover_path = _layer_source("LAND_COVER")

        requested_pixel_size = self.parameterAsDouble(parameters, "PIXEL_SIZE", context)
        pixel_size_arg = requested_pixel_size if requested_pixel_size > 0 else None
        wall_limit = self.parameterAsDouble(parameters, "WALL_LIMIT", context)
        dsm_relative = self.parameterAsEnum(parameters, "DSM_HEIGHT_MODE", context) == 0
        cdsm_relative = self.parameterAsEnum(parameters, "CDSM_HEIGHT_MODE", context) == 0
        tdsm_relative = self.parameterAsEnum(parameters, "TDSM_HEIGHT_MODE", context) == 0
        min_object_height = self.parameterAsDouble(parameters, "MIN_OBJECT_HEIGHT", context)

        extent_rect = self.parameterAsExtent(parameters, "EXTENT", context)
        bbox = None
        if not extent_rect.isNull():
            bbox = [
                extent_rect.xMinimum(),
                extent_rect.yMinimum(),
                extent_rect.xMaximum(),
                extent_rect.yMaximum(),
            ]
            feedback.pushInfo(f"Using custom extent: {bbox}")

        output_dir = self.parameterAsString(parameters, "OUTPUT_DIR", context)
        os.makedirs(output_dir, exist_ok=True)

        if pixel_size_arg is None:
            feedback.pushInfo("Output pixel size: using native DSM resolution")
        else:
            feedback.pushInfo(f"Requested output pixel size: {pixel_size_arg:.2f} m")

        try:
            surface = solweig.SurfaceData.prepare(
                dsm=dsm_path,
                working_dir=output_dir,
                cdsm=cdsm_path,
                dem=dem_path,
                tdsm=tdsm_path,
                land_cover=land_cover_path,
                bbox=bbox,
                pixel_size=pixel_size_arg,
                dsm_relative=dsm_relative,
                cdsm_relative=cdsm_relative,
                tdsm_relative=tdsm_relative,
                min_object_height=min_object_height,
                feedback=feedback,
            )
        except solweig.ComputationCancelled:
            feedback.pushInfo("Surface preparation cancelled by user.")
            return {}
        except solweig.SolweigError as e:
            from ..base import format_solweig_error

            raise QgsProcessingException(format_solweig_error("Surface preparation failed", e)) from e
        except Exception as e:
            raise QgsProcessingException(f"Surface preparation failed: {e}") from e

        retry_cdsm, retry_tdsm = _needs_relative_retry(
            surface,
            cdsm_path=cdsm_path,
            tdsm_path=tdsm_path,
            cdsm_relative=cdsm_relative,
            tdsm_relative=tdsm_relative,
        )
        if retry_cdsm or retry_tdsm:
            corrected_cdsm_relative = cdsm_relative or retry_cdsm
            corrected_tdsm_relative = tdsm_relative or retry_tdsm
            retry_layers: list[str] = []
            if retry_cdsm:
                retry_layers.append("CDSM")
            if retry_tdsm:
                retry_layers.append("TDSM")
            feedback.pushInfo(
                "Detected likely relative vegetation heights in "
                f"{', '.join(retry_layers)} despite an Absolute selection; "
                "retrying preparation with terrain-relative interpretation."
            )
            try:
                surface = solweig.SurfaceData.prepare(
                    dsm=dsm_path,
                    working_dir=output_dir,
                    cdsm=cdsm_path,
                    dem=dem_path,
                    tdsm=tdsm_path,
                    land_cover=land_cover_path,
                    bbox=bbox,
                    pixel_size=pixel_size_arg,
                    dsm_relative=dsm_relative,
                    cdsm_relative=corrected_cdsm_relative,
                    tdsm_relative=corrected_tdsm_relative,
                    min_object_height=min_object_height,
                    force_recompute=True,
                    feedback=feedback,
                )
            except solweig.ComputationCancelled:
                feedback.pushInfo("Surface preparation cancelled by user.")
                return {}
            except Exception as e:
                raise QgsProcessingException(
                    f"Surface preparation retry with corrected vegetation height conventions failed: {e}"
                ) from e
            cdsm_relative = corrected_cdsm_relative
            tdsm_relative = corrected_tdsm_relative

        pixel_size = surface.pixel_size

        feedback.pushInfo(f"Prepared grid: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} pixels")
        feedback.pushInfo(f"Output pixel size: {pixel_size:.2f} m")

        # SurfaceData.prepare() computes walls with a 1.0 m limit and has already
        # written them twice: to walls/<px>/ (the wall cache prepare() reuses)
        # and to cleaned/wall_height + wall_aspect, which is what
        # SurfaceData.load(), and so the calculation, reads. A custom limit is
        # applied here and BOTH copies are replaced. Walls changed only in
        # memory had no effect (up to 0.1.0b100), and replacing only walls/
        # left the calculation on the 1.0 m walls (0.1.0b101 and b102). The
        # marker file records that the directory holds custom walls, so that a
        # later run with the default limit puts the 1.0 m walls back.
        from solweig.geospatial import pixel_size_tag

        walls_dir = Path(output_dir) / "walls" / pixel_size_tag(pixel_size)
        limit_marker = walls_dir / "wall_limit.txt"
        if wall_limit != 1.0 or limit_marker.exists():
            from solweig import io as solweig_io
            from solweig.geospatial import wallalgorithms as wa

            feedback.pushInfo(
                f"Recomputing walls with minimum height {wall_limit:.1f} m (core SurfaceData.prepare() uses 1.0 m)"
            )
            walls = wa.findwalls(surface.dsm, wall_limit)
            dirwalls = wa.filter1Goodwin_as_aspect_v3(walls, 1.0 / pixel_size, surface.dsm, feedback=feedback)
            surface.wall_height = walls
            surface.wall_aspect = dirwalls

            walls_dir.mkdir(parents=True, exist_ok=True)
            geotransform = surface.geotransform or [0, pixel_size, 0, 0, 0, -pixel_size]
            crs = surface.crs or ""
            solweig_io.save_raster(str(walls_dir / "wall_hts.tif"), walls, geotransform, crs, ensure_float32=True)
            solweig_io.save_raster(
                str(walls_dir / "wall_aspects.tif"), dirwalls, geotransform, crs, ensure_float32=True
            )

            # The copies the calculation loads. load() memory-maps the .npy
            # when present, and `surface` may itself be mapped onto it (warm
            # run), so the new file is written beside it and renamed over it:
            # saving onto a file that is mapped would truncate it under the
            # mapping.
            import numpy as np

            cleaned_dir = Path(output_dir) / "cleaned"
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            for name, array in (("wall_height", walls), ("wall_aspect", dirwalls)):
                staged = cleaned_dir / f"{name}.staged.npy"
                np.save(staged, np.ascontiguousarray(array, dtype=np.float32))
                os.replace(staged, cleaned_dir / f"{name}.npy")
                solweig_io.save_raster(str(cleaned_dir / f"{name}.tif"), array, geotransform, crs, ensure_float32=True)
            if wall_limit != 1.0:
                limit_marker.write_text(f"{wall_limit}\n")
            else:
                limit_marker.unlink()

        if feedback.isCanceled():
            return {}
        if surface.svf is None:
            raise QgsProcessingException(
                "SurfaceData.prepare() completed without producing SVF arrays "
                "(surface.svf is None). Check that the active solweig build "
                "matches the current plugin code."
            )

        if feedback.isCanceled():
            return {}

        # Rasters, walls, SVF, shadow matrices, and metadata.json are all
        # written by SurfaceData.prepare() → save_cleaned() into cleaned/,
        # walls/, svf/ and the working directory root. The only extra file
        # this algorithm adds is parametersforsolweig.json (UMEP-compatible
        # materials / vegetation settings — plugin-specific, not part of the
        # core SurfaceData schema).
        feedback.setProgressText("Saving UMEP materials...")
        feedback.setProgress(80)

        from ...utils.converters import build_materials_from_lc_mapping

        materials = build_materials_from_lc_mapping(parameters, context, self, feedback)
        ts = materials.Tree_settings.Value
        ts.Transmissivity = parameters.get("TRANSMISSIVITY", 0.03)
        ts.Transmissivity_leafoff = parameters.get("TRANSMISSIVITY_LEAFOFF", 0.5)
        ts.First_day_leaf = int(parameters.get("LEAF_START", 97))
        ts.Last_day_leaf = int(parameters.get("LEAF_END", 300))

        try:
            from solweig.geospatial import namespace_to_dict

            params_path = os.path.join(output_dir, "parametersforsolweig.json")
            with open(params_path, "w") as f:
                json.dump(namespace_to_dict(materials), f, indent=2)
            feedback.pushInfo("Saved parametersforsolweig.json (UMEP-compatible)")
        except ImportError:
            pass

        feedback.setProgress(95)

        # Report summary
        computation_time = time.time() - start_time
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Surface preprocessing complete!")
        feedback.pushInfo(f"  Grid size: {surface.dsm.shape[1]}x{surface.dsm.shape[0]} pixels")
        feedback.pushInfo(f"  Pixel size: {pixel_size:.2f} m")
        feedback.pushInfo(f"  Walls: {'present' if surface.wall_height is not None else 'missing'}")
        feedback.pushInfo(f"  SVF:   {'present' if surface.svf is not None else 'missing'}")
        feedback.pushInfo(f"  Total time: {computation_time:.1f} seconds")
        feedback.pushInfo(f"  Output directory: {output_dir}")
        feedback.pushInfo("=" * 60)

        feedback.setProgress(100)

        return {
            "SURFACE_DIR": output_dir,
            "COMPUTATION_TIME": computation_time,
        }
