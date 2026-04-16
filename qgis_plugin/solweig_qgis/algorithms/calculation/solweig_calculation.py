"""
Unified SOLWEIG Calculation Algorithm

Supports single timestep, EPW timeseries, or UMEP met timeseries,
with optional tiled processing and UTCI/PET post-processing.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

import numpy as np
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
)

from ...utils.converters import (
    build_materials_from_lc_mapping,
    create_human_params_from_parameters,
    create_location_from_parameters,
    create_physics_from_parameters,
    create_weather_from_parameters,
    load_prepared_surface,
    load_weather_from_epw,
    load_weather_from_umep_met,
)
from ...utils.parameters import (
    add_date_filter_parameters,
    add_epw_parameters,
    add_heat_threshold_parameters,
    add_human_body_parameters,
    add_human_parameters,
    add_location_parameters,
    add_options_parameters,
    add_umep_met_parameters,
    add_vegetation_parameters,
    add_weather_parameters,
)
from ..base import SolweigAlgorithmBase


def _apply_saved_surface_settings(
    prepared_dir: str,
    parameters: dict,
    feedback: QgsProcessingFeedback,
) -> None:
    """Use ``parametersforsolweig.json`` saved during surface preparation.

    If the prepared surface directory contains a ``parametersforsolweig.json``
    and the user has not supplied a custom materials file, point the
    ``CUSTOM_MATERIALS_FILE`` parameter at it so that ``build_materials_from_lc_mapping``
    picks it up automatically.
    """
    params_path = os.path.join(prepared_dir, "parametersforsolweig.json")
    if os.path.exists(params_path):
        # Only apply if the user hasn't explicitly provided a custom file
        custom = parameters.get("CUSTOM_MATERIALS_FILE")
        if not custom:
            parameters["CUSTOM_MATERIALS_FILE"] = params_path
            feedback.pushInfo("Using saved parametersforsolweig.json from prepared surface")


class SolweigCalculationAlgorithm(SolweigAlgorithmBase):
    """
    Unified SOLWEIG calculation algorithm.

    Combines single timestep, timeseries, and optional UTCI/PET post-processing
    into a single Processing algorithm. Large rasters are automatically tiled.
    """

    # Weather source enum values
    WEATHER_SINGLE = 0
    WEATHER_EPW = 1
    WEATHER_UMEP = 2

    def name(self) -> str:
        return "solweig_calculation"

    def displayName(self) -> str:
        return self.tr("3. SOLWEIG Calculation")

    def shortHelpString(self) -> str:
        return self.tr(
            """Calculate Mean Radiant Temperature (Tmrt) with SOLWEIG.

<b>Surface data:</b>
Provide the <b>prepared surface directory</b> from "Prepare Surface Data".
All rasters (DSM, CDSM, DEM, walls) are loaded automatically.

<b>Weather modes:</b>
<ul>
<li><b>Single timestep:</b> Manual weather input for one date/time</li>
<li><b>EPW weather file:</b> Load hourly data from an EnergyPlus Weather file.
EPW header overrides manual or auto-extracted coordinates and UTC offset.</li>
<li><b>UMEP met file:</b> Load from UMEP/SUEWS meteorological forcing files</li>
</ul>
For timeseries modes, thermal state (ground heating/cooling) accumulates
across timesteps for physically accurate results. Large rasters are
automatically processed using overlapping tiles to manage memory.

<b>Post-processing (optional):</b>
<ul>
<li><b>UTCI</b> - fast polynomial (~200 timesteps/sec)</li>
<li><b>PET</b> - iterative heat balance (~4 timesteps/sec, ~50x slower than UTCI)</li>
</ul>

<b>Outputs:</b>
GeoTIFF files organised into subfolders of the output directory:
<pre>
  output_dir/
    tmrt/        tmrt_YYYYMMDD_HHMM.tif  (default on)
    shadow/      shadow_...              (if selected)
    kdown/       kdown_...               (if selected)
    kup/         kup_...                 (if selected)
    ldown/       ldown_...               (if selected)
    lup/         lup_...                 (if selected)
    utci/        utci_...                (if enabled)
    pet/         pet_...                 (if enabled)
</pre>

<b>Recommended workflow:</b>
<ol>
<li>Run "Download / Preview Weather File" to get an EPW file</li>
<li>Run "Prepare Surface Data" to align rasters, compute walls and SVF</li>
<li>Run this algorithm with the prepared surface directory</li>
</ol>

<b>Documentation:</b>
<ul>
<li><a href="https://umep-dev.github.io/solweig/guide/qgis-plugin/">QGIS Plugin Guide</a></li>
<li><a href="https://umep-dev.github.io/solweig/guide/timeseries/">Timeseries Calculations</a></li>
<li><a href="https://umep-dev.github.io/solweig/guide/thermal-comfort/">Thermal Comfort (UTCI/PET)</a></li>
<li><a href="https://umep-dev.github.io/solweig/">SOLWEIG Documentation</a></li>
</ul>"""
        )

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        # --- Prepared surface directory (required) ---
        self.addParameter(
            QgsProcessingParameterFile(
                "PREPARED_SURFACE_DIR",
                self.tr("Prepared surface directory (from 'Prepare Surface Data')"),
                behavior=QgsProcessingParameterFile.Behavior.Folder,
            )
        )

        # --- Location ---
        add_location_parameters(self)

        # --- Weather source selector ---
        self.addParameter(
            QgsProcessingParameterEnum(
                "WEATHER_SOURCE",
                self.tr("Weather data source"),
                options=[
                    "Single timestep (manual entry)",
                    "EPW weather file (timeseries)",
                    "UMEP met file (timeseries)",
                ],
                defaultValue=self.WEATHER_EPW,
            )
        )

        # --- Single timestep weather (advanced - collapsed by default) ---
        add_weather_parameters(self)
        for name in ("DATETIME", "TEMPERATURE", "HUMIDITY", "GLOBAL_RADIATION", "WIND_SPEED", "PRESSURE"):
            param = self.parameterDefinition(name)
            if param:
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)

        # --- EPW weather ---
        add_epw_parameters(self)

        # --- UMEP met weather ---
        add_umep_met_parameters(self)

        # --- Date/time filtering (shared by EPW and UMEP) ---
        add_date_filter_parameters(self)

        # --- Human parameters ---
        add_human_parameters(self)

        # --- Options ---
        add_options_parameters(self)

        # --- Vegetation (advanced) ---
        add_vegetation_parameters(self)

        # --- Post-processing ---
        self.addParameter(
            QgsProcessingParameterBoolean(
                "COMPUTE_UTCI",
                self.tr("Compute UTCI (Universal Thermal Climate Index)"),
                defaultValue=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                "COMPUTE_PET",
                self.tr("Compute PET (Physiological Equivalent Temperature) - ~50x slower than UTCI"),
                defaultValue=False,
            )
        )

        # PET body parameters (advanced)
        add_human_body_parameters(self)
        for name in ("AGE", "WEIGHT", "HEIGHT", "SEX", "ACTIVITY", "CLOTHING"):
            param = self.parameterDefinition(name)
            if param:
                param.setFlags(param.flags() | QgsProcessingParameterDefinition.Flag.FlagAdvanced)

        # --- Heat-stress thresholds (for UTCI exceedance summary grids) ---
        add_heat_threshold_parameters(self)

        # --- Output selection ---
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_TMRT",
                self.tr("Save Tmrt (Mean Radiant Temperature) per timestep"),
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_SHADOW",
                self.tr("Save shadow fraction"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_KDOWN",
                self.tr("Save Kdown (incoming shortwave)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_KUP",
                self.tr("Save Kup (reflected shortwave)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_LDOWN",
                self.tr("Save Ldown (incoming longwave)"),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                "OUTPUT_LUP",
                self.tr("Save Lup (emitted longwave)"),
                defaultValue=False,
            )
        )

        # --- Output directory ---
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                "OUTPUT_DIR",
                self.tr("Output directory (defaults to 'results' inside prepared surface dir)"),
                optional=True,
            )
        )

        # --- Output metadata ---
        self.addOutput(
            QgsProcessingOutputFolder(
                "OUTPUT_FOLDER",
                self.tr("Output folder"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "TIMESTEP_COUNT",
                self.tr("Number of timesteps processed"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "UTCI_COUNT",
                self.tr("Number of UTCI files created"),
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                "PET_COUNT",
                self.tr("Number of PET files created"),
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
        feedback.pushInfo("SOLWEIG Calculation")
        feedback.pushInfo("=" * 60)

        start_time = time.time()

        # Import solweig
        solweig = self.import_solweig()

        # Determine weather mode
        weather_mode = self.parameterAsEnum(parameters, "WEATHER_SOURCE", context)
        is_single = weather_mode == self.WEATHER_SINGLE
        mode_names = {0: "Single timestep", 1: "EPW timeseries", 2: "UMEP met timeseries"}
        feedback.pushInfo(f"Mode: {mode_names.get(weather_mode, 'Unknown')}")

        # Step 1: Load surface data from prepared directory
        feedback.setProgressText("Loading surface data...")
        feedback.setProgress(5)

        prepared_dir = self.parameterAsFile(parameters, "PREPARED_SURFACE_DIR", context)
        surface = load_prepared_surface(prepared_dir, feedback)

        # Apply saved settings from surface preparation as parameter defaults
        _apply_saved_surface_settings(prepared_dir, parameters, feedback)

        if feedback.isCanceled():
            return {}

        epw_path = self.parameterAsFile(parameters, "EPW_FILE", context) if weather_mode == self.WEATHER_EPW else None

        # Step 2: Create Location
        feedback.setProgressText("Setting up location...")
        feedback.setProgress(10)

        location = create_location_from_parameters(parameters, surface, feedback, epw_path=epw_path)

        if feedback.isCanceled():
            return {}

        # Step 3: Load weather
        feedback.setProgressText("Loading weather data...")
        feedback.setProgress(15)

        # Parse shared date/hour filters (used by both EPW and UMEP)
        start_qdt = self.parameterAsDateTime(parameters, "START_DATE", context)
        end_qdt = self.parameterAsDateTime(parameters, "END_DATE", context)
        start_dt = start_qdt if start_qdt.isValid() else None
        end_dt = end_qdt if end_qdt.isValid() else None
        hours_filter = self.parameterAsString(parameters, "HOURS_FILTER", context)

        if is_single:
            weather = create_weather_from_parameters(parameters, feedback)
            weather_series = [weather]
        elif weather_mode == self.WEATHER_EPW:
            assert epw_path is not None  # guaranteed by weather_mode guard above
            weather_series = load_weather_from_epw(
                epw_path=epw_path,
                start_dt=start_dt,
                end_dt=end_dt,
                hours_filter=hours_filter,
                feedback=feedback,
            )
            if not weather_series:
                raise QgsProcessingException("No timesteps found in specified date range")
        elif weather_mode == self.WEATHER_UMEP:
            umep_path = self.parameterAsFile(parameters, "UMEP_MET_FILE", context)
            weather_series = load_weather_from_umep_met(
                met_path=umep_path,
                start_dt=start_dt,
                end_dt=end_dt,
                hours_filter=hours_filter,
                feedback=feedback,
            )
            if not weather_series:
                raise QgsProcessingException("No timesteps found in UMEP met file")

        if feedback.isCanceled():
            return {}

        # Step 4: Get options
        human = create_human_params_from_parameters(parameters)
        physics = create_physics_from_parameters(parameters)
        materials = build_materials_from_lc_mapping(parameters, context, self, feedback)
        use_anisotropic_sky = self.parameterAsBool(parameters, "USE_ANISOTROPIC_SKY", context)
        conifer = self.parameterAsBool(parameters, "CONIFER", context)
        max_shadow_distance_m = self.parameterAsDouble(parameters, "MAX_SHADOW_DISTANCE", context)
        output_dir = self.parameterAsString(parameters, "OUTPUT_DIR", context)

        # Default output to 'results/' inside prepared surface directory
        if not output_dir or output_dir.rstrip("/").endswith("OUTPUT_DIR"):
            output_dir = os.path.join(prepared_dir, "results")
            feedback.pushInfo(f"Output directory: {output_dir} (inside prepared surface dir)")

        # Parse heat-stress thresholds
        heat_thresholds_day = self._parse_thresholds(self.parameterAsString(parameters, "HEAT_THRESHOLDS_DAY", context))
        heat_thresholds_night = self._parse_thresholds(
            self.parameterAsString(parameters, "HEAT_THRESHOLDS_NIGHT", context)
        )
        if heat_thresholds_day:
            feedback.pushInfo(f"Daytime UTCI thresholds: {heat_thresholds_day}")
        if heat_thresholds_night:
            feedback.pushInfo(f"Nighttime UTCI thresholds: {heat_thresholds_night}")

        # Parse output components
        selected_outputs = []
        if self.parameterAsBool(parameters, "OUTPUT_TMRT", context):
            selected_outputs.append("tmrt")
        for comp in ["shadow", "kdown", "kup", "ldown", "lup"]:
            if self.parameterAsBool(parameters, f"OUTPUT_{comp.upper()}", context):
                selected_outputs.append(comp)
        # UTCI/PET are now computed inline during the main calculation
        if self.parameterAsBool(parameters, "COMPUTE_UTCI", context):
            selected_outputs.append("utci")
        if self.parameterAsBool(parameters, "COMPUTE_PET", context):
            selected_outputs.append("pet")
        feedback.pushInfo(f"Outputs: {', '.join(selected_outputs)}")

        # SVF: SurfaceData.load() already embeds SVF + shadows on the surface.
        # Only use PrecomputedData if the user explicitly overrides with SVF_DIR.
        precomputed = None
        svf_dir = parameters.get("SVF_DIR") or None
        if svf_dir:
            feedback.pushInfo(f"Loading SVF override from {svf_dir}")
            try:
                precomputed = solweig.PrecomputedData.prepare(svf_dir=svf_dir)
            except Exception as e:
                feedback.reportError(
                    f"Could not load SVF from {svf_dir}: {e}",
                    fatalError=False,
                )
        elif surface.svf is not None:
            feedback.pushInfo("Using SVF from prepared surface")

        if feedback.isCanceled():
            return {}

        # Auto-fallback: anisotropic sky requires precomputed shadow matrices
        if use_anisotropic_sky:
            has_shadow = (precomputed is not None and precomputed.shadow_matrices is not None) or (
                surface.shadow_matrices is not None
            )
            if not has_shadow:
                feedback.reportError(
                    "Anisotropic sky requires pre-computed SVF with shadow matrices. "
                    "Falling back to isotropic sky model. To use anisotropic sky, "
                    "first run 'Compute Sky View Factor' and provide the SVF directory.",
                    fatalError=False,
                )
                use_anisotropic_sky = False

        # Step 5: Validate inputs
        feedback.setProgressText("Validating inputs...")
        feedback.setProgress(20)

        try:
            warnings = solweig.validate_inputs(
                surface=surface,
                location=location,
                weather=weather_series[0],
                use_anisotropic_sky=use_anisotropic_sky,
                precomputed=precomputed,
            )
            for warning in warnings:
                feedback.reportError(f"Warning: {warning}", fatalError=False)
        except solweig.SolweigError as e:
            from ..base import format_solweig_error

            raise QgsProcessingException(format_solweig_error("Validation failed", e)) from e

        if feedback.isCanceled():
            return {}

        # Step 6: Run calculation
        os.makedirs(output_dir, exist_ok=True)

        n_steps = len(weather_series)
        feedback.setProgressText(
            "Calculating Mean Radiant Temperature..." if is_single else f"Running timeseries ({n_steps} timesteps)..."
        )
        feedback.setProgress(25)
        progress_state = {"completed": 0}

        def _qgis_progress(current: int, total: int) -> None:
            progress_state["completed"] = current
            if feedback.isCanceled():
                raise KeyboardInterrupt
            if current > 0 and weather_series:
                n_steps = len(weather_series)
                # current/total may be tile-granular (n_tiles * n_steps);
                # derive the timestep index from the modular position.
                timestep_idx = (current - 1) % n_steps
                timestamp_str = weather_series[timestep_idx].datetime.strftime("%Y-%m-%d %H:%M")
                feedback.setProgressText(f"Step {current}/{total} \u2014 {timestamp_str}")
            total_safe = max(total, 1)
            pct = 25 + int(55 * current / total_safe)
            feedback.setProgress(pct)

        summary = None
        try:
            summary = solweig.calculate(
                surface=surface,
                weather=weather_series,
                location=location,
                human=human,
                precomputed=precomputed,
                use_anisotropic_sky=use_anisotropic_sky,
                conifer=conifer,
                physics=physics,
                materials=materials,
                max_shadow_distance_m=max_shadow_distance_m,
                output_dir=output_dir,
                outputs=selected_outputs,
                heat_thresholds_day=heat_thresholds_day or None,
                heat_thresholds_night=heat_thresholds_night or None,
                progress_callback=_qgis_progress,
            )
        except KeyboardInterrupt:
            feedback.pushInfo("Calculation cancelled by user.")
        except solweig.SolweigError as e:
            from ..base import format_solweig_error

            raise QgsProcessingException(format_solweig_error("Calculation failed", e)) from e
        except Exception as e:
            raise QgsProcessingException(f"Calculation failed: {e}") from e

        if feedback.isCanceled():
            return {}

        n_timesteps = summary.n_timesteps if summary is not None else 0
        calc_elapsed = time.time() - start_time
        feedback.pushInfo(f"Calculation complete: {n_timesteps} timestep(s) in {calc_elapsed:.1f}s")

        # Step 7: Add first Tmrt to canvas (single timestep only)
        if is_single:
            tmrt_files = sorted(Path(output_dir, "tmrt").glob("tmrt_*.tif"))
            if tmrt_files:
                timestamp_str = weather_series[0].datetime.strftime("%Y-%m-%d %H:%M")
                self.add_raster_to_canvas(
                    path=str(tmrt_files[0]),
                    layer_name=f"Tmrt {timestamp_str}",
                    style="tmrt",
                    context=context,
                )

        # Report summary
        feedback.setProgress(80)
        total_elapsed = time.time() - start_time
        utci_count = n_timesteps if "utci" in selected_outputs else 0
        pet_count = n_timesteps if "pet" in selected_outputs else 0
        tmrt_stats = {}
        if summary is not None and summary.tmrt_mean is not None:
            valid = summary.tmrt_mean[~np.isnan(summary.tmrt_mean)]
            if valid.size > 0:
                tmrt_stats = {
                    "mean": float(valid.mean()),
                    "min": float(np.nanmin(summary.tmrt_min)) if summary.tmrt_min is not None else float(valid.min()),
                    "max": float(np.nanmax(summary.tmrt_max)) if summary.tmrt_max is not None else float(valid.max()),
                }
            if summary.report is not None:
                feedback.pushInfo("")
                for line in summary.report().splitlines():
                    feedback.pushInfo(line)

        self._report_summary(n_timesteps, total_elapsed, utci_count, pet_count, output_dir, feedback, tmrt_stats)

        return {
            "OUTPUT_FOLDER": output_dir,
            "TIMESTEP_COUNT": n_timesteps,
            "UTCI_COUNT": utci_count,
            "PET_COUNT": pet_count,
        }

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_thresholds(raw: str | None) -> list[float]:
        """Parse a comma-separated string of temperature thresholds into a list of floats."""
        if not raw or not raw.strip():
            return []
        values = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                with contextlib.suppress(ValueError):
                    values.append(float(part))
        return values

    @staticmethod
    def _report_summary(
        n_timesteps,
        elapsed,
        utci_count,
        pet_count,
        output_dir,
        feedback,
        tmrt_stats=None,
    ) -> None:
        """Report calculation summary statistics.

        Args:
            n_timesteps: Number of timesteps processed.
            elapsed: Total elapsed time in seconds.
            utci_count: Number of UTCI files created.
            pet_count: Number of PET files created.
            output_dir: Output directory path.
            feedback: QGIS feedback object.
            tmrt_stats: Dict with 'mean', 'min', 'max' Tmrt values (optional).
        """
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Calculation complete!")
        feedback.pushInfo(f"  Timesteps: {n_timesteps}")
        feedback.pushInfo(f"  Total time: {elapsed:.1f} seconds")

        if n_timesteps > 1:
            feedback.pushInfo(f"  Per timestep: {elapsed / n_timesteps:.2f} seconds")

        if tmrt_stats:
            feedback.pushInfo(f"  Tmrt range: {tmrt_stats['min']:.1f}C - {tmrt_stats['max']:.1f}C")
            feedback.pushInfo(f"  Mean Tmrt: {tmrt_stats['mean']:.1f}C")

        if utci_count > 0:
            feedback.pushInfo(f"  UTCI files: {utci_count}")
        if pet_count > 0:
            feedback.pushInfo(f"  PET files: {pet_count}")

        feedback.pushInfo(f"  Output: {output_dir}")
        feedback.pushInfo("=" * 60)

        feedback.setProgress(100)
