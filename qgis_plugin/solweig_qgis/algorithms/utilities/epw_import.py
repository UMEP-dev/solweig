"""
EPW Weather File Tool

Download EPW files from PVGIS or preview/validate existing EPW files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingOutputHtml,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)

from ...utils.parameters import _canvas_center_latlon
from ..base import SolweigAlgorithmBase


class EpwImportAlgorithm(SolweigAlgorithmBase):
    """
    Download EPW files from PVGIS or preview existing EPW files.

    In download mode, fetches a Typical Meteorological Year (TMY) EPW
    file from the EU PVGIS service for any location (no API key needed).

    In preview mode, displays location, date range, and data statistics
    for an existing EPW file.
    """

    # Mode enum values
    MODE_DOWNLOAD = 0
    MODE_PREVIEW = 1

    def name(self) -> str:
        return "epw_import"

    def displayName(self) -> str:
        return self.tr("1. Download / Preview Weather File")

    def shortHelpString(self) -> str:
        return self.tr(
            """Download or preview EnergyPlus Weather (EPW) files.

<b>Download mode:</b>
Downloads a Typical Meteorological Year (TMY) EPW file from the EU
PVGIS service (no API key required). Near-global coverage using
ERA5 reanalysis data.

A TMY is a statistical composite — each calendar month is selected
from the most "typical" month across a multi-year reference period
(2005–2023 for PVGIS v5.3). The result represents long-term average
climate conditions, <i>not</i> observations from a specific year.
Because each month is taken from a real historical year, the row
timestamps in the downloaded file will legitimately span multiple
years within the reference period (e.g. a 2022 January alongside a
2014 February). This is the defining characteristic of a TMY, not
an error. Data freshness depends on the upstream PVGIS release,
not on SOLWEIG.

Enter latitude and longitude, and the file will be downloaded and
saved to the specified output path.

<b>Preview mode:</b>
Inspect an existing EPW file before running SOLWEIG calculations.
Generates an HTML report with location, date range, and data statistics.

<b>EPW files contain hourly data including:</b>
<ul>
<li>Air temperature, relative humidity</li>
<li>Wind speed and direction</li>
<li>Solar radiation (global, direct, diffuse)</li>
<li>Atmospheric pressure</li>
</ul>

<b>Data source:</b>
PVGIS (Photovoltaic Geographical Information System) by the
EU Joint Research Centre. Data derived from ERA5 reanalysis.

<b>Documentation:</b>
<ul>
<li><a href="https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en"
>PVGIS TMY Methodology</a></li>
<li><a href="https://umep-dev.github.io/solweig/guide/qgis-plugin/">QGIS Plugin Guide</a></li>
<li><a href="https://umep-dev.github.io/solweig/getting-started/quick-start/">Quick Start</a></li>
<li><a href="https://umep-dev.github.io/solweig/">SOLWEIG Documentation</a></li>
</ul>"""
        )

    def group(self) -> str:
        return ""

    def groupId(self) -> str:
        return ""

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""
        # Mode selector
        self.addParameter(
            QgsProcessingParameterEnum(
                "MODE",
                self.tr("Mode"),
                options=["Download EPW from PVGIS", "Preview existing EPW file"],
                defaultValue=self.MODE_DOWNLOAD,
            )
        )

        # Download parameters — default to map canvas centre so users
        # don't accidentally download weather for the wrong location.
        canvas_lat, canvas_lon = _canvas_center_latlon()

        self.addParameter(
            QgsProcessingParameterNumber(
                "LATITUDE",
                self.tr("Latitude (for download)"),
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=canvas_lat,
                minValue=-90.0,
                maxValue=90.0,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                "LONGITUDE",
                self.tr("Longitude (for download)"),
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=canvas_lon,
                minValue=-180.0,
                maxValue=180.0,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                "OUTPUT_EPW",
                self.tr("Save EPW file to (for download)"),
                fileFilter="EPW files (*.epw)",
                optional=True,
            )
        )

        # Preview parameters
        self.addParameter(
            QgsProcessingParameterFile(
                "EPW_FILE",
                self.tr("EPW weather file (for preview)"),
                extension="epw",
                optional=True,
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFile(
                "DOWNLOADED_FILE",
                self.tr("Downloaded EPW file"),
            )
        )

        self.addOutput(
            QgsProcessingOutputHtml(
                "OUTPUT_HTML",
                self.tr("EPW Information Report"),
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Execute the algorithm."""
        mode = self.parameterAsEnum(parameters, "MODE", context)

        if mode == self.MODE_DOWNLOAD:
            return self._download_epw(parameters, context, feedback)
        else:
            return self._preview_epw(parameters, context, feedback)

    def _download_epw(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Download EPW from PVGIS and generate preview report."""
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("EPW Download from PVGIS (Typical Meteorological Year)")
        feedback.pushInfo("=" * 60)

        # Import solweig
        self.import_solweig()
        from solweig.io import read_epw

        # Get parameters
        latitude = self.parameterAsDouble(parameters, "LATITUDE", context)
        longitude = self.parameterAsDouble(parameters, "LONGITUDE", context)
        output_path = self.parameterAsFileOutput(parameters, "OUTPUT_EPW", context)

        if not output_path:
            output_path = str(Path(tempfile.gettempdir()) / f"pvgis_{latitude:.2f}_{longitude:.2f}.epw")

        if not -90 <= latitude <= 90:
            raise QgsProcessingException(f"Latitude must be between -90 and 90, got {latitude}")
        if not -180 <= longitude <= 180:
            raise QgsProcessingException(f"Longitude must be between -180 and 180, got {longitude}")

        feedback.pushInfo(f"Location: {latitude:.4f}N, {longitude:.4f}E")
        feedback.pushInfo(f"Output: {output_path}")
        feedback.pushInfo("")
        feedback.setProgressText("Downloading from PVGIS...")
        feedback.setProgress(10)

        from qgis.core import QgsBlockingNetworkRequest
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtNetwork import QNetworkRequest

        url = f"https://re.jrc.ec.europa.eu/api/v5_3/tmy?lat={latitude}&lon={longitude}&outputformat=epw"
        req = QgsBlockingNetworkRequest()
        err = req.get(QNetworkRequest(QUrl(url)))

        if err != QgsBlockingNetworkRequest.NoError:
            error_msg = req.errorMessage()
            raise QgsProcessingException(
                f"PVGIS download failed: {error_msg}\n"
                f"URL: {url}\n"
                "Check your internet connection and QGIS proxy settings."
            )

        reply = req.reply()
        http_status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        data = bytes(reply.content())

        if http_status == 400:
            raise QgsProcessingException(
                f"PVGIS has no data for ({latitude}, {longitude}). The location may be over ocean or outside coverage."
            )
        if http_status and http_status != 200:
            raise QgsProcessingException(f"PVGIS download failed (HTTP {http_status})")

        if len(data) < 1000:
            text = data.decode("utf-8", errors="replace")
            raise QgsProcessingException(f"PVGIS returned an error: {text.strip()}")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)

        feedback.setProgress(60)
        feedback.pushInfo(f"Downloaded EPW file: {output_path}")
        feedback.pushInfo(
            "Note: This is a Typical Meteorological Year (TMY) file — a statistical"
            " composite of typical months from the PVGIS v5.3 reference period (2005-2023),"
            " not observations from a specific year. Row timestamps will span multiple"
            " years within that window because each month is a real historical month."
        )

        # Generate preview report
        feedback.setProgressText("Generating report...")
        try:
            df, metadata = read_epw(output_path)
        except Exception as e:
            raise QgsProcessingException(f"Error reading downloaded EPW: {e}") from e

        feedback.pushInfo("")
        feedback.pushInfo(f"Location: {metadata.get('city', 'Unknown')}")
        feedback.pushInfo(f"Coordinates: {metadata.get('latitude', 'N/A')}N, {metadata.get('longitude', 'N/A')}E")
        feedback.pushInfo(f"Data range: {df.index.min()} to {df.index.max()}")
        feedback.pushInfo(f"Timesteps: {len(df)}")

        html = self._generate_html_report(df, metadata, output_path)
        html_path = str(Path(tempfile.gettempdir()) / f"epw_report_{Path(output_path).stem}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        feedback.setProgress(100)
        feedback.pushInfo("")
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("Download complete!")
        feedback.pushInfo(f"  EPW file: {output_path}")
        feedback.pushInfo(f"  Report: {html_path}")
        feedback.pushInfo("=" * 60)

        return {
            "DOWNLOADED_FILE": output_path,
            "OUTPUT_HTML": html_path,
        }

    def _preview_epw(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Preview an existing EPW file."""
        feedback.pushInfo("=" * 60)
        feedback.pushInfo("EPW Weather File Preview")
        feedback.pushInfo("=" * 60)

        # Import solweig
        self.import_solweig()
        from solweig.io import read_epw

        # Get parameters
        epw_path = self.parameterAsFile(parameters, "EPW_FILE", context)
        if not epw_path:
            raise QgsProcessingException("No EPW file specified for preview mode")

        feedback.pushInfo(f"Reading: {epw_path}")

        # Read EPW file
        try:
            df, metadata = read_epw(epw_path)
        except FileNotFoundError as e:
            raise QgsProcessingException(f"EPW file not found: {epw_path}") from e
        except Exception as e:
            raise QgsProcessingException(f"Error reading EPW file: {e}") from e

        # Display key info
        feedback.pushInfo("")
        feedback.pushInfo(f"Location: {metadata.get('city', 'Unknown')}")
        feedback.pushInfo(f"Coordinates: {metadata.get('latitude', 'N/A')}N, {metadata.get('longitude', 'N/A')}E")
        feedback.pushInfo(f"Elevation: {metadata.get('elevation', 'N/A')} m")
        feedback.pushInfo(f"UTC offset: {metadata.get('tz_offset', 'N/A')} hours")
        feedback.pushInfo("")
        feedback.pushInfo(f"Data range: {df.index.min()} to {df.index.max()}")
        feedback.pushInfo(f"Timesteps: {len(df)}")

        # Generate HTML report
        html = self._generate_html_report(df, metadata, epw_path)

        # Save to temp file
        output_html = str(Path(tempfile.gettempdir()) / f"epw_report_{Path(epw_path).stem}.html")
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)

        feedback.pushInfo("")
        feedback.pushInfo(f"Report saved: {output_html}")

        return {"OUTPUT_HTML": output_html}

    @staticmethod
    def _column_stats(df, col: str) -> tuple:
        """Compute (min, max, mean, missing_count) for a column.

        Works with both pandas DataFrames and the lightweight _EpwDataFrame.
        """
        import math

        values = df[col]
        valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        n_missing = len(values) - len(valid)
        if valid:
            return min(valid), max(valid), sum(valid) / len(valid), n_missing
        return 0.0, 0.0, 0.0, n_missing

    def _generate_html_report(self, df, metadata: dict, epw_path: str) -> str:
        """Generate HTML report for EPW file."""
        # Map column names to friendly names
        column_names = {
            "temp_air": "Air Temperature (°C)",
            "relative_humidity": "Relative Humidity (%)",
            "wind_speed": "Wind Speed (m/s)",
            "ghi": "Global Horizontal Irradiance (W/m²)",
            "dni": "Direct Normal Irradiance (W/m²)",
            "dhi": "Diffuse Horizontal Irradiance (W/m²)",
            "atmospheric_pressure": "Atmospheric Pressure (Pa)",
        }

        # Build statistics table rows
        stats_rows = ""
        has_missing = False
        for col in ["temp_air", "relative_humidity", "wind_speed", "ghi"]:
            if col in df.columns:
                friendly_name = column_names.get(col, col)
                col_min, col_max, col_mean, col_missing = self._column_stats(df, col)
                if col_missing > 0:
                    has_missing = True
                stats_rows += f"""
                <tr>
                    <td>{friendly_name}</td>
                    <td>{col_min:.1f}</td>
                    <td>{col_max:.1f}</td>
                    <td>{col_mean:.1f}</td>
                    <td>{col_missing}</td>
                </tr>
                """

        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EPW Weather File Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{ color: #333; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:hover {{ background: #f5f5f5; }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .info {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .file-path {{
            background: #eee;
            padding: 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>EPW Weather File Report</h1>

    <div class="card">
        <h2>File Information</h2>
        <div class="file-path">{epw_path}</div>
    </div>

    <div class="card">
        <h2>Location</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>City</td><td>{metadata.get("city", "Unknown")}</td></tr>
            <tr><td>State/Province</td><td>{metadata.get("state", "-")}</td></tr>
            <tr><td>Country</td><td>{metadata.get("country", "Unknown")}</td></tr>
            <tr><td>Latitude</td><td>{metadata.get("latitude", "N/A")}&deg; N</td></tr>
            <tr><td>Longitude</td><td>{metadata.get("longitude", "N/A")}&deg; E</td></tr>
            <tr><td>Elevation</td><td>{metadata.get("elevation", "N/A")} m</td></tr>
            <tr><td>UTC Offset</td><td>UTC{metadata.get("tz_offset", 0):+.0f}</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>Data Range</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Start Date</td><td>{df.index.min()}</td></tr>
            <tr><td>End Date</td><td>{df.index.max()}</td></tr>
            <tr><td>Total Timesteps</td><td>{len(df):,}</td></tr>
            <tr><td>Timestep Interval</td><td>Hourly</td></tr>
        </table>
    </div>

    <div class="info">
        <strong>About Typical Meteorological Year (TMY) data:</strong>
        A TMY file is a statistical composite &mdash; each calendar month is selected
        from the most &ldquo;typical&rdquo; month across a multi-year reference period
        (2005&ndash;2023 for PVGIS v5.3). It represents long-term average climate conditions,
        not observations from any single year. Individual row timestamps in the file
        will legitimately span multiple years within the reference window (e.g. a
        January from 2022 alongside a February from 2014), because each month is a
        real historical month selected as the most typical. This is the defining
        property of a TMY and not an artefact.
        See the
        <a href="https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/pvgis-typical-meteorological-year-tmy-generator_en"
>PVGIS TMY documentation</a>
        for full methodology and data sources.
    </div>

    <div class="card">
        <h2>Data Statistics</h2>
        <table>
            <tr>
                <th>Variable</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Missing</th>
            </tr>
            {stats_rows}
        </table>
    </div>

    {
            "<div class='warning'><strong>Warning:</strong> "
            "Some variables have missing values. Check the Missing column above.</div>"
            if has_missing
            else ""
        }

    <div class="info">
        <strong>Next Steps:</strong>
        <ul>
            <li>Use this EPW file with the "SOLWEIG Calculation" algorithm</li>
            <li>Set weather source to "EPW weather file"</li>
            <li>UTC offset is important for accurate sun position calculation</li>
            <li>Consider filtering hours (e.g., 9-17) for daylight-only analysis</li>
        </ul>
    </div>

    <div class="card">
        <h2>SOLWEIG Compatibility</h2>
        <p>This EPW file is compatible with SOLWEIG calculations. The following
        variables will be used:</p>
        <ul>
            <li><strong>temp_air</strong>: Air temperature for Tmrt and thermal comfort</li>
            <li><strong>relative_humidity</strong>: For UTCI and PET calculations</li>
            <li><strong>ghi</strong>: Global solar radiation for shortwave radiation</li>
            <li><strong>wind_speed</strong>: For UTCI and PET calculations</li>
        </ul>
    </div>

    <div class="card">
        <h2>Documentation</h2>
        <ul>
            <li><a href="https://umep-dev.github.io/solweig/guide/qgis-plugin/">QGIS Plugin Guide</a></li>
            <li><a href="https://umep-dev.github.io/solweig/getting-started/quick-start/">Quick Start</a></li>
            <li><a href="https://umep-dev.github.io/solweig/">SOLWEIG Documentation</a></li>
        </ul>
    </div>

</body>
</html>
        """

        return html
