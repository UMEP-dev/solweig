"""
SOLWEIG Processing Provider

Registers all SOLWEIG algorithms with the QGIS Processing framework.
"""

import importlib
import os

from qgis.core import Qgis, QgsMessageLog, QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon


class SolweigProvider(QgsProcessingProvider):
    """
    QGIS Processing provider for SOLWEIG algorithms.

    Algorithms (in workflow order):
    1. Download / Preview Weather File
    2. Prepare Surface Data (align, walls, SVF)
    3. SOLWEIG Calculation
    """

    def id(self):
        """Unique provider ID used in processing scripts."""
        return "solweig"

    def name(self):
        """Display name shown in Processing Toolbox."""
        return "SOLWEIG"

    def longName(self):
        """Extended name for provider description."""
        return "SOLWEIG - Solar and Longwave Environmental Irradiance Geometry"

    def icon(self):
        """Provider icon shown in Processing Toolbox."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QgsProcessingProvider.icon(self)

    # Algorithm modules in workflow order: (relative module path, class name).
    _ALGORITHMS = (
        (".algorithms.utilities.epw_import", "EpwImportAlgorithm"),
        (".algorithms.preprocess.surface_preprocessing", "SurfacePreprocessingAlgorithm"),
        (".algorithms.calculation.solweig_calculation", "SolweigCalculationAlgorithm"),
    )

    def loadAlgorithms(self):
        """
        Load and register all SOLWEIG algorithms.

        Called by QGIS when the provider is initialized. Each algorithm is
        imported independently so an ImportError in one module (e.g. a stale
        install or a missing optional dependency) is logged and skipped
        instead of preventing the remaining algorithms from registering.
        """
        for module_path, class_name in self._ALGORITHMS:
            try:
                module = importlib.import_module(module_path, package=__package__)
                algorithm_class = getattr(module, class_name)
            except ImportError as exc:
                QgsMessageLog.logMessage(
                    f"Could not load algorithm {class_name} from {module_path}: {exc}",
                    "SOLWEIG",
                    Qgis.MessageLevel.Warning,
                )
                continue
            self.addAlgorithm(algorithm_class())
