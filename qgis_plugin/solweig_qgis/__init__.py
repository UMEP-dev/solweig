"""
SOLWEIG QGIS Plugin

Provides QGIS Processing algorithms for calculating Mean Radiant Temperature (Tmrt),
UTCI, and PET thermal comfort indices using the SOLWEIG model.

Adapted from UMEP (Urban Multi-scale Environmental Predictor).
Original code by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, and Nils Wallenberg.

Citation:
    Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L,
    Ward H, Capel-Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F,
    Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor
    (UMEP) - An integrated tool for city-based climate services.
    Environmental Modelling and Software 99, 70-87
    https://doi.org/10.1016/j.envsoft.2017.09.020
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).resolve().parent


def _read_required_version() -> str:
    """
    Read the required solweig version from metadata.txt.

    The plugin version in metadata.txt is kept in sync with the solweig library
    version by build_plugin.py, which reads pyproject.toml as the single source
    of truth. The QGIS metadata format uses hyphens (0.1.0-beta5) while PEP 440
    uses letters (0.1.0b5), so we normalize here.
    """
    import configparser

    metadata_path = _PLUGIN_DIR / "metadata.txt"
    config = configparser.ConfigParser()
    config.read(metadata_path)
    qgis_version = config.get("general", "version", fallback="0.0.0")

    # Normalize QGIS format (0.1.0-beta5) to PEP 440 (0.1.0b5)
    import re

    normalized = re.sub(r"-?alpha", "a", qgis_version)
    normalized = re.sub(r"-?beta", "b", normalized)
    normalized = re.sub(r"-?rc", "rc", normalized)
    return normalized


_REQUIRED_SOLWEIG_VERSION = _read_required_version()
_SOLWEIG_AVAILABLE = False
_SOLWEIG_OUTDATED = False  # True when installed but too old
_SOLWEIG_SOURCE = None  # "system", "development", or None
_SOLWEIG_IMPORT_ERROR = None
_SOLWEIG_INSTALLED_VERSION = None


def _parse_version(version_str: str) -> tuple:
    """
    Parse a PEP 440 version string into a comparable tuple.

    Handles release versions (0.1.0) and pre-release versions (0.1.0b5, 0.1.0a1, 0.1.0rc1).
    Pre-release versions sort before their release (0.1.0b5 < 0.1.0).
    """
    import re

    match = re.match(r"^(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?", version_str)
    if not match:
        return (0, 0, 0, "z", 0)  # unparseable sorts high to avoid false outdated

    release = tuple(int(x) for x in match.group(1).split("."))
    pre_type = match.group(2)  # "a", "b", "rc", or None
    pre_num = int(match.group(3)) if match.group(3) else 0

    # "z" sorts after "a", "b", "rc" — so final releases are higher than pre-releases
    pre_key = pre_type if pre_type else "z"
    return release + (pre_key, pre_num)


def _check_version(solweig_module) -> bool:
    """
    Check if the imported solweig module meets the minimum version requirement.

    Sets _SOLWEIG_OUTDATED and _SOLWEIG_IMPORT_ERROR if the version is too old.

    Returns:
        True if version is acceptable, False if outdated.
    """
    global _SOLWEIG_OUTDATED, _SOLWEIG_IMPORT_ERROR, _SOLWEIG_INSTALLED_VERSION

    installed = getattr(solweig_module, "__version__", None) or "0.0.0"
    _SOLWEIG_INSTALLED_VERSION = installed

    # Version check (prefer robust PEP 440 parsing when available)
    try:
        from packaging.version import Version

        if Version(installed) < Version(_REQUIRED_SOLWEIG_VERSION):
            _SOLWEIG_OUTDATED = True
            _SOLWEIG_IMPORT_ERROR = (
                f"solweig {installed} is installed but this plugin requires >= {_REQUIRED_SOLWEIG_VERSION}"
            )
            return False
    except Exception:
        # Fallback for minimal environments (should be rare)
        if _parse_version(installed) < _parse_version(_REQUIRED_SOLWEIG_VERSION):
            _SOLWEIG_OUTDATED = True
            _SOLWEIG_IMPORT_ERROR = (
                f"solweig {installed} is installed but this plugin requires >= {_REQUIRED_SOLWEIG_VERSION}"
            )
            return False

    # Feature check: ensure the imported SurfaceData supports the API used by this plugin.
    # This guards against environments where a different/old `solweig` package is importable
    # (or where version strings are missing/non-standard).
    missing: list[str] = []
    surface_cls = getattr(solweig_module, "SurfaceData", None)
    if surface_cls is None:
        missing.append("SurfaceData")
    else:
        for method_name in ("preprocess", "fill_nan", "compute_valid_mask", "apply_valid_mask", "crop_to_valid_bbox"):
            if not hasattr(surface_cls, method_name):
                missing.append(f"SurfaceData.{method_name}()")

    if missing:
        _SOLWEIG_OUTDATED = True
        _SOLWEIG_IMPORT_ERROR = (
            "The imported solweig package is missing required APIs: "
            + ", ".join(missing)
            + f". Please upgrade solweig to >= {_REQUIRED_SOLWEIG_VERSION} and restart QGIS."
        )
        return False

    return True


def _setup_solweig_path():
    """
    Set up the import path for solweig library.

    Priority:
    1. System-installed solweig (via pip)
    2. Development path (for local development)
    """
    global _SOLWEIG_AVAILABLE, _SOLWEIG_OUTDATED, _SOLWEIG_SOURCE, _SOLWEIG_IMPORT_ERROR, _SOLWEIG_INSTALLED_VERSION

    # Already found in a previous call
    if _SOLWEIG_AVAILABLE:
        return

    def _try_import_system() -> bool:
        global _SOLWEIG_AVAILABLE, _SOLWEIG_SOURCE, _SOLWEIG_IMPORT_ERROR
        try:
            import solweig  # noqa: F401

            if not _check_version(solweig):
                return False
            _SOLWEIG_AVAILABLE = True
            _SOLWEIG_SOURCE = "system"
            _SOLWEIG_IMPORT_ERROR = None
            return True
        except Exception as exc:
            # Catch all exceptions, not just ImportError — older solweig versions
            # may crash on import (e.g. rasterio incompatibility) and we must
            # still load the plugin so the upgrade prompt can appear.
            _SOLWEIG_IMPORT_ERROR = f"system import failed: {exc}"
            return False

    def _try_import_dev(dev_path: Path) -> bool:
        global _SOLWEIG_AVAILABLE, _SOLWEIG_SOURCE, _SOLWEIG_IMPORT_ERROR
        if not (dev_path.exists() and (dev_path / "solweig").exists()):
            return False
        inserted = False
        if str(dev_path) not in sys.path:
            sys.path.insert(0, str(dev_path))
            inserted = True
        try:
            # If solweig was already imported (e.g. from a system install),
            # remove the cached module so Python re-discovers it from pysrc/.
            if "solweig" in sys.modules:
                # Remove the main module and all submodules so the fresh
                # import picks up the development source tree.
                stale = [k for k in sys.modules if k == "solweig" or k.startswith("solweig.")]
                for k in stale:
                    del sys.modules[k]

            import solweig  # noqa: F401

            if not _check_version(solweig):
                return False
            _SOLWEIG_AVAILABLE = True
            _SOLWEIG_SOURCE = "development"
            _SOLWEIG_IMPORT_ERROR = None
            return True
        except Exception as exc:
            _SOLWEIG_IMPORT_ERROR = f"development import failed: {exc}"
            return False
        finally:
            # If dev import didn't succeed, keep sys.path clean.
            if not _SOLWEIG_AVAILABLE and inserted and str(dev_path) in sys.path:
                sys.path.remove(str(dev_path))

    # Development mode - look for pysrc in parent directories
    dev_paths = [
        _PLUGIN_DIR.parent.parent / "pysrc",  # repo_root/pysrc
        _PLUGIN_DIR.parent.parent.parent / "pysrc",  # One more level up
    ]

    # If we're running from a repository checkout (symlinked plugin), prefer local pysrc
    # to avoid accidentally using an older system-installed solweig.
    repo_root = _PLUGIN_DIR.parent.parent
    prefer_dev = (repo_root / "pyproject.toml").exists() and (repo_root / "pysrc" / "solweig").exists()

    if prefer_dev:
        for dev_path in dev_paths:
            if _try_import_dev(dev_path):
                return
        if _try_import_system():
            return
    else:
        if _try_import_system():
            return
        for dev_path in dev_paths:
            if _try_import_dev(dev_path):
                return

    # No solweig found
    _SOLWEIG_AVAILABLE = False
    _SOLWEIG_SOURCE = None
    if _SOLWEIG_IMPORT_ERROR is None:
        _SOLWEIG_IMPORT_ERROR = "solweig package not installed"


def _install_solweig() -> tuple[bool, str]:
    """
    Install or upgrade solweig via pip in-process.

    Uses pip's internal API rather than subprocess because QGIS embeds Python
    and sys.executable points to the QGIS binary, not a usable Python interpreter.
    See: https://github.com/qgis/QGIS/issues/45646

    ``--no-deps`` is used because QGIS ships its own GDAL; pulling pip's
    GDAL in as a transitive dep crashes QGIS. solweig's GDAL path only
    needs numpy and osgeo (both bundled with QGIS), and the other pip-
    declared deps (rasterio, pyproj, shapely, tqdm, pillow) are all
    guarded behind branches that are never taken when
    ``UMEP_USE_GDAL=1`` is set by the plugin.

    Returns:
        Tuple of (success, message).
    """
    import contextlib
    import io

    try:
        from pip._internal.cli.main import main as pip_main
    except ImportError:
        return False, "pip is not available in this QGIS Python environment."

    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exit_code = pip_main(["install", "--upgrade", "--no-deps", "solweig"])
        if exit_code == 0:
            return True, "SOLWEIG installed successfully."
        return False, f"pip install failed (exit code {exit_code}):\n{output.getvalue()}"
    except Exception as e:
        return False, f"Installation failed: {e}"


# Force GDAL backend before importing solweig — we know we're in QGIS,
# so bypass _compat.py heuristic detection to avoid rasterio/numpy crashes.
os.environ["UMEP_USE_GDAL"] = "1"

# Run setup on module load
_setup_solweig_path()


def check_dependencies() -> tuple[bool, str]:
    """
    Check if all required dependencies are available.

    Returns:
        Tuple of (success, message)
    """
    if _SOLWEIG_AVAILABLE:
        return True, f"SOLWEIG library loaded ({_SOLWEIG_SOURCE})"

    if _SOLWEIG_OUTDATED:
        msg = (
            f"SOLWEIG {_SOLWEIG_INSTALLED_VERSION} is installed but this plugin "
            f"requires >= {_REQUIRED_SOLWEIG_VERSION}.\n\n"
            "To upgrade manually:\n\n"
            "  In OSGeo4W Shell (Windows) or Terminal (macOS/Linux):\n"
            "  pip install --upgrade --no-deps solweig\n\n"
            "After upgrading, restart QGIS."
        )
        return False, msg

    error_hint = f"\nLast import error: {_SOLWEIG_IMPORT_ERROR}\n" if _SOLWEIG_IMPORT_ERROR else ""

    msg = f"""SOLWEIG library not found.{error_hint}

To install SOLWEIG manually:

  In OSGeo4W Shell (Windows) or Terminal (macOS/Linux):
  pip install --no-deps solweig

After installation, restart QGIS and re-enable the plugin.
"""
    return False, msg


def _prompt_install():
    """Offer to auto-install or upgrade solweig if it's missing or outdated."""
    global _SOLWEIG_AVAILABLE, _SOLWEIG_OUTDATED, _SOLWEIG_SOURCE, _SOLWEIG_IMPORT_ERROR, _SOLWEIG_INSTALLED_VERSION

    success, message = check_dependencies()
    if success:
        return

    try:
        from qgis.PyQt.QtWidgets import QMessageBox

        if _SOLWEIG_OUTDATED:
            title = "SOLWEIG Plugin - Update Required"
            prompt = (
                f"SOLWEIG {_SOLWEIG_INSTALLED_VERSION} is installed but this plugin "
                f"requires >= {_REQUIRED_SOLWEIG_VERSION}.\n\n"
                "Would you like to upgrade now?\n\n"
                "This will run:  pip install --upgrade --no-deps solweig"
            )
            decline_msg = (
                "SOLWEIG was not upgraded. You can upgrade manually:\n\n"
                "  pip install --upgrade --no-deps solweig\n\n"
                "Then restart QGIS."
            )
        else:
            title = "SOLWEIG Plugin - Install Dependencies"
            prompt = (
                "The SOLWEIG library is required but not installed.\n\n"
                "Would you like to install it now?\n\n"
                "This will run:  pip install --no-deps solweig"
            )
            decline_msg = (
                "SOLWEIG was not installed. You can install it manually:\n\n  "
                "pip install --no-deps solweig\n\nThen restart QGIS."
            )

        reply = QMessageBox.question(
            None,
            title,
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            QMessageBox.information(None, "SOLWEIG Plugin", decline_msg)
            return

        # Show a wait cursor while installing
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, install_msg = _install_solweig()
        finally:
            QApplication.restoreOverrideCursor()

        if ok:
            # Reset state so _setup_solweig_path() can re-check
            _SOLWEIG_AVAILABLE = False
            _SOLWEIG_OUTDATED = False
            _SOLWEIG_IMPORT_ERROR = None
            _SOLWEIG_INSTALLED_VERSION = None

            # Reload the module if it was already imported (upgrade case)
            if "solweig" in sys.modules:
                import importlib

                importlib.reload(sys.modules["solweig"])

            _setup_solweig_path()
            if _SOLWEIG_AVAILABLE:
                QMessageBox.information(
                    None,
                    "SOLWEIG Plugin",
                    "SOLWEIG installed successfully! The plugin is ready to use.",
                )
            else:
                QMessageBox.information(
                    None,
                    "SOLWEIG Plugin",
                    "SOLWEIG installed successfully.\n\nPlease restart QGIS to complete setup.",
                )
        else:
            QMessageBox.warning(
                None,
                "SOLWEIG Plugin - Installation Failed",
                f"{install_msg}\n\nYou can try installing manually:\n\n  pip install solweig\n\nThen restart QGIS.",
            )

    except ImportError:
        # Not in QGIS environment
        print(f"WARNING: {message}")


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

from .provider import SolweigProvider  # noqa: E402


def classFactory(iface):
    """
    QGIS plugin entry point.

    Called by QGIS when the plugin is loaded. Returns the provider instance
    that will register all processing algorithms.

    Args:
        iface: QgisInterface instance providing access to QGIS components.

    Returns:
        SolweigPlugin instance that manages the processing provider.
    """
    return SolweigPlugin(iface)


class SolweigPlugin:
    """
    Main plugin class that manages the SOLWEIG processing provider.

    This class handles plugin lifecycle (load/unload) and registers
    the SolweigProvider with QGIS Processing framework.
    """

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        """Initialize the processing provider."""
        from qgis.core import QgsApplication

        self.provider = SolweigProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Initialize the plugin GUI (called when plugin is activated)."""
        # Register the Processing provider first — unconditionally — so
        # SOLWEIG always appears in the Processing Toolbox even when the
        # library isn't installed yet.  Showing a QMessageBox during
        # initGui() can fail or block on some platforms (especially macOS),
        # which would prevent initProcessing() from ever being called.
        self.initProcessing()

        if not _SOLWEIG_AVAILABLE or _SOLWEIG_OUTDATED:
            # Defer the install prompt to after the event loop starts,
            # so it doesn't block plugin registration.
            from qgis.PyQt.QtCore import QTimer

            QTimer.singleShot(500, _prompt_install)

    def unload(self):
        """Unload the plugin (called when plugin is deactivated)."""
        from qgis.core import QgsApplication

        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)


# ---------------------------------------------------------------------------
# Module-level info for debugging
# ---------------------------------------------------------------------------

__solweig_available__ = _SOLWEIG_AVAILABLE
__solweig_source__ = _SOLWEIG_SOURCE
