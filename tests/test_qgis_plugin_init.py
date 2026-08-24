"""
Tests for QGIS plugin package init and provider registration.

Covers two previously untested plugin surfaces:

1. Version helpers in ``qgis_plugin/solweig_qgis/__init__.py``:
   ``_parse_version``, ``_read_required_version`` (QGIS ``0.1.0-beta88``
   to PEP 440 ``0.1.0b88`` normalization), and ``_check_version``.
2. ``provider.py`` registration: ``loadAlgorithms()`` registers the
   expected algorithm ids, and an ImportError in one algorithm module
   is logged and skipped instead of aborting provider loading.

Uses the shared QGIS mocks so no QGIS installation is needed.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from tests.qgis_mocks import install, install_osgeo, preserve_solweig_modules, uninstall_osgeo

install()  # Must be called before any qgis_plugin imports
install_osgeo()

with preserve_solweig_modules():
    import qgis_plugin.solweig_qgis as plugin_pkg  # noqa: E402
    from qgis_plugin.solweig_qgis import provider as provider_module  # noqa: E402

    # Pre-import all algorithm modules so provider.loadAlgorithms() finds
    # them cached in sys.modules and the tests below never re-import solweig
    # (or osgeo) outside the preserve context.
    from qgis_plugin.solweig_qgis.algorithms.calculation import solweig_calculation  # noqa: E402, F401
    from qgis_plugin.solweig_qgis.algorithms.preprocess import surface_preprocessing  # noqa: E402, F401
    from qgis_plugin.solweig_qgis.algorithms.utilities import epw_import  # noqa: E402, F401

uninstall_osgeo()


# ---------------------------------------------------------------------------
# _parse_version
# ---------------------------------------------------------------------------


class TestParseVersion:
    """Tests for the PEP 440 fallback parser."""

    def test_beta_version_tuple(self):
        assert plugin_pkg._parse_version("0.1.0b88") == (0, 1, 0, "b", 88)

    def test_release_version_tuple(self):
        assert plugin_pkg._parse_version("0.1.0") == (0, 1, 0, "z", 0)

    def test_prerelease_sorts_before_release(self):
        assert plugin_pkg._parse_version("0.1.0b88") < plugin_pkg._parse_version("0.1.0")

    def test_prerelease_type_ordering(self):
        alpha = plugin_pkg._parse_version("0.1.0a1")
        beta = plugin_pkg._parse_version("0.1.0b1")
        rc = plugin_pkg._parse_version("0.1.0rc1")
        final = plugin_pkg._parse_version("0.1.0")
        assert alpha < beta < rc < final

    def test_prerelease_number_is_numeric_not_lexicographic(self):
        assert plugin_pkg._parse_version("0.1.0b9") < plugin_pkg._parse_version("0.1.0b88")

    def test_newer_release_wins_over_prerelease_of_older(self):
        assert plugin_pkg._parse_version("0.1.0") < plugin_pkg._parse_version("0.2.0b1")

    def test_unparseable_sorts_as_release(self):
        assert plugin_pkg._parse_version("garbage") == (0, 0, 0, "z", 0)


# ---------------------------------------------------------------------------
# _read_required_version
# ---------------------------------------------------------------------------


def _write_metadata(tmp_path, version: str) -> None:
    (tmp_path / "metadata.txt").write_text(f"[general]\nname=SOLWEIG\nversion={version}\n")


class TestReadRequiredVersion:
    """Tests for metadata.txt reading and QGIS-to-PEP-440 normalization."""

    @pytest.mark.parametrize(
        ("qgis_version", "expected"),
        [
            ("0.1.0-beta88", "0.1.0b88"),
            ("0.1.0beta88", "0.1.0b88"),
            ("0.2.0-alpha3", "0.2.0a3"),
            ("1.0.0-rc1", "1.0.0rc1"),
            ("1.2.3", "1.2.3"),
        ],
    )
    def test_normalization(self, tmp_path, monkeypatch, qgis_version, expected):
        _write_metadata(tmp_path, qgis_version)
        monkeypatch.setattr(plugin_pkg, "_PLUGIN_DIR", tmp_path)
        assert plugin_pkg._read_required_version() == expected

    def test_missing_metadata_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plugin_pkg, "_PLUGIN_DIR", tmp_path)
        assert plugin_pkg._read_required_version() == "0.0.0"

    def test_normalized_version_matches_pep440(self, tmp_path, monkeypatch):
        """QGIS '0.1.0-beta88' must compare equal to PEP 440 '0.1.0b88'."""
        from packaging.version import Version

        _write_metadata(tmp_path, "0.1.0-beta88")
        monkeypatch.setattr(plugin_pkg, "_PLUGIN_DIR", tmp_path)
        required = plugin_pkg._read_required_version()
        assert Version(required) == Version("0.1.0b88")
        assert plugin_pkg._parse_version(required) == plugin_pkg._parse_version("0.1.0b88")

    def test_real_metadata_is_pep440_parseable(self):
        """The shipped metadata.txt must normalize to a valid PEP 440 version."""
        from packaging.version import Version

        Version(plugin_pkg._read_required_version())  # must not raise


class TestShippedMetadataIsLoadable:
    """
    Guards on the shipped ``metadata.txt``.

    QGIS reads plugin metadata with ``configparser`` interpolation enabled,
    so a bare ``%`` anywhere in a value (a changelog percentage, say) makes
    plugin loading fail with "'%' must be followed by '%' or '('". Percent
    signs must be written as ``%%``. Only the value actually fetched gets
    interpolated, which is why ``_read_required_version`` (it reads just
    ``version``) never noticed.
    """

    @staticmethod
    def _shipped_config():
        import configparser

        config = configparser.ConfigParser()  # BasicInterpolation, matching QGIS
        config.read(plugin_pkg._PLUGIN_DIR / "metadata.txt")
        return config

    def test_every_field_interpolates(self):
        config = self._shipped_config()
        for section in config.sections():
            for option in config.options(section):
                config.get(section, option)  # must not raise InterpolationSyntaxError

    def test_version_matches_pyproject(self):
        """metadata.txt tracks pyproject.toml, the single source of truth."""
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        assert match is not None, "no version in pyproject.toml"
        assert plugin_pkg._read_required_version() == match.group(1)

    def test_changelog_has_an_entry_for_the_current_version(self):
        config = self._shipped_config()
        assert config.get("general", "version") in config.get("general", "changelog")


# ---------------------------------------------------------------------------
# _check_version
# ---------------------------------------------------------------------------


class _FakeSurfaceData:
    """Stub exposing all APIs _check_version probes for."""

    def preprocess(self): ...

    def fill_nan(self): ...

    def compute_valid_mask(self): ...

    def apply_valid_mask(self): ...

    def crop_to_valid_bbox(self): ...


class _IncompleteSurfaceData:
    """Stub missing crop_to_valid_bbox."""

    def preprocess(self): ...

    def fill_nan(self): ...

    def compute_valid_mask(self): ...

    def apply_valid_mask(self): ...


def _fake_solweig(version: str = "0.1.0b88", surface_cls: type | None = _FakeSurfaceData):
    module = SimpleNamespace(__version__=version)
    if surface_cls is not None:
        module.SurfaceData = surface_cls
    return module


@pytest.fixture()
def version_state(monkeypatch):
    """Reset the module-level version flags and pin the required version."""
    monkeypatch.setattr(plugin_pkg, "_SOLWEIG_OUTDATED", False)
    monkeypatch.setattr(plugin_pkg, "_SOLWEIG_IMPORT_ERROR", None)
    monkeypatch.setattr(plugin_pkg, "_SOLWEIG_INSTALLED_VERSION", None)
    monkeypatch.setattr(plugin_pkg, "_REQUIRED_SOLWEIG_VERSION", "0.1.0b88")
    return monkeypatch


class TestCheckVersion:
    """Tests for the installed-version and feature gate."""

    def test_equal_version_passes(self, version_state):
        assert plugin_pkg._check_version(_fake_solweig("0.1.0b88")) is True
        assert plugin_pkg._SOLWEIG_OUTDATED is False
        assert plugin_pkg._SOLWEIG_INSTALLED_VERSION == "0.1.0b88"

    def test_newer_beta_passes(self, version_state):
        assert plugin_pkg._check_version(_fake_solweig("0.1.0b89")) is True

    def test_final_release_passes_beta_requirement(self, version_state):
        assert plugin_pkg._check_version(_fake_solweig("0.1.0")) is True

    def test_older_beta_is_outdated(self, version_state):
        assert plugin_pkg._check_version(_fake_solweig("0.1.0b87")) is False
        assert plugin_pkg._SOLWEIG_OUTDATED is True
        assert "0.1.0b88" in str(plugin_pkg._SOLWEIG_IMPORT_ERROR)

    def test_missing_version_attribute_is_outdated(self, version_state):
        module = SimpleNamespace(SurfaceData=_FakeSurfaceData)
        assert plugin_pkg._check_version(module) is False
        assert plugin_pkg._SOLWEIG_INSTALLED_VERSION == "0.0.0"

    def test_outdated_via_fallback_parser(self, version_state):
        """When packaging is unavailable, _parse_version still rejects old versions."""
        version_state.setitem(sys.modules, "packaging.version", None)
        assert plugin_pkg._check_version(_fake_solweig("0.1.0b87")) is False
        assert plugin_pkg._SOLWEIG_OUTDATED is True

    def test_missing_surface_data_fails_feature_check(self, version_state):
        assert plugin_pkg._check_version(_fake_solweig(surface_cls=None)) is False
        assert plugin_pkg._SOLWEIG_OUTDATED is True
        assert "SurfaceData" in str(plugin_pkg._SOLWEIG_IMPORT_ERROR)

    def test_missing_api_fails_feature_check(self, version_state):
        assert plugin_pkg._check_version(_fake_solweig(surface_cls=_IncompleteSurfaceData)) is False
        assert "crop_to_valid_bbox" in str(plugin_pkg._SOLWEIG_IMPORT_ERROR)


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------

EXPECTED_ALGORITHM_IDS = {"epw_import", "surface_preprocessing", "solweig_calculation"}


class TestProvider:
    """Tests for SolweigProvider.loadAlgorithms()."""

    def _load(self, provider):
        added = []
        provider.addAlgorithm = added.append
        provider.loadAlgorithms()
        return added

    def test_provider_id_and_name(self):
        provider = provider_module.SolweigProvider()
        assert provider.id() == "solweig"
        assert provider.name() == "SOLWEIG"

    def test_load_algorithms_registers_expected_ids(self):
        provider = provider_module.SolweigProvider()
        added = self._load(provider)
        assert {alg.name() for alg in added} == EXPECTED_ALGORITHM_IDS

    def test_import_error_in_one_algorithm_is_logged_and_skipped(self, monkeypatch):
        """A broken algorithm module must not prevent the others from registering."""
        provider = provider_module.SolweigProvider()
        log = MagicMock()
        monkeypatch.setattr(provider_module, "QgsMessageLog", log)
        # None in sys.modules makes any import of the module raise ImportError.
        broken = "qgis_plugin.solweig_qgis.algorithms.utilities.epw_import"
        monkeypatch.setitem(sys.modules, broken, None)

        added = self._load(provider)

        assert {alg.name() for alg in added} == {"surface_preprocessing", "solweig_calculation"}
        log.logMessage.assert_called_once()
        assert "EpwImportAlgorithm" in log.logMessage.call_args[0][0]


# ---------------------------------------------------------------------------
# Graceful degradation: algorithm modules import without solweig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "qgis_plugin.solweig_qgis.algorithms.utilities.epw_import",
        "qgis_plugin.solweig_qgis.algorithms.preprocess.surface_preprocessing",
        "qgis_plugin.solweig_qgis.algorithms.calculation.solweig_calculation",
    ],
)
def test_algorithm_module_imports_without_solweig(module_name):
    """Registration-time imports must not require solweig (deferred to processAlgorithm).

    Guards the graceful-degradation design: when solweig is not installed,
    loadAlgorithms() must still register every algorithm so the install
    prompt can appear.
    """
    saved_module = sys.modules.get(module_name)
    try:
        with preserve_solweig_modules():
            for key in [k for k in sys.modules if k == "solweig" or k.startswith("solweig.")]:
                del sys.modules[key]
            # None in sys.modules makes any solweig import raise ImportError.
            sys.modules["solweig"] = cast(ModuleType, None)
            sys.modules.pop(module_name, None)
            importlib.import_module(module_name)  # must not raise
    finally:
        if saved_module is not None:
            sys.modules[module_name] = saved_module
