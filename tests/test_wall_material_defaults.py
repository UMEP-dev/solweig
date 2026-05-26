"""Tests for :class:`solweig.models.materials.WallMaterialDefaults`."""

from __future__ import annotations

from types import SimpleNamespace

from solweig.models.materials import WallMaterialDefaults


def test_from_namespace_none_returns_all_none_defaults():
    d = WallMaterialDefaults.from_namespace(None)
    assert d == WallMaterialDefaults(tgk=None, tstart=None, tmaxlst=None)


def test_from_namespace_empty_namespace_returns_all_none():
    d = WallMaterialDefaults.from_namespace(SimpleNamespace())
    assert d == WallMaterialDefaults(tgk=None, tstart=None, tmaxlst=None)


def test_from_namespace_partial_namespace_only_reads_what_exists():
    ns = SimpleNamespace(
        Ts_deg=SimpleNamespace(Value=SimpleNamespace(Walls=0.42)),
        # Tstart and TmaxLST absent entirely
    )
    d = WallMaterialDefaults.from_namespace(ns)
    assert d.tgk == 0.42
    assert d.tstart is None
    assert d.tmaxlst is None


def test_from_namespace_all_three_set():
    ns = SimpleNamespace(
        Ts_deg=SimpleNamespace(Value=SimpleNamespace(Walls=0.5)),
        Tstart=SimpleNamespace(Value=SimpleNamespace(Walls=-2.0)),
        TmaxLST=SimpleNamespace(Value=SimpleNamespace(Walls=18.0)),
    )
    d = WallMaterialDefaults.from_namespace(ns)
    assert (d.tgk, d.tstart, d.tmaxlst) == (0.5, -2.0, 18.0)


def test_apply_uses_defaults_when_overrides_none():
    d = WallMaterialDefaults()  # all None
    assert d.apply(0.37, -3.41, 15.0) == (0.37, -3.41, 15.0)


def test_apply_each_field_independently():
    d = WallMaterialDefaults(tgk=0.5, tstart=None, tmaxlst=18.0)
    assert d.apply(0.37, -3.41, 15.0) == (0.5, -3.41, 18.0)


def test_intermediate_missing_node_returns_none():
    """When `Ts_deg` exists but lacks `.Value`, the read must safely return None."""
    ns = SimpleNamespace(Ts_deg=SimpleNamespace())  # no .Value
    d = WallMaterialDefaults.from_namespace(ns)
    assert d.tgk is None
