"""Round-trip tests for `solweig.metadata` run-metadata helpers."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from solweig.metadata import create_run_metadata, load_run_metadata, save_run_metadata
from solweig.models import HumanParams


def _stub_surface(rows=10, cols=12, pixel_size=1.0, crs="EPSG:3006"):
    surface = MagicMock()
    surface.shape = (rows, cols)
    surface.pixel_size = pixel_size
    surface.crs = crs
    return surface


def _stub_location(lat=57.7, lon=12.0, utc_offset=1):
    return SimpleNamespace(latitude=lat, longitude=lon, utc_offset=utc_offset)


def _stub_weather(year=2024, month=7, day=15, hour=12):
    return SimpleNamespace(datetime=datetime(year, month, day, hour))


def test_create_metadata_minimal_required_fields():
    md = create_run_metadata(
        surface=_stub_surface(),
        location=_stub_location(),
        weather_series=[_stub_weather(), _stub_weather(hour=13), _stub_weather(hour=14)],
        human=None,
        physics=None,
        materials=None,
        use_anisotropic_sky=True,
        conifer=False,
        output_dir="/tmp/foo",
        outputs=["tmrt"],
    )

    assert md["solweig_version"]  # not empty (could be "unknown" if not installed)
    assert md["grid"] == {"rows": 10, "cols": 12, "pixel_size": 1.0, "crs": "EPSG:3006"}
    assert md["location"]["latitude"] == 57.7
    assert md["timeseries"]["timesteps"] == 3
    assert md["timeseries"]["start"] == "2024-07-15T12:00:00"
    assert md["timeseries"]["end"] == "2024-07-15T14:00:00"
    assert md["parameters"]["use_anisotropic_sky"] is True
    assert md["outputs"]["variables"] == ["tmrt"]
    # Optional sections must NOT be present when their inputs are None.
    assert "human" not in md
    assert "physics" not in md
    assert "materials" not in md


def test_create_metadata_with_human_and_namespace_params():
    physics = SimpleNamespace(svf=SimpleNamespace(option=2), wall=SimpleNamespace(emis=0.9))
    materials = SimpleNamespace(Ts_deg=SimpleNamespace(Value=SimpleNamespace(Walls=0.37)))
    human = HumanParams(weight=75.0, height=1.75)

    md = create_run_metadata(
        surface=_stub_surface(),
        location=_stub_location(),
        weather_series=[_stub_weather()],
        human=human,
        physics=physics,
        materials=materials,
        use_anisotropic_sky=False,
        conifer=True,
        output_dir="/tmp/bar",
        outputs=None,
    )

    assert md["human"]["posture"] == human.posture
    assert md["human"]["abs_k"] == human.abs_k
    assert md["physics"]["full_params"]["svf"]["option"] == 2
    assert md["materials"]["full_params"]["Ts_deg"]["Value"]["Walls"] == 0.37
    assert md["outputs"]["variables"] == []  # None → []


def test_save_and_load_roundtrip(tmp_path):
    md = create_run_metadata(
        surface=_stub_surface(),
        location=_stub_location(),
        weather_series=[_stub_weather()],
        human=None,
        physics=None,
        materials=None,
        use_anisotropic_sky=True,
        conifer=False,
        output_dir=str(tmp_path),
        outputs=["tmrt", "shadow"],
    )
    path = save_run_metadata(md, tmp_path)
    assert path.exists()
    assert path.name == "run_metadata.json"

    # File is valid JSON
    with open(path) as f:
        raw = json.load(f)
    assert raw["grid"]["rows"] == 10

    # load_run_metadata returns the same dict
    loaded = load_run_metadata(path)
    assert loaded == md


def test_save_run_metadata_respects_custom_filename(tmp_path):
    md = {"hello": "world"}
    path = save_run_metadata(md, tmp_path, filename="custom.json")
    assert path.name == "custom.json"
    assert load_run_metadata(path) == md
