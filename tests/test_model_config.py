"""Tests for ModelConfig.from_json parameter extraction.

Regression gate for the legacy-JSON nesting bug: UMEP parameter files nest
each section's values under a ``Value`` key. ``from_json`` previously read
``params.Tmrt_params.absK`` (which does not exist) and silently fell back to
the hardcoded defaults, so user-supplied values were never loaded.
"""

from __future__ import annotations

import json

import pytest
from solweig.models.config import ModelConfig


def _write_params(path, absk=0.7, absl=0.97, posture="Standing", height=180, sex="Male"):
    data = {
        "Tmrt_params": {
            "Value": {"absK": absk, "absL": absl, "posture": posture},
            "Comment": "test",
        },
        "PET_settings": {
            "Value": {
                "Age": 35,
                "Weight": 75.0,
                "Height": height,
                "Sex": sex,
                "Activity": 80.0,
                "clo": 0.9,
            },
            "Comment": "test",
        },
    }
    path.write_text(json.dumps(data))
    return path


def test_from_json_reads_nested_values(tmp_path):
    """Non-default values in the file must actually land in HumanParams."""
    path = _write_params(tmp_path / "params.json", absk=0.62, absl=0.95, posture="Sitting", height=165, sex="Female")
    config = ModelConfig.from_json(path)

    assert config.human is not None
    assert config.human.abs_k == pytest.approx(0.62)
    assert config.human.abs_l == pytest.approx(0.95)
    assert config.human.posture == "sitting"
    # Legacy files store height in centimetres
    assert config.human.height == pytest.approx(1.65)
    assert config.human.sex == 2


def test_from_json_bundled_defaults():
    """The bundled parameter file loads and matches the dataclass defaults."""
    # load_params(None) resolves to the bundled default JSON, so from_json(None)
    # reads pysrc/solweig/data/default_params.json.
    config = ModelConfig.from_json(None)
    assert config.human is not None
    assert config.human.abs_k == pytest.approx(0.7)
    assert config.human.abs_l == pytest.approx(0.97)
    assert config.human.posture == "standing"
    assert config.human.height == pytest.approx(1.80)
    assert config.human.sex == 1


def test_from_json_validates_values(tmp_path):
    """File values pass through HumanParams validation instead of bypassing it."""
    path = _write_params(tmp_path / "params.json", absk=1.5)
    with pytest.raises(ValueError, match="abs_k"):
        ModelConfig.from_json(path)
