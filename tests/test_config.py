"""Smoke tests for the config loader (src/utils/config.py)."""

from pathlib import Path

import pytest

from src.utils.config import _deep_merge, load_config

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"
_INTERNAL_CONFIG = _CONFIGS_DIR / "gpu_server_internal.yaml"
_EXTERNAL_CONFIG = _CONFIGS_DIR / "gpu_server_external.yaml"


class TestDeepMerge:
    def test_override_wins_on_conflict(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_dicts_merge_key_by_key(self):
        """A nested section in override must not wipe out sibling base keys."""
        base = {"model": {"freeze_esm2": True, "esm2_variant": "8M"}}
        override = {"model": {"injection_mode": "internal"}}
        merged = _deep_merge(base, override)
        assert merged["model"] == {
            "freeze_esm2": True,
            "esm2_variant": "8M",
            "injection_mode": "internal",
        }

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}


class TestLoadConfig:
    def test_internal_config_injection_mode(self):
        config = load_config(_INTERNAL_CONFIG)
        assert config["model"]["injection_mode"] == "internal"

    def test_external_config_injection_mode(self):
        config = load_config(_EXTERNAL_CONFIG)
        assert config["model"]["injection_mode"] == "external"

    def test_base_yaml_values_inherited(self):
        """model.freeze_esm2 lives only in base.yaml, not in the override file."""
        config = load_config(_INTERNAL_CONFIG)
        assert config["model"]["freeze_esm2"] is True

    def test_classifier_and_loss_sections_inherited_from_base(self):
        config = load_config(_INTERNAL_CONFIG)
        assert config["classifier"] == {"hidden_dim": 512, "dropout": 0.1}
        assert config["loss"] == {"weight_amr_gene_family": 1.0}

    def test_hyperparameters_are_v1_defaults(self):
        for config_path in (_INTERNAL_CONFIG, _EXTERNAL_CONFIG):
            config = load_config(config_path)
            assert config["training"]["batch_size"] == 32
            assert config["training"]["learning_rate"] == pytest.approx(1e-4)
            assert config["training"]["epochs"] == 50
            assert config["training"]["optimizer"] == "adam"

    def test_learning_rate_is_float_not_string(self):
        """Regression guard: YAML '1e-4' (no decimal point) parses as a string in
        PyYAML, not a float -- configs must write '1.0e-4' or '0.0001' instead."""
        config = load_config(_INTERNAL_CONFIG)
        assert isinstance(config["training"]["learning_rate"], float)

    def test_internal_and_external_differ_only_in_injection_mode_and_output_dir(self):
        """injection_mode and paths.output_dir are the only intentional differences.

        output_dir must differ so the two ablations never overwrite each
        other's checkpoint; everything else must stay in lockstep so this
        remains a controlled A/B comparison.
        """
        internal = load_config(_INTERNAL_CONFIG)
        external = load_config(_EXTERNAL_CONFIG)
        assert internal["model"]["injection_mode"] != external["model"]["injection_mode"]

        internal_without_mode = {**internal["model"]}
        external_without_mode = {**external["model"]}
        del internal_without_mode["injection_mode"]
        del external_without_mode["injection_mode"]
        assert internal_without_mode == external_without_mode

        assert internal["training"] == external["training"]

        assert internal["paths"]["output_dir"] != external["paths"]["output_dir"]
        internal_paths_without_output_dir = {**internal["paths"]}
        external_paths_without_output_dir = {**external["paths"]}
        del internal_paths_without_output_dir["output_dir"]
        del external_paths_without_output_dir["output_dir"]
        assert internal_paths_without_output_dir == external_paths_without_output_dir

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")
