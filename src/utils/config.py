"""Load training configs: an environment file merged over configs/base.yaml."""

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base; override wins on any key conflict.

    Nested dicts are merged key-by-key rather than replaced wholesale, so an
    environment override only needs to specify the keys it actually changes --
    e.g. an override's `model:` section can set just `injection_mode` and still
    inherit `freeze_esm2` from base rather than losing it.

    Args:
        base: The base config dict (e.g. loaded from base.yaml).
        override: The environment-specific config dict, takes precedence.

    Returns:
        A new merged dict; neither input is mutated.
    """
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load an environment config, merged over configs/base.yaml's shared defaults.

    base.yaml is resolved relative to config_path's own directory, so this
    works regardless of the current working directory -- config_path must live
    in the same configs/ directory as base.yaml.

    Args:
        config_path: Path to an environment-specific config file, e.g.
            configs/gpu_server_internal.yaml.

    Returns:
        Merged config dict: base.yaml's values with config_path's values
        overriding on any conflicting key (nested sections merged recursively
        via _deep_merge, not replaced wholesale).

    Raises:
        FileNotFoundError: If config_path or the sibling base.yaml is missing.
    """
    config_path = Path(config_path)
    base_path = config_path.parent / "base.yaml"

    with open(base_path, encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh) or {}

    with open(config_path, encoding="utf-8") as fh:
        override_config = yaml.safe_load(fh) or {}

    return _deep_merge(base_config, override_config)
