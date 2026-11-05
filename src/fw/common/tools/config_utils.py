"""
Configuration helper utilities shared by framework modules.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping


def normalize_configuration(configuration: Any) -> Any:
    """
    Normalize placeholder configuration values to stable runtime objects.

    Args:
        configuration: Configuration object passed by callers. Supported values:
            - `None`: returned unchanged.
            - `Mapping`: copied into a plain dict.
            - empty `list`/`tuple`: converted to `{}`.
            - anything else: returned unchanged.

    Returns:
        A normalized configuration object suitable for environment factories.
    """

    if configuration is None:
        return None
    if isinstance(configuration, Mapping):
        return dict(configuration)
    if isinstance(configuration, (list, tuple)) and len(configuration) == 0:
        return {}
    return configuration


def get_validated_bool(mapping: Mapping, key: str, default: bool) -> bool:
    """
    Fetch a boolean flag from a JSON-sourced config mapping, failing fast on
    non-bool values.

    Guards against `bool("false") == True`: a quoted "false" in a JSON config
    would otherwise be silently truthy.

    Args:
        mapping: Configuration dict (or other Mapping) to read from.
        key: Key to look up.
        default: Value to return if `key` is absent. Not type-checked, since
            defaults are supplied by code, not by config files.

    Returns:
        The boolean value stored at `key`, or `default` if absent.

    Raises:
        ValueError: If `key` is present but its value is not a `bool`.
    """

    if key not in mapping:
        return default

    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"Configuration key '{key}' must be a boolean, got {value!r} "
            f"({type(value).__name__}). Check for a quoted \"true\"/\"false\" "
            f"string in the JSON config where a literal true/false was intended."
        )
    return value
