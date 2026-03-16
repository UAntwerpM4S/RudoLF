"""Configuration helper utilities shared by framework modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_configuration(configuration: Any) -> Any:
    """Normalize placeholder configuration values to stable runtime objects.

    Args:
        configuration: Configuration object passed by callers. Supported values:
            - ``None``: returned unchanged.
            - ``Mapping``: copied into a plain dict.
            - empty ``list``/``tuple``: converted to ``{}``.
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
