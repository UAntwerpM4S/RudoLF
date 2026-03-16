"""Small DTO used by fw common tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvironmentRequest:
    """Factory request payload for environment creation.

    Attributes:
        name: Registered environment name.
        configuration: Optional configuration object forwarded to the factory.
    """

    name: str
    configuration: Any = None
