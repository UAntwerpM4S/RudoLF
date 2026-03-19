"""AirSim integration package for the RudoLF RL framework.

Provides all components needed to connect the RudoLF reinforcement-learning
framework to CosysAirSim / Unreal Engine 5:

Subpackages:
    common: Reusable shared classes and utility helpers.
    config: Validated connection configuration (IP, port, timeout, retries).
    communication: Bridge adapter (with UE5 detection) and coordinate transform.
    environments: Gymnasium RL environment using UE5 physics.
    train: Entry points for RL training inside UE5.
    inference: Live and headless telemetry visualisation.
    tools: Backward-compatible wrappers for utility scripts.
    poc: Frozen snapshots of the original proof-of-concept scripts.
    tests: Pytest unit tests (no UE5 required).

Example::

    >>> from airsim.config.airsim_config import AirSimConfig
    >>> from airsim.communication.airsim_bridge import AirSimBridge, is_ue5_reachable
    >>> from airsim.communication.coord_transform import CoordTransform
"""

import logging

from airsim.communication.coord_transform import CoordTransform
from airsim.config.airsim_config import AirSimConfig, get_airsim_config
from airsim.communication.airsim_bridge import is_ue5_reachable

# Library best practice: prevent "No handler found" warnings.
# The consuming application (e.g. main.py) configures the actual handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AirSimConfig",
    "CoordTransform",
    "get_airsim_config",
    "is_ue5_reachable",
]
