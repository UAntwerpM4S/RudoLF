"""
AirSim integration for RudoLF: connects the framework to CosysAirSim / Unreal Engine 5.

Subpackages:
    config: Validated connection configuration (IP, port, timeout, retries).
    communication: Bridge adapter (with UE5 detection), control mapping, coordinate transform.
    environments: Gymnasium RL environment using UE5 physics.
    train: Entry points for RL training inside UE5.
    inference: Live and headless telemetry visualization.

Example::

    >>> from airsim.config.airsim_config import AirSimConfig
    >>> from airsim.communication.airsim_bridge import AirSimBridge, is_ue5_reachable
    >>> from airsim.communication.coord_transform import CoordTransform
"""

import logging

from airsim.communication.coord_transform import CoordTransform
from airsim.config.airsim_config import AirSimConfig, get_airsim_config
from airsim.communication.airsim_bridge import is_ue5_reachable

# Library convention: the application (e.g. main.py) configures the real handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AirSimConfig",
    "CoordTransform",
    "get_airsim_config",
    "is_ue5_reachable",
]
