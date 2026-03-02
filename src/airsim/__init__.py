"""AirSim integration package for the RudoLF RL framework.

Provides all components needed to connect the RudoLF reinforcement-learning
framework to CosysAirSim / Unreal Engine 5:

Subpackages:
    config: AirSim-specific PID gains, network settings, and calibration data.
    communication: Bridge adapter and coordinate transform (Fossen ↔ AirSim).
    environments: Gymnasium RL environment using UE5 physics.
    train: Entry points for RL training inside UE5.
    inference: Live and headless telemetry visualisation.
    tools: Coordinate calibration wizard and telemetry analysis.
    poc: Frozen snapshots of the original proof-of-concept scripts.
    tests: Pytest unit tests (no UE5 required).

Example:
    >>> from airsim.communication.coord_transform import CoordTransform
    >>> from airsim.config.airsim_config import get_airsim_config
"""

from airsim.communication.coord_transform import CoordTransform
from airsim.config.airsim_config import get_airsim_config

__all__ = ["CoordTransform", "get_airsim_config"]
