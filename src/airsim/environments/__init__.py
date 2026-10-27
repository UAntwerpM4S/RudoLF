"""
Gymnasium RL environments for AirSim / UE5.

Modules:
    airsim_env: `AirSimEnv`, drop-in replacement for `PySimEnv`
        that uses Unreal Engine 5 physics via CosysAirSim.
"""

from .airsim_env import AirSimEnv

__all__ = ["AirSimEnv"]
