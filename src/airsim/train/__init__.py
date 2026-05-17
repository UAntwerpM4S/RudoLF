"""
Training entry points for AirSim RL environments.

Modules:
    train_airsim: Launches PPO training inside UE5 via CosysAirSim.
"""

from airsim.train.strict_success_callback import StrictSuccessStoppingCallback

__all__ = ["StrictSuccessStoppingCallback"]
