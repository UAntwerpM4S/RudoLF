"""AirSim-specific configuration parameters.

Centralises all CosysAirSim PID gains, network settings, and
telemetry options that were previously scattered in ``fw/config.py``.

Example:
    >>> from airsim.config.airsim_config import get_airsim_config
    >>> cfg = get_airsim_config()
    >>> cfg["airsim_kp"]
    5.0
"""

AIRSIM_CONFIG: dict[str, object] = {
    "use_airsim": True,
    "airsim_ip": "",
    "airsim_port": 41451,

    # Heading PID controller
    "airsim_kp": 5.0,
    "airsim_ki": 0.5,
    "airsim_kd": 1.0,
    "airsim_ff": 0.5,

    # Cross-track error correction
    "airsim_lat_gain": 0.5,

    # Speed matching
    "airsim_speed_scale": 1.05,

    # Throttle PI controller
    "airsim_throttle_kp": 1.0,
    "airsim_throttle_ki": 0.2,

    # Distance error correction
    "airsim_dist_gain": 0.1,
    "airsim_dist_ki": 0.01,

    # Telemetry
    "airsim_log_telemetry": True,
}
"""Module-level configuration dictionary.

Keys:
    use_airsim (bool): Enable or disable the AirSim bridge.
    airsim_ip (str): AirSim server IP address (empty = localhost).
    airsim_port (int): AirSim RPC port.
    airsim_kp (float): Proportional gain for heading PID.
    airsim_ki (float): Integral gain for heading PID.
    airsim_kd (float): Derivative gain for heading PID.
    airsim_ff (float): Feedforward gain for yaw-rate anticipation.
    airsim_lat_gain (float): Cross-track error correction gain.
    airsim_speed_scale (float): Multiplier from Fossen to AirSim speeds.
    airsim_throttle_kp (float): Proportional gain for throttle PI.
    airsim_throttle_ki (float): Integral gain for throttle PI.
    airsim_dist_gain (float): Distance-error proportional gain.
    airsim_dist_ki (float): Distance-error integral gain.
    airsim_log_telemetry (bool): Write per-step data to CSV.
"""


def get_airsim_config() -> dict:
    """Return a shallow copy of the AirSim configuration dictionary.

    The copy prevents accidental mutation of the module-level defaults.

    Returns:
        A dictionary with all AirSim-specific parameters.

    Example:
        >>> cfg = get_airsim_config()
        >>> cfg["airsim_port"]
        41451
    """
    return AIRSIM_CONFIG.copy()
