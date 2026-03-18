"""Communication layer between the RL framework and CosysAirSim.

Modules:
    airsim_bridge: Robust connection manager and pass-through control
        adapter with UE5 auto-detection, retry, and health-checks.
    coord_transform: Bidirectional coordinate mapping between Fossen
        (east, north) and AirSim (NED) frames.
"""
