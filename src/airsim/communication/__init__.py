"""Communication layer between Fossen simulation and CosysAirSim.

Modules:
    airsim_bridge: PID heading-tracking adapter that translates Fossen
        ship state into AirSim ``CarControls``.
    coord_transform: Bidirectional coordinate mapping between Fossen
        (east, north) and AirSim (NED) frames.
"""
