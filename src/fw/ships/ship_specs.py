from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ShipSpecifications:
    """
    Container for ship physical properties, actuator limits, and kinematic constraints.

    This mutable dataclass defines the vessel parameters required for
    simulation, control, and actuator modeling. It provides:

        - Physical properties (length, mass, optional yaw inertia I_z)
        - Actuator limits (rudder and thrust)
        - Jitter suppression thresholds
        - Kinematic limits for simulation constraints
        - Fossen 3-DOF hydrodynamic, rudder and thrust coefficients

    All units are expressed in SI unless otherwise specified.
    """

    # Physical properties
    length: float
    """
    Vessel length [m]. Used for calculating inertia and added-mass terms.
    """

    mass: float
    """
    Vessel mass [kg]. Used in dynamics calculations and damping scaling.
    """

    # Actuator limits
    max_rudder_angle: float
    """
    Maximum rudder angle [rad] (magnitude; the command is symmetric, positive = turn to port).
    """

    max_rudder_rate: float
    """
    Maximum rudder rate [rad/s]. Limits the speed at which rudder can move.
    """

    min_thrust: float
    """
    Minimum normalized thrust [-]. Used to define the forward thrust envelope.
    """

    max_thrust: float
    """
    Maximum normalized thrust [-]. Defines the upper bound of propulsion.
    """

    max_thrust_rate: float
    """
    Maximum thrust rate change [-/s]. Limits the speed of thrust adjustment.
    """

    # Jitter suppression thresholds
    rudder_jitter_threshold: float
    """
    Minimum rudder deviation [rad] to consider for actuation updates.
    Smaller deviations are ignored to prevent high-frequency oscillations.
    """

    thrust_jitter_threshold: float
    """
    Minimum thrust deviation [-] to consider for actuation updates.
    Prevents small oscillations around the current thrust value.
    """

    # Kinematic limits (simulation constraints)
    surge_limits: Tuple[float, float]
    """
    Allowed surge velocity range [m/s] for the vessel in simulation.
    """

    sway_limits: Tuple[float, float]
    """
    Allowed sway velocity range [m/s] for the vessel in simulation.
    """

    yaw_rate_limits: Tuple[float, float]
    """
    Allowed yaw rate range [rad/s] for the vessel in simulation.
    """

    # Fossen 3-DOF hydrodynamic coefficients. Each one is the dimensional
    # derivative divided by a mass/length scale; fw.dynamics.fossen multiplies
    # it back by the SAME scale, using `mass` and `length` above. A coefficient
    # derived with another length than `length` is therefore wrong.
    #   scale = mass        : X_udot, Y_vdot, X_u, Y_v, X_uu, Y_vv, Y_uv
    #   scale = mass*length : N_uv
    #   scale = mass*length^2: N_rdot, N_r, N_rr

    X_udot_coeff: float
    """Surge added mass X_udot / m [-]."""

    Y_vdot_coeff: float
    """Sway added mass Y_vdot / m [-]."""

    N_rdot_coeff: float
    """Yaw added inertia N_rdot / (m L^2) [-]."""

    X_u_coeff: float
    """Linear surge damping X_u / m [1/s]."""

    Y_v_coeff: float
    """Linear sway damping Y_v / m [1/s]."""

    N_r_coeff: float
    """Linear yaw damping N_r / (m L^2) [1/s]."""

    X_uu_coeff: float
    """Quadratic surge damping X_u|u| / m [1/m]."""

    Y_vv_coeff: float
    """Quadratic sway damping Y_v|v| / m [1/m]."""

    N_rr_coeff: float
    """Quadratic yaw damping N_r|r| / (m L^2) [-]."""

    Y_uv_coeff: float
    """Cross-flow sway term Y_uv / m [1/m]."""

    N_uv_coeff: float
    """Cross-flow yaw term N_uv / (m L) [1/m]."""

    Y_rudder_coeff: float = 0.1
    """
    Sway-force-from-rudder coefficient [-]. Scaled by mass: Y_rudder = Y_rudder_coeff * mass.
    """

    N_rudder_coeff: float = 0.001
    """
    Yaw-moment-from-rudder coefficient [-]. Scaled by mass*length: N_rudder = N_rudder_coeff * mass * length.
    """

    X_thrust_coeff: float = 0.05
    """
    Surge-force-from-thrust coefficient [-]. Scaled by mass: X_thrust = X_thrust_coeff * mass.
    """

    I_z: Optional[float] = None
    """
    Rigid-body yaw moment of inertia [kg m^2] (RIZZ in the FH ship data).
    None = fall back to the slender-rod approximation mass * length^2 / 12,
    which overestimates I_z by roughly 29-38 % for the Myzako and Tripoli hulls.
    The FH-identified presets deliberately keep None: FH fitted their
    coefficients with the slender-rod inertia.
    """

    def __post_init__(self):
        """Validate I_z: positive or None (raises ValueError otherwise)."""

        # "not > 0" also rejects NaN
        if self.I_z is not None and not self.I_z > 0.0:
            raise ValueError(f"I_z must be positive or None, got {self.I_z}")
