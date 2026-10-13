"""
UGent 3-DOF confined-water manoeuvring model of the Tripoli (E01), full scale.

Source: Van Damme & Delefortrie (2026), DDSHIP D2.3, FH Reports 23_002_1
(equations §2.2, eq. 11-19; coefficients Table 4, ITTC X_u|u| Table 5).

Notes:
    - Table 4 errata (confirmed by A. Van Damme): Y_u|u|/y'B / λ,
      N_r|r| × λ, N_ur / λ (wrong Froude-scaling exponents).
    - UKC is fixed to one towing-tank condition: 363, 125 or 43 %.
    - Bank coefficients at 363 % are extrapolated linearly in T/(h-T)
      from the 125 % and 43 % values (as in UGent's coef_extrapolation.py).
    - Frames: equations are in the report frame (y starboard); the framework
      frame has y to port, so v and r change sign at the boundary. Rudder
      angle = Z-drive azimuth, positive to starboard: the ship turns to port,
      same as Fossen.
    - Validity: straight running and small manoeuvres (zigzag 10-25°), not
      steady turning. Bank effects validated for y' = 1-4 (Fig. 16-17).
"""

import logging
import math
import numbers
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.common.tools.config_utils import get_validated_bool

logger = logging.getLogger(__name__)

# state -> (dist_port, dist_starboard) [m]; None = no bank on that side.
BankDistanceProvider = Callable[[VesselState], Tuple[Optional[float], Optional[float]]]

LAMBDA = 25.0  # Froude scale factor model -> full scale

_REQUIRED_CONFIG_KEYS = frozenset({"ukc_percent"})
_OPTIONAL_CONFIG_KEYS = frozenset({"ittc_correction", "n_max_rpm", "min_bank_clearance",
                                   "max_bank_clearance"})


@dataclass(frozen=True)
class TripoliParticulars:
    """Full-scale main particulars and propulsion data of the Tripoli."""

    mass: float = 6161870.27          # kg
    length_pp: float = 110.0          # m
    beam: float = 17.1                # m
    i_zz: float = 4651416900.21       # kg m²
    x_g: float = 1.1874               # m, CoG forward of geometrical centre
    d_prop: float = 1.6573            # m
    draft: float = 0.151 * LAMBDA     # m (report Table 1)
    n_propellers: int = 2             # twin Z-drive
    wake_fraction: float = 0.3        # eq. 14
    kt_0: float = 0.6723              # K_T = kt_0 - kt_slope * J (eq. 13)
    kt_slope: float = 0.7968
    rho_water: float = 1025.0         # kg/m³


@dataclass(frozen=True)
class ManoeuvringCoefficients:
    """Coefficient set for one UKC condition."""

    ukc_percent: float
    X_udot: float
    X_u: float
    X_uabsu: float
    X_uabsv: float
    Y_vdot: float
    Y_vabsv: float
    Y_uv: float
    Y_uabsu_yB: float    # bank term, sway
    N_rdot: float
    N_rabsr: float
    N_ur: float
    N_uabsu_yB: float    # bank term, yaw

    @property
    def depth_ratio(self) -> float:
        """T/(h-T) = 100 / UKC[%]."""
        return 100.0 / self.ukc_percent


def _extrapolate_linear(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    """Straight line through (x0, y0) and (x1, y1), evaluated at x."""
    return y0 + (y1 - y0) / (x1 - x0) * (x - x0)


def tripoli_coefficient_sets(ittc_correction: bool) -> Dict[int, ManoeuvringCoefficients]:
    """Full-scale coefficient sets keyed by UKC [%]: Table 4, with the errata applied."""
    # X_u|u|: Table 4, or Table 5 (ITTC-corrected).
    x_uabsu = {363: 2.55e3, 125: 3.85e3, 43: 5.08e3} if ittc_correction \
        else {363: 5.92e3, 125: 7.21e3, 43: 8.45e3}

    ukc125 = ManoeuvringCoefficients(
        ukc_percent=125.0,
        X_udot=-8.53e5, X_u=1.65e-11, X_uabsu=x_uabsu[125], X_uabsv=-6.40e4,
        Y_vdot=-1.52e6, Y_vabsv=-3.50e5, Y_uv=-1.14e5,
        Y_uabsu_yB=-7.11e4 / LAMBDA,           # erratum: λ³ -> λ²
        N_rdot=-1.38e8,
        N_rabsr=-2.18e7 * LAMBDA,              # erratum: λ⁴ -> λ⁵
        N_ur=-3.70e8 / LAMBDA,                 # erratum: λ⁵ -> λ⁴
        N_uabsu_yB=9.10e4,
    )
    ukc43 = ManoeuvringCoefficients(
        ukc_percent=43.0,
        X_udot=-8.53e5, X_u=1.30e-6, X_uabsu=x_uabsu[43], X_uabsv=-9.58e4,
        Y_vdot=-6.16e5, Y_vabsv=-9.02e5, Y_uv=-2.05e5,
        Y_uabsu_yB=-7.05e4 / LAMBDA,           # erratum: λ³ -> λ²
        N_rdot=-2.57e-9,
        N_rabsr=-1.19e8 * LAMBDA,              # erratum: λ⁴ -> λ⁵
        N_ur=-1.97e8 / LAMBDA,                 # erratum: λ⁵ -> λ⁴
        N_uabsu_yB=1.70e5,
    )

    # Deep water: bank terms not measured, extrapolated from 125 % and 43 %.
    deep_ratio = 100.0 / 363.0
    ukc363 = ManoeuvringCoefficients(
        ukc_percent=363.0,
        X_udot=-8.53e5, X_u=2.12e3, X_uabsu=x_uabsu[363], X_uabsv=-3.61e4,
        Y_vdot=-3.03e6, Y_vabsv=-1.27e5, Y_uv=-8.79e4,
        Y_uabsu_yB=_extrapolate_linear(ukc125.depth_ratio, ukc125.Y_uabsu_yB,
                                       ukc43.depth_ratio, ukc43.Y_uabsu_yB, deep_ratio),
        N_rdot=-1.29e9,
        N_rabsr=-1.28e8 * LAMBDA,              # erratum: λ⁴ -> λ⁵
        N_ur=-5.07e-11 / LAMBDA,               # erratum: λ⁵ -> λ⁴
        N_uabsu_yB=_extrapolate_linear(ukc125.depth_ratio, ukc125.N_uabsu_yB,
                                       ukc43.depth_ratio, ukc43.N_uabsu_yB, deep_ratio),
    )
    return {363: ukc363, 125: ukc125, 43: ukc43}


def _positive_finite(config: Mapping, key: str, default: float) -> float:
    """config[key] as a finite float > 0; raises ValueError otherwise."""
    value = config.get(key, default)
    if isinstance(value, (bool, str)) or not isinstance(value, numbers.Real) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"ugent_surrogate_config[{key!r}] must be a finite number > 0, got {value!r}")
    return float(value)


SUPPORTED_UKC_PERCENT = tuple(sorted(tripoli_coefficient_sets(False), reverse=True))  # (363, 125, 43)


class UGentSurrogate(DynamicsBase):
    """
    UGent manoeuvring model of the Tripoli at one fixed UKC.

    Bank distances come from a provider (set_bank_distance_provider), queried
    with the current state at every evaluation, or else from the last
    set_bank_distances() call. Without either, no banks are present.
    """

    def __init__(self, ship_spec: ShipSpecifications, ugent_surrogate_config: Optional[Mapping] = None):
        """
        Args:
            ship_spec: Kept for the DynamicsBase interface; the model uses
                its own Tripoli particulars.
            ugent_surrogate_config:
                ukc_percent (required): 363, 125 or 43.
                ittc_correction (default False): use Table 5 X_u|u|.
                n_max_rpm (default 320): propeller rate at thrust 1.0.
                min_bank_clearance (default 1.0): closer banks act as at this y'.
                max_bank_clearance (default 4.0): farther banks are ignored.

        Raises:
            ValueError: missing, unknown or invalid configuration key.
        """
        config = dict(ugent_surrogate_config or {})
        missing = _REQUIRED_CONFIG_KEYS - config.keys()
        if missing:
            raise ValueError(
                f"ugent_surrogate_config is missing required key(s) {sorted(missing)}; "
                f"e.g. {{'ukc_percent': 363}}. Supported UKC values: {SUPPORTED_UKC_PERCENT}.")
        unknown = config.keys() - _REQUIRED_CONFIG_KEYS - _OPTIONAL_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"ugent_surrogate_config has unknown key(s) {sorted(unknown)}; allowed: "
                f"{sorted(_REQUIRED_CONFIG_KEYS | _OPTIONAL_CONFIG_KEYS)}.")

        ukc_percent = config["ukc_percent"]
        if ukc_percent not in SUPPORTED_UKC_PERCENT:
            raise ValueError(
                f"ugent_surrogate_config['ukc_percent']={ukc_percent!r} is not a tested towing-tank "
                f"UKC; choose one of {SUPPORTED_UKC_PERCENT}.")

        n_max_rpm = _positive_finite(config, "n_max_rpm", 320.0)
        min_bank_clearance = _positive_finite(config, "min_bank_clearance", 1.0)
        max_bank_clearance = _positive_finite(config, "max_bank_clearance", 4.0)
        if max_bank_clearance <= min_bank_clearance:
            raise ValueError(
                f"ugent_surrogate_config: max_bank_clearance ({max_bank_clearance}) must exceed "
                f"min_bank_clearance ({min_bank_clearance}).")
        ittc_correction = get_validated_bool(config, "ittc_correction", False)

        self.ship_spec: ShipSpecifications = ship_spec
        self.particulars = TripoliParticulars()
        self.coefficients = tripoli_coefficient_sets(ittc_correction)[int(ukc_percent)]
        self.n_max_rpm = n_max_rpm
        self.min_bank_clearance = min_bank_clearance
        self.max_bank_clearance = max_bank_clearance

        self._dist_port = math.inf
        self._dist_starboard = math.inf
        self._bank_distance_provider: Optional[BankDistanceProvider] = None

        # Warn only once per instance (hot path).
        self._warned_reverse_thrust = False
        self._warned_bank_clamp = False

        logger.info("UGentSurrogate: UKC %d %%, ITTC correction %s, n_max %.0f rpm.",
                    int(ukc_percent), ittc_correction, n_max_rpm)

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    def set_bank_distance_provider(self, provider: Optional[BankDistanceProvider]) -> None:
        """
        Set the function that returns bank distances for a state (None removes it).

        Called at every evaluation, so it must be fast. Use a bound method,
        not a lambda: deepcopy re-binds a bound method to the copied object.
        """
        self._bank_distance_provider = provider

    def set_bank_distances(self, dist_port: Optional[float], dist_starboard: Optional[float]) -> None:
        """Set bank distances from midship [m]; None or inf = no bank on that side."""
        self._dist_port = self._validate_distance("dist_port", dist_port)
        self._dist_starboard = self._validate_distance("dist_starboard", dist_starboard)

    def get_ukc(self, pos: Optional[object] = None) -> float:
        """Under-keel clearance [m] of the configured condition (constant; pos ignored)."""
        return self.coefficients.ukc_percent / 100.0 * self.particulars.draft

    @staticmethod
    def _validate_distance(name: str, value: Optional[float]) -> float:
        if value is None:
            return math.inf
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"UGentSurrogate.set_bank_distances: {name} is NaN")
        return value

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Accelerations (du, dv, dr) in the framework frame [m/s², m/s², rad/s²].

        Args:
            state:        Vessel state (framework frame: v, r positive to port).
            rudder_angle: Z-drive azimuth [rad], positive to starboard.
            thrust:       Propeller-rate command n / n_max, in [0, 1].
        """
        if self._bank_distance_provider is not None:
            self.set_bank_distances(*self._bank_distance_provider(state))
        du, dv_report, dr_report = self._accelerations_report_frame(
            u=state.u, v=-state.v, r=-state.r, delta=rudder_angle, thrust_cmd=thrust)
        return du, -dv_report, -dr_report

    def _accelerations_report_frame(
        self, u: float, v: float, r: float, delta: float, thrust_cmd: float,
    ) -> Tuple[float, float, float]:
        """Equations of motion (eq. 11, 17, 19) in the report frame (y starboard)."""
        p = self.particulars
        c = self.coefficients
        thrust_force = self._propeller_thrust(u, thrust_cmd)
        bank_term = self._bank_term(u)
        sin_d, cos_d = math.sin(delta), math.cos(delta)

        surge = (thrust_force * cos_d
                 - c.X_u * u - c.X_uabsu * u * abs(u)
                 + p.mass * v * r
                 + c.X_uabsv * u * abs(v))
        sway = (thrust_force * sin_d
                - p.mass * u * r
                + c.Y_vabsv * v * abs(v)
                + c.Y_uv * u * v
                + c.Y_uabsu_yB * bank_term)
        yaw = (-thrust_force * (p.length_pp / 2.0) * sin_d
               - p.mass * p.x_g * u * r
               + c.N_rabsr * r * abs(r)
               + c.N_ur * u * r
               + c.N_uabsu_yB * bank_term)

        return (surge / (p.mass - c.X_udot),
                sway / (p.mass - c.Y_vdot),
                yaw / (p.i_zz - c.N_rdot))

    def _propeller_thrust(self, u: float, thrust_cmd: float) -> float:
        """Total thrust of both Z-drives [N] (eq. 12-14)."""
        if thrust_cmd < 0.0:
            if not self._warned_reverse_thrust:
                logger.warning(
                    "UGentSurrogate received thrust=%.4f < 0 (reverse thrust). The "
                    "propeller model is not defined for negative rpm; clamping to 0. "
                    "Further occurrences are not logged.", thrust_cmd)
                self._warned_reverse_thrust = True
            thrust_cmd = 0.0

        p = self.particulars
        n_rps = thrust_cmd * self.n_max_rpm / 60.0
        if n_rps < 1e-6:
            return 0.0  # avoids J = u / 0
        advance_ratio = u * (1.0 - p.wake_fraction) / (n_rps * p.d_prop)
        k_t = p.kt_0 - p.kt_slope * advance_ratio
        return p.n_propellers * k_t * p.rho_water * n_rps ** 2 * p.d_prop ** 4

    def _bank_term(self, u: float) -> float:
        """u|u| / y'_BP - u|u| / y'_BS (eq. 17-18); 0 without banks."""
        u_abs_u = u * abs(u)
        return u_abs_u / self._clearance(self._dist_port) - \
            u_abs_u / self._clearance(self._dist_starboard)

    def _clearance(self, distance: float) -> float:
        """y' = (d - B/2) / B, clamped below at min; inf (no force) above max."""
        if math.isinf(distance):
            return math.inf
        beam = self.particulars.beam
        y_nondim = (distance - beam / 2.0) / beam
        if y_nondim > self.max_bank_clearance:
            return math.inf
        if y_nondim < self.min_bank_clearance:
            if not self._warned_bank_clamp:
                logger.warning(
                    "UGentSurrogate: bank clearance y'=%.3f below the validated minimum "
                    "%.3f; bank forces are held at that minimum. Further occurrences "
                    "are not logged.", y_nondim, self.min_bank_clearance)
                self._warned_bank_clamp = True
            return self.min_bank_clearance
        return y_nondim
