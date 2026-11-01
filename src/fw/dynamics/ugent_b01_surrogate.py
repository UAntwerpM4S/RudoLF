"""
UGent 3-DOF confined-water manoeuvring model of the B01 (Myzako), MODEL SCALE.

Source: Van Damme, Delefortrie & Eloot (2026), DDSHIP WP2 D2.3, FH Reports
23_002_6 (equations §2.3, eq. 20-26; coefficients Table 3; ship data Table 1),
and A. Van Damme's draft B01_surrogate_model.py (2026-08-20).

Notes:
    - MODEL SCALE (lambda = 25), for the towing tank: L_pp 4.32 m, m 260.8 kg.
      The coefficients are used exactly as identified; no Froude conversion and
      no ITTC correction. Velocities, forces, distances, the propeller rate and
      time are all model-scale quantities. Use with ShipModel.B01.
    - UKC: accelerations are interpolated linearly in T/(h-T) = 100/UKC[%]
      between the tested conditions 379, 153 and 35 %, and the closest tested
      condition is used outside that range (as calc_next_state in the draft).
    - Bank coefficients at 379 % are not in Table 3 ('/'). The draft's values
      are used: Y = -2.74, N = 2.08 (B01_surrogate_model.py, coeff_UKC379).
      Note: extrapolating linearly in T/(h-T) from 153 % and 35 % (the method
      of coef_extrapolation.py) gives Y = -2.757, N = 2.084, so the draft's Y
      differs by 0.6 % (open question for UGent).
    - Banks: y' = (d - B/2) / B (eq. 18), as in the draft: no upper cut-off by
      default (max_bank_clearance = None) and no clamp in the data range. The
      report (§3.1.2, and "analogous" for the B01 in §3.2.2) computes the
      distances to both banks from the basin geometry. Only y' below
      min_bank_clearance (default 0.001, i.e. the ship at or inside the bank,
      where the draft divides by zero or flips sign) is held at that value.
      A larger min_bank_clearance (e.g. 1.0) can be configured as a guard:
      the B01 bank terms were identified at y' = 1.25 and 2.0 (report Table 17)
      and validated at y' = 1.0 and 1.5 (Figures 28-31).
    - Frames: the equations are in the report frame (y starboard); the framework
      frame has y to port, so v and r change sign at the boundary. The rudder
      angle needs no sign change: in the report equations a positive angle gives
      Y > 0 and N < 0, i.e. the ship turns to port, the framework convention.
    - Units: the draft works in degrees (rudder angle, yaw rate and yaw
      acceleration); this class uses radians throughout, like the framework.
    - Rudder: limited to 35 deg by ShipModel.B01 (actuator), well below 45 deg
      where the sin(2 delta) lift model peaks. Identified at 20-25 deg only.
    - Wheel effect and low thrust: the bias terms Y_n n and N_n n (n in rpm) do
      not depend on speed, while the rudder force scales with u|u|; at 153 % and
      35 % UKC the yaw damping N_r|r| is ~0 as well, so the yaw damping left
      (N_ur u r) vanishes as u -> 0. At low thrust the ship cannot be held on a
      straight course and spins up at almost zero speed. Steady straight
      running with |rudder| <= 35 deg (ShipModel.B01) needs at least
      n/n_max = 0.18 (379 %), 0.16 (153 %), 0.26 (100 %), 0.41 (60 %),
      0.60 (35 % UKC); the rudder needed for that at 0.2 / 0.4 / 0.7 / 1.0 is
      about 15 / 5 / 2.5 / 1.7 deg at 379 and 153 %, and -, -, 19, 7.7 deg at
      35 %. ShipModel.B01 therefore has min_thrust = 0.2 (the lowest tested
      propeller rate); below the UKC-specific minimum (computed at
      construction, see min_straight_running_thrust) a warning is logged once.
    - Validity: straight running and small manoeuvres. The model under-predicts
      the rudder's course-changing ability in zigzags (report §5.2.2) and does
      not predict turning circles (report §6).
    - Wind and current: not part of the model, and EnvironmentModel's fixed
      full-scale accelerations (0.35 m/s²) are about 9x the B01's maximum thrust
      acceleration, so ShipSimulator refuses wind/current for this model
      (supports_environment_model = False).
"""

import logging
import math
import numbers
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase

logger = logging.getLogger(__name__)

# state -> (dist_port, dist_starboard) [m]; None = no bank on that side.
BankDistanceProvider = Callable[[VesselState], Tuple[Optional[float], Optional[float]]]

_CONFIG_NAME = "ugent_b01_surrogate_config"
_REQUIRED_CONFIG_KEYS = frozenset({"ukc_percent"})
_OPTIONAL_CONFIG_KEYS = frozenset({"n_max_rpm", "min_bank_clearance", "max_bank_clearance"})

# Default lower bound on y': only keeps 1/y' finite and positive (ship at or
# inside the bank); within the data range y' is used unclamped, as in the draft.
DEFAULT_MIN_BANK_CLEARANCE = 1e-3


@dataclass(frozen=True)
class B01Particulars:
    """Model-scale main particulars and propulsion data of the B01 (Table 1, §2.3)."""

    mass: float = 260.8               # kg
    length_pp: float = 4.32           # m
    beam: float = 0.458               # m
    i_zz: float = 307.4               # kg m²
    x_g: float = 0.043                # m
    d_prop: float = 0.0724            # m, single ducted propeller
    draft: float = 0.146              # m
    wake_fraction: float = 0.15       # J = u (1 - w) / (n D)
    kt_0: float = 0.6673              # K_T = kt_0 - kt_slope * J (propeller + nozzle)
    kt_slope: float = 0.7518
    rho_water: float = 1000.0         # kg/m³, towing tank (fresh water)


@dataclass(frozen=True)
class B01Coefficients:
    """Coefficient set for one UKC condition (Table 3, model scale, rudder in rad)."""

    ukc_percent: float
    X_udot: float
    X_u: float
    X_uabsu: float
    X_uabsv: float
    X_deltauabsu: float  # rudder drag, multiplies (1 - cos 2δ) u|u|
    Y_vdot: float
    Y_vabsv: float
    Y_uv: float
    Y_deltauabsu: float  # rudder lift, multiplies sin 2δ u|u|
    Y_n: float           # wheel effect, sway [N per rpm]
    Y_uabsu_yB: float    # bank term, sway
    N_rdot: float
    N_rabsr: float
    N_ur: float
    N_deltauabsu: float  # rudder moment, multiplies (L/2) sin 2δ u|u|
    N_n: float           # wheel effect, yaw [Nm per rpm]
    N_uabsu_yB: float    # bank term, yaw

    @property
    def depth_ratio(self) -> float:
        """T/(h-T) = 100 / UKC[%]."""
        return 100.0 / self.ukc_percent


def b01_coefficient_sets() -> Tuple[B01Coefficients, ...]:
    """Model-scale coefficient sets (Table 3), sorted by increasing T/(h-T): 379, 153, 35 %."""
    ukc153 = B01Coefficients(
        ukc_percent=153.0,
        X_udot=-9.68e1, X_u=1.29e0, X_uabsu=7.99e0, X_uabsv=-7.37e1, X_deltauabsu=-1.08e1,
        Y_vdot=-2.61e2, Y_vabsv=-7.11e2, Y_uv=-9.47e1, Y_deltauabsu=1.93e1, Y_n=7.34e-4,
        Y_uabsu_yB=-6.34e0,
        N_rdot=-1.40e-40, N_rabsr=-4.10e-11, N_ur=-2.26e1, N_deltauabsu=-7.62e0, N_n=-4.93e-4,
        N_uabsu_yB=4.10e0,
    )
    ukc35 = B01Coefficients(
        ukc_percent=35.0,
        X_udot=-9.68e1, X_u=2.24e1, X_uabsu=1.16e1, X_uabsv=-1.04e2, X_deltauabsu=-1.51e1,
        Y_vdot=-2.61e2, Y_vabsv=-7.00e2, Y_uv=-4.66e2, Y_deltauabsu=2.38e1, Y_n=1.14e-3,
        Y_uabsu_yB=-2.66e1,
        N_rdot=-3.46e-26, N_rabsr=-1.56e-16, N_ur=-6.23e1, N_deltauabsu=-7.72e0, N_n=-4.44e-4,
        N_uabsu_yB=1.55e1,
    )

    # Deep water: bank terms not measured; the draft's values (coeff_UKC379).
    ukc379 = B01Coefficients(
        ukc_percent=379.0,
        X_udot=-9.68e1, X_u=8.46e-1, X_uabsu=7.84e0, X_uabsv=-2.26e1, X_deltauabsu=-3.31e1,
        Y_vdot=-2.61e2, Y_vabsv=-4.04e2, Y_uv=-1.43e2, Y_deltauabsu=2.57e1, Y_n=8.62e-4,
        Y_uabsu_yB=-2.74e0,
        N_rdot=-5.45e-19, N_rabsr=-1.61e2, N_ur=-1.49e1, N_deltauabsu=-5.95e0, N_n=-3.95e-4,
        N_uabsu_yB=2.08e0,
    )
    return ukc379, ukc153, ukc35


B01_COEFFICIENT_SETS = b01_coefficient_sets()
TESTED_UKC_PERCENT = tuple(c.ukc_percent for c in B01_COEFFICIENT_SETS)  # (379, 153, 35)

# ((coefficient set, weight), ...) used to blend accelerations at one UKC.
_Blend = Tuple[Tuple[B01Coefficients, float], ...]


def interpolation_blend(ukc_percent: float) -> _Blend:
    """
    Coefficient sets and weights for linear interpolation in T/(h-T) = 100/UKC.

    Outside the tested range the closest tested set is used with weight 1
    (same as np.interp in UGent's draft).

    Args:
        ukc_percent: UKC [%], > 0.

    Returns:
        One or two (coefficient set, weight) pairs; the weights sum to 1.
    """
    x = 100.0 / ukc_percent
    sets = B01_COEFFICIENT_SETS
    if x <= sets[0].depth_ratio:
        return ((sets[0], 1.0),)
    if x >= sets[-1].depth_ratio:
        return ((sets[-1], 1.0),)
    for lo, hi in zip(sets[:-1], sets[1:]):
        if x <= hi.depth_ratio:
            w = (x - lo.depth_ratio) / (hi.depth_ratio - lo.depth_ratio)
            return ((lo, 1.0 - w), (hi, w))
    raise AssertionError("unreachable: depth ratio not bracketed")  # pragma: no cover


def _positive_finite(config: Mapping, key: str, default: Optional[float] = None) -> float:
    """config[key] (or default) as a finite float > 0; raises ValueError otherwise."""
    value = config.get(key, default)
    if isinstance(value, (bool, str)) or not isinstance(value, numbers.Real) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{_CONFIG_NAME}[{key!r}] must be a finite number > 0, got {value!r}")
    return float(value)


class UGentB01Surrogate(DynamicsBase):
    """
    UGent manoeuvring model of the B01 (Myzako) at model scale.

    UKC is set from the configuration (set_ukc_percent changes it); bank
    distances come from a provider (set_bank_distance_provider), queried with
    the current state at every evaluation, or else from the last
    set_bank_distances() call. Without either, no banks are present.
    """

    supports_environment_model = False
    """Model scale: EnvironmentModel's full-scale wind/current must not be added (see DynamicsBase)."""

    def __init__(self, ship_spec: ShipSpecifications, ugent_b01_surrogate_config: Optional[Mapping] = None):
        """
        Args:
            ship_spec: Kept for the DynamicsBase interface; the model uses
                its own model-scale B01 particulars.
            ugent_b01_surrogate_config:
                ukc_percent (required): UKC [%] = (h-T)/T * 100, > 0. Interpolated
                    between 379, 153 and 35 %, closest tested set outside.
                n_max_rpm (default 1650): model-scale propeller rate at thrust 1.0.
                min_bank_clearance (default 0.001): closer banks act as at this y'
                    (only guards y' <= 0; e.g. 1.0 = closest validated y').
                max_bank_clearance (default None): farther banks are ignored;
                    None = no cut-off (as identified in the towing tank).

        Raises:
            ValueError: missing, unknown or invalid configuration key.
        """
        config = dict(ugent_b01_surrogate_config or {})
        missing = _REQUIRED_CONFIG_KEYS - config.keys()
        if missing:
            raise ValueError(
                f"{_CONFIG_NAME} is missing required key(s) {sorted(missing)}; "
                f"e.g. {{'ukc_percent': 379}}. Tested UKC values: {TESTED_UKC_PERCENT}.")
        unknown = config.keys() - _REQUIRED_CONFIG_KEYS - _OPTIONAL_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"{_CONFIG_NAME} has unknown key(s) {sorted(unknown)}; allowed: "
                f"{sorted(_REQUIRED_CONFIG_KEYS | _OPTIONAL_CONFIG_KEYS)}.")

        n_max_rpm = _positive_finite(config, "n_max_rpm", 1650.0)
        min_bank_clearance = _positive_finite(config, "min_bank_clearance", DEFAULT_MIN_BANK_CLEARANCE)
        if config.get("max_bank_clearance") is None:
            max_bank_clearance = math.inf
        else:
            max_bank_clearance = _positive_finite(config, "max_bank_clearance")
            if max_bank_clearance <= min_bank_clearance:
                raise ValueError(
                    f"{_CONFIG_NAME}: max_bank_clearance ({max_bank_clearance}) must exceed "
                    f"min_bank_clearance ({min_bank_clearance}).")

        self.ship_spec: ShipSpecifications = ship_spec
        self.particulars = B01Particulars()
        self.n_max_rpm = n_max_rpm
        self.min_bank_clearance = min_bank_clearance
        self.max_bank_clearance = max_bank_clearance

        self._dist_port = math.inf
        self._dist_starboard = math.inf
        self._bank_distance_provider: Optional[BankDistanceProvider] = None

        # Warn only once per instance (hot path).
        self._warned_reverse_thrust = False
        self._warned_bank_clamp = False
        self._warned_low_thrust = False

        self.ukc_percent: float = math.nan
        self.min_straight_running_thrust: float = math.nan
        self._blend: _Blend = ()
        self.set_ukc_percent(_positive_finite(config, "ukc_percent"))

        logger.info("UGentB01Surrogate (model scale): UKC %.1f %%, n_max %.0f rpm, bank y' "
                    "clamp %g, cut-off %s.", self.ukc_percent, n_max_rpm, min_bank_clearance,
                    "none" if math.isinf(max_bank_clearance) else f"{max_bank_clearance:.2f}")

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    def set_ukc_percent(self, ukc_percent: float) -> None:
        """
        Set the UKC [%] = (h-T)/T * 100 (must be finite and > 0).

        Values outside the tested range 35-379 % use the closest tested
        coefficient set (logged as a warning). Also recomputes
        min_straight_running_thrust for the new UKC.

        Args:
            ukc_percent: Under-keel clearance [%].
        """
        ukc_percent = _positive_finite({"ukc_percent": ukc_percent}, "ukc_percent")
        lo, hi = min(TESTED_UKC_PERCENT), max(TESTED_UKC_PERCENT)
        if not lo <= ukc_percent <= hi:
            logger.warning(
                "UGentB01Surrogate: UKC %.1f %% is outside the tested range %.0f-%.0f %%; the "
                "coefficients of the closest tested UKC are used.", ukc_percent, lo, hi)
        self.ukc_percent = ukc_percent
        self._blend = interpolation_blend(ukc_percent)
        self.min_straight_running_thrust = self._min_straight_running_thrust(
            float(self.ship_spec.max_rudder_angle))
        self._warned_low_thrust = False

    def set_bank_distance_provider(self, provider: Optional[BankDistanceProvider]) -> None:
        """
        Set the function that returns bank distances for a state (None removes it).

        Called at every evaluation, so it must be fast. Use a bound method,
        not a lambda: deepcopy re-binds a bound method to the copied object.

        Args:
            provider: state -> (dist_port, dist_starboard) [m], or None.
        """
        self._bank_distance_provider = provider

    def set_bank_distances(self, dist_port: Optional[float], dist_starboard: Optional[float]) -> None:
        """
        Set bank distances from midship [m]; None or inf = no bank on that side.

        Args:
            dist_port:      Distance to the port bank [m].
            dist_starboard: Distance to the starboard bank [m].
        """
        self._dist_port = self._validate_distance("dist_port", dist_port)
        self._dist_starboard = self._validate_distance("dist_starboard", dist_starboard)

    def get_ukc(self, pos: Optional[object] = None) -> float:
        """
        Under-keel clearance of the configured condition (constant; pos ignored).

        Args:
            pos: Ignored.

        Returns:
            UKC [m, model scale].
        """
        return self.ukc_percent / 100.0 * self.particulars.draft

    @staticmethod
    def _validate_distance(name: str, value: Optional[float]) -> float:
        """Bank distance [m] as a float: None becomes inf (no bank); NaN raises ValueError."""
        if value is None:
            return math.inf
        value = float(value)
        if math.isnan(value):
            raise ValueError(f"UGentB01Surrogate.set_bank_distances: {name} is NaN")
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
            rudder_angle: Rudder angle [rad]; positive = the ship turns to PORT
                (framework convention, see DynamicsBase).
            thrust:       Propeller-rate command n / n_max, in [0, 1].

        Returns:
            Tuple (du, dv, dr).
        """
        if self._bank_distance_provider is not None:
            self.set_bank_distances(*self._bank_distance_provider(state))
        if thrust < self.min_straight_running_thrust and not self._warned_low_thrust:
            logger.warning(
                "UGentB01Surrogate: thrust %.3f is below %.3f, the minimum at UKC %.1f %% for which "
                "the wheel effect can be trimmed with the maximum rudder angle; the ship cannot keep "
                "a straight course. Further occurrences are not logged.",
                thrust, self.min_straight_running_thrust, self.ukc_percent)
            self._warned_low_thrust = True

        du, dv_report, dr_report = self._blended_accelerations_report_frame(
            state.u, -state.v, -state.r, rudder_angle, thrust)
        return du, -dv_report, -dr_report

    def _blended_accelerations_report_frame(
        self, u: float, v: float, r: float, delta: float, thrust_cmd: float,
    ) -> Tuple[float, float, float]:
        """Accelerations at the configured UKC in the report frame (y starboard), current banks."""
        n_rpm = self._propeller_rate_rpm(thrust_cmd)
        thrust_force = self._propeller_thrust(u, n_rpm)
        bank_term = self._bank_term(u)

        du = dv = dr = 0.0
        for coefficients, weight in self._blend:
            a_u, a_v, a_r = self._accelerations_report_frame(
                coefficients, u, v, r, delta, n_rpm, thrust_force, bank_term)
            du += weight * a_u
            dv += weight * a_v
            dr += weight * a_r
        return du, dv, dr

    def _straight_running_trim(self, thrust_cmd: float) -> Tuple[float, float]:
        """
        Steady straight running (v = r = 0, no banks) at a thrust command.

        Solves du = 0 for u and dr = 0 for the rudder angle (dr is linear in
        sin 2δ; the rudder drag couples them, so a few fixed-point iterations).

        Returns:
            (u [m/s], rudder angle [rad]); the angle is NaN if no |δ| <= 45 deg trims it.
        """
        delta, u = 0.0, 0.0
        for _ in range(20):
            lo, hi = 0.0, 5.0
            for _ in range(50):
                u = 0.5 * (lo + hi)
                if self._blended_accelerations_report_frame(u, 0.0, 0.0, delta, thrust_cmd)[0] > 0.0:
                    lo = u
                else:
                    hi = u
            dr_0 = self._blended_accelerations_report_frame(u, 0.0, 0.0, 0.0, thrust_cmd)[2]
            dr_45 = self._blended_accelerations_report_frame(u, 0.0, 0.0, math.pi / 4.0, thrust_cmd)[2]
            if dr_45 == dr_0:
                return u, math.nan
            sin_2d = -dr_0 / (dr_45 - dr_0)
            if abs(sin_2d) > 1.0:
                return u, math.nan
            new_delta = 0.5 * math.asin(sin_2d)
            if abs(new_delta - delta) < 1e-9:
                return u, new_delta
            delta = new_delta
        return u, delta

    def _min_straight_running_thrust(self, max_rudder_angle: float) -> float:
        """
        Smallest thrust command for which straight running can be trimmed with |δ| <= max_rudder_angle.

        Evaluated without banks at the configured UKC (bisection; the trim
        angle decreases monotonically with thrust). Returns 1.0 if even full
        thrust cannot be trimmed.
        """
        saved = (self._dist_port, self._dist_starboard)
        self._dist_port = self._dist_starboard = math.inf
        try:
            def trimmable(thrust_cmd: float) -> bool:
                """True if straight running at thrust_cmd needs at most max_rudder_angle."""
                delta = self._straight_running_trim(thrust_cmd)[1]
                return not math.isnan(delta) and abs(delta) <= max_rudder_angle

            if not trimmable(1.0):
                return 1.0
            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                if trimmable(mid):
                    hi = mid
                else:
                    lo = mid
            return hi
        finally:
            self._dist_port, self._dist_starboard = saved

    def _accelerations_report_frame(
        self, c: B01Coefficients, u: float, v: float, r: float, delta: float,
        n_rpm: float, thrust_force: float, bank_term: float,
    ) -> Tuple[float, float, float]:
        """
        Equations of motion (eq. 24-26) in the report frame (y starboard), one UKC set.

        delta [rad], n_rpm [rpm], thrust_force [N] and bank_term [m²/s²] do not
        depend on the coefficient set.
        """
        p = self.particulars
        u_abs_u = u * abs(u)
        sin_2d = math.sin(2.0 * delta)

        surge = (thrust_force
                 - c.X_u * u - c.X_uabsu * u_abs_u
                 + p.mass * v * r
                 + c.X_uabsv * u * abs(v)
                 + c.X_deltauabsu * (1.0 - math.cos(2.0 * delta)) * u_abs_u)
        sway = (-p.mass * u * r
                + c.Y_uv * u * v
                + c.Y_vabsv * v * abs(v)
                + c.Y_deltauabsu * sin_2d * u_abs_u
                + c.Y_n * n_rpm
                + c.Y_uabsu_yB * bank_term)
        yaw = (-p.mass * p.x_g * u * r
               + c.N_ur * u * r
               + c.N_rabsr * r * abs(r)
               + c.N_deltauabsu * (p.length_pp / 2.0) * sin_2d * u_abs_u
               + c.N_n * n_rpm
               + c.N_uabsu_yB * bank_term)

        return (surge / (p.mass - c.X_udot),
                sway / (p.mass - c.Y_vdot),
                yaw / (p.i_zz - c.N_rdot))

    def _propeller_rate_rpm(self, thrust_cmd: float) -> float:
        """Propeller rate n [rpm] from the command n / n_max; negative commands clamp to 0."""
        if thrust_cmd < 0.0:
            if not self._warned_reverse_thrust:
                logger.warning(
                    "UGentB01Surrogate received thrust=%.4f < 0 (reverse thrust). The "
                    "propeller model is not defined for negative rpm; clamping to 0. "
                    "Further occurrences are not logged.", thrust_cmd)
                self._warned_reverse_thrust = True
            thrust_cmd = 0.0
        return thrust_cmd * self.n_max_rpm

    def _propeller_thrust(self, u: float, n_rpm: float) -> float:
        """Thrust of the single ducted propeller [N] (§2.3, T = K_T ρ n² D⁴)."""
        p = self.particulars
        n_rps = n_rpm / 60.0
        if n_rps < 1e-6:
            return 0.0  # avoids J = u / 0
        advance_ratio = u * (1.0 - p.wake_fraction) / (n_rps * p.d_prop)
        k_t = p.kt_0 - p.kt_slope * advance_ratio
        return k_t * p.rho_water * n_rps ** 2 * p.d_prop ** 4

    def _bank_term(self, u: float) -> float:
        """u|u| / y'_BP - u|u| / y'_BS (eq. 25-26); 0 without banks."""
        u_abs_u = u * abs(u)
        return u_abs_u / self._clearance(self._dist_port) - \
            u_abs_u / self._clearance(self._dist_starboard)

    def _clearance(self, distance: float) -> float:
        """y' = (d - B/2) / B, held at min_bank_clearance below it; inf (no force) above the cut-off."""
        if math.isinf(distance):
            return math.inf
        beam = self.particulars.beam
        y_nondim = (distance - beam / 2.0) / beam
        if y_nondim > self.max_bank_clearance:
            return math.inf
        if y_nondim < self.min_bank_clearance:
            if not self._warned_bank_clamp:
                logger.warning(
                    "UGentB01Surrogate: bank clearance y'=%.4f below min_bank_clearance %g; "
                    "bank forces are held at that value. Further occurrences are not logged.",
                    y_nondim, self.min_bank_clearance)
                self._warned_bank_clamp = True
            return self.min_bank_clearance
        return y_nondim
