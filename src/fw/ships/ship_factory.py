import numpy as np

from enum import auto, Enum
from fw.ships.ship_specs import ShipSpecifications


# Maximum rudder deflection per ship family [deg]. This is the single source of
# truth for the rudder limit: actuators, PySimEnv action scaling and the
# UAntwerp surrogate's rudder normalisation all read
# `ShipSpecifications.max_rudder_angle`, which is set from these values.
MYZAKO_MAX_RUDDER_ANGLE_DEG = 75.0
TRIPOLI_MAX_RUDDER_ANGLE_DEG = 35.0
B01_MAX_RUDDER_ANGLE_DEG = 35.0

# Geometric scale of the B01 towing-tank model (Myzako, full scale / model).
B01_MODEL_SCALE = 25.0

# Minimum thrust command (n / n_max) of the B01: 20 % n_max is the lowest
# propeller rate in every B01 test of the DDSHIP report (Appendix B). Below it
# the wheel effect cannot be trimmed with 35 deg rudder at deep water either,
# and the model spins up at almost zero speed (see fw.dynamics.ugent_b01_surrogate).
B01_MIN_THRUST = 0.2


class ShipModel(Enum):
    """
    Enumeration of supported ship specification presets.

    MYZAKO_29 / MYZAKO_37 and TRIPOLI_38 / TRIPOLI_45 carry the FH-identified
    Fossen hull coefficients for one draft each (T = 2.85 / 3.65 m and
    T = 3.78 / 4.50 m). MYZAKO_DEFAULT and TRIPOLI are heuristic presets.

    TRIPOLI_38 / TRIPOLI_45 are Fossen/Nomoto-only for now, see
    FOSSEN_ONLY_SHIP_MODELS.

    B01 is the MODEL-SCALE (lambda = 25) Myzako towing-tank model used by
    UGentB01Surrogate (L_pp 4.32 m, m 260.8 kg, I_zz 307.4 kg m^2). Its
    actuator and kinematic envelope is Froude-scaled from the full-scale
    defaults (see _MODEL_SCALE_HULLS); it is intended for the towing tank,
    not for the full-scale Scheldt environment, and has no FH model.
    """

    MYZAKO_DEFAULT = auto()
    MYZAKO_29 = auto()
    MYZAKO_37 = auto()
    TRIPOLI = auto()
    TRIPOLI_38 = auto()
    TRIPOLI_45 = auto()
    B01 = auto()


# TEMPORARY (until the ShipModel refactor): the FH code paths, the dataset
# generators and the validation tools identify the Tripoli hull with
# `ship_model == ShipModel.TRIPOLI` (azipod controls, exercise file, FH
# rudder sign). These presets are Tripoli hulls but would take the
# conventional-rudder branch there, so they are rejected up front by every
# FH entry point (FhSimBridge, FhTabularModel) via require_fh_supported().
# UGentSurrogate and UAntwerpSurrogate reject them through their own checks.
FOSSEN_ONLY_SHIP_MODELS = frozenset({ShipModel.TRIPOLI_38, ShipModel.TRIPOLI_45})

# Model-scale presets: there is no FH ship model for them at all.
MODEL_SCALE_SHIP_MODELS = frozenset({ShipModel.B01})

# Every preset the FH entry points refuse (see require_fh_supported).
FH_UNSUPPORTED_SHIP_MODELS = FOSSEN_ONLY_SHIP_MODELS | MODEL_SCALE_SHIP_MODELS


def require_full_scale_ship(ship_model: ShipModel, where: str) -> None:
    """
    Fail fast if a model-scale preset is used in a full-scale environment.

    The PySim environments use the full-scale Western Scheldt map, distances,
    speeds and rewards; a model-scale ship (ShipModel.B01, towing tank) needs
    a towing-tank environment instead.

    Args:
        ship_model: The ship model about to be used.
        where:      Name of the caller, for the error message.

    Raises:
        ValueError: If ship_model is in MODEL_SCALE_SHIP_MODELS.
    """

    if ship_model in MODEL_SCALE_SHIP_MODELS:
        raise ValueError(
            f"{where}: ShipModel.{ship_model.name} is a model-scale (towing tank) preset, but this "
            f"environment is full scale (Western Scheldt map, distances, speeds and rewards). Use a "
            f"towing-tank environment for it."
        )


def require_hull_coefficients(ship_model: ShipModel, where: str) -> None:
    """
    Fail fast if a model-scale preset is used with a coefficient-based model.

    Fossen and Nomoto read the hull coefficients in ShipSpecifications. For
    model-scale presets (ShipModel.B01) those are the full-scale heuristic
    placeholders, which are not meaningful at model scale; such presets are
    only supported by their own surrogate (Model.UGENT_B01_SURROGATE).

    Args:
        ship_model: The ship model about to be used.
        where:      Name of the caller and dynamics model, for the error message.

    Raises:
        ValueError: If ship_model is in MODEL_SCALE_SHIP_MODELS.
    """

    if ship_model in MODEL_SCALE_SHIP_MODELS:
        raise ValueError(
            f"{where}: ShipModel.{ship_model.name} is a model-scale preset; its Fossen/Nomoto "
            f"coefficients are full-scale placeholders. Use Model.UGENT_B01_SURROGATE."
        )


def require_fh_supported(ship_model: ShipModel, where: str) -> None:
    """
    Fail fast if `ship_model` cannot be used with the FH simulator (yet).

    Args:
        ship_model: The ship model about to be used with FH.
        where:      Name of the caller, for the error message.

    Raises:
        ValueError: If ship_model is in FH_UNSUPPORTED_SHIP_MODELS.
    """

    if ship_model in MODEL_SCALE_SHIP_MODELS:
        raise ValueError(
            f"{where}: ShipModel.{ship_model.name} is a model-scale (towing tank) preset and has no "
            f"FH ship model. Use it with Model.UGENT_B01_SURROGATE."
        )
    if ship_model in FOSSEN_ONLY_SHIP_MODELS:
        raise ValueError(
            f"{where}: ShipModel.{ship_model.name} carries FH-identified Fossen coefficients but is "
            f"not supported by the FH simulator code paths yet (they recognise the Tripoli hull as "
            f"ShipModel.TRIPOLI only). Use it with Model.FOSSEN or Model.NOMOTO, or use "
            f"ShipModel.TRIPOLI for FH."
        )


# FH "Fossen model" identification in full scale, dimensional values as
# delivered in WL memo WL2026M23_002_03 v4.0 (FH columns of Tables 1-4).
# Units: Xudot, Yvdot [kg]; Nrdot [kg m^2]; Xu, Yv [kg/s]; Xuu, Yvv, Yuv [kg/m];
# Nr [kg m^2/s]; Nrr [kg m^2]; Nuv [kg].
# length = LPP. The coefficients are converted with this same length, so the
# scaling in fw.dynamics.fossen reproduces the FH values (to machine precision).
# I_z = None on purpose: FH identified the coefficients with the slender-rod
# inertia m * L^2 / 12 used by fw.dynamics.fossen, so the measured RIZZ (kept
# in the comments for reference) must not be used with these coefficients.
_FH_FOSSEN_HULLS = {
    # Myzako, scale model B01, T = 2.85 m (B0102A01.xml), 200 % UKC
    ShipModel.MYZAKO_29: dict(
        mass=3.13e6, length=108.0, I_z=None,  # RIZZ = 2.29e9 kg m^2 (not used, see note above)
        max_rudder_angle_deg=MYZAKO_MAX_RUDDER_ANGLE_DEG,
        Xudot=-7.994e5, Yvdot=-2.646e6, Nrdot=-1.733e9,
        Xu=1.5634e-12, Xuu=-7.0719e3,
        Yv=-7.6307e3, Yvv=-2.2810e5, Yuv=-1.3364e4,
        Nr=3.1966e7, Nrr=-1.4327e9, Nuv=-2.3127e4,
    ),
    # Myzako, scale model B01, T = 3.65 m (B0101A01.xml), 200 % UKC
    ShipModel.MYZAKO_37: dict(
        mass=4.19e6, length=108.0, I_z=None,  # RIZZ = 2.95e9 kg m^2 (not used, see note above)
        max_rudder_angle_deg=MYZAKO_MAX_RUDDER_ANGLE_DEG,
        Xudot=-1.111e6, Yvdot=-3.601e6, Nrdot=-2.319e9,
        Xu=2.8125e-11, Xuu=-6.2821e3,
        Yv=-7.5016e3, Yvv=-2.3724e5, Yuv=-1.9562e4,
        Nr=3.9387e7, Nrr=-1.7707e9, Nuv=-3.0532e4,
    ),
    # Tripoli / Deseo, scale model E01, T = 3.78 m (E0102A01.xml), 198 % UKC
    ShipModel.TRIPOLI_38: dict(
        mass=0.585e7, length=106.0, I_z=None,  # RIZZ = 0.423e10 kg m^2 (not used, see note above)
        max_rudder_angle_deg=TRIPOLI_MAX_RUDDER_ANGLE_DEG,
        Xudot=-1.026e6, Yvdot=-3.794e6, Nrdot=-2.379e9,
        Xu=0.0, Xuu=-8.3809e3,
        Yv=-9.2787e3, Yvv=-2.0171e5, Yuv=-2.3033e4,
        Nr=5.1032e7, Nrr=-2.3353e9, Nuv=-3.0132e4,
    ),
    # Tripoli / Deseo, scale model E01, T = 4.50 m (E0101A01.xml), 150 % UKC
    ShipModel.TRIPOLI_45: dict(
        mass=0.716e7, length=106.0, I_z=None,  # RIZZ = 0.521e10 kg m^2 (not used, see note above)
        max_rudder_angle_deg=TRIPOLI_MAX_RUDDER_ANGLE_DEG,
        Xudot=-1.270e6, Yvdot=-4.697e6, Nrdot=-2.946e9,
        Xu=0.0, Xuu=-8.2521e3,
        Yv=-1.8003e4, Yvv=-3.0101e5, Yuv=-1.1577e4,
        Nr=5.4531e7, Nrr=-2.4970e9, Nuv=-3.7226e4,
    ),
}


def _fossen_coeffs_from_fh(hull: dict) -> dict:
    """
    Convert dimensional FH Fossen derivatives to the ShipSpecifications ratios.

    Mirrors the scaling in fw.dynamics.fossen.Fossen._initialize_coefficients:
    mass for surge/sway terms, mass*length for N_uv, mass*length^2 for the
    other yaw terms.

    Args:
        hull: One entry of _FH_FOSSEN_HULLS.

    Returns:
        dict: The *_coeff keyword arguments for ShipSpecifications.
    """

    m = hull["mass"]
    L = hull["length"]
    # "not > 0" also rejects NaN
    if not (m > 0.0 and L > 0.0):
        raise ValueError(f"mass and length must be positive, got mass={m}, length={L}")

    return dict(
        X_udot_coeff=hull["Xudot"] / m,
        Y_vdot_coeff=hull["Yvdot"] / m,
        N_rdot_coeff=hull["Nrdot"] / (m * L ** 2),
        X_u_coeff=hull["Xu"] / m,
        Y_v_coeff=hull["Yv"] / m,
        N_r_coeff=hull["Nr"] / (m * L ** 2),
        X_uu_coeff=hull["Xuu"] / m,
        Y_vv_coeff=hull["Yvv"] / m,
        N_rr_coeff=hull["Nrr"] / (m * L ** 2),
        Y_uv_coeff=hull["Yuv"] / m,
        N_uv_coeff=hull["Nuv"] / (m * L),
    )


# Heuristic (not identified) Fossen coefficient set, already in the
# ShipSpecifications ratio form. Shared by all heuristic presets.
_HEURISTIC_FOSSEN_COEFFS = dict(
    X_udot_coeff=-0.05,
    Y_vdot_coeff=-0.5,
    N_rdot_coeff=-0.05,
    X_u_coeff=-0.002,
    Y_v_coeff=-0.02,
    N_r_coeff=-0.001,
    X_uu_coeff=-0.0005,
    Y_vv_coeff=-0.005,
    N_rr_coeff=-0.0005,
    Y_uv_coeff=-0.005,
    N_uv_coeff=-0.0005,
)

# Heuristic presets: hull data only; they all use _HEURISTIC_FOSSEN_COEFFS.
_HEURISTIC_HULLS = {
    ShipModel.MYZAKO_DEFAULT: dict(
        mass=3.86e6, length=110.0,
        max_rudder_angle_deg=MYZAKO_MAX_RUDDER_ANGLE_DEG,
    ),
    # Alternative values noted in the original code (origin not documented):
    # mass 7.16e6 kg (equals the TRIPOLI_45 mass), length 109.0 m.
    ShipModel.TRIPOLI: dict(
        mass=6161870.27,
        length=110.0,
        max_rudder_angle_deg=TRIPOLI_MAX_RUDDER_ANGLE_DEG,
    ),
}

# Model-scale presets (hull data plus an explicit actuator / kinematic envelope).
# The hydrodynamics live in UGentB01Surrogate; the Fossen coefficients in the
# specifications are the heuristic placeholders and are not used by that model.
# Envelope = full-scale defaults Froude-scaled with sqrt(lambda) = 5:
#   velocities / sqrt(lambda): surge 5 m/s -> 1.0 m/s, sway 2 m/s -> 0.4 m/s
#   rates * sqrt(lambda): rudder rate, thrust rate, yaw rate.
# ASSUMPTION: derived, not taken from the DDSHIP report. Jitter thresholds and
# max_thrust are scale-free and unchanged; min_thrust is B01_MIN_THRUST.
_SQRT_LAMBDA = B01_MODEL_SCALE ** 0.5
_MODEL_SCALE_HULLS = {
    # B01 (Myzako model), Table 1 of the DDSHIP WP2.2 SI report.
    ShipModel.B01: dict(
        mass=260.8, length=4.32, I_z=307.4,
        max_rudder_angle_deg=B01_MAX_RUDDER_ANGLE_DEG,
        actuator_envelope=dict(
            min_thrust=B01_MIN_THRUST,
            max_rudder_rate=np.radians(6.0) * _SQRT_LAMBDA,
            max_thrust_rate=0.07 * _SQRT_LAMBDA,
            surge_limits=(0.0, 5.0 / _SQRT_LAMBDA),
            sway_limits=(-2.0 / _SQRT_LAMBDA, 2.0 / _SQRT_LAMBDA),
            yaw_rate_limits=(-0.5 * _SQRT_LAMBDA, 0.5 * _SQRT_LAMBDA),
        ),
    ),
}

_FH_HULL_KEYS = frozenset({
    "mass", "length", "I_z", "max_rudder_angle_deg",
    "Xudot", "Yvdot", "Nrdot", "Xu", "Xuu", "Yv", "Yvv", "Yuv", "Nr", "Nrr", "Nuv",
})
_HEURISTIC_HULL_KEYS = frozenset({"mass", "length", "max_rudder_angle_deg"})
_MODEL_SCALE_HULL_KEYS = frozenset({"mass", "length", "I_z", "max_rudder_angle_deg", "actuator_envelope"})
_ENVELOPE_OVERRIDE_KEYS = frozenset({
    "min_thrust", "max_rudder_rate", "max_thrust_rate", "surge_limits", "sway_limits", "yaw_rate_limits",
})


def _validate_preset_tables() -> None:
    """
    Fail fast at import if the preset tables are inconsistent.

    Checks that every ShipModel member is in exactly one table, that every
    table entry has exactly the expected keys (so a misspelled or extra key
    cannot be silently ignored), and that mass and length are positive.

    Raises:
        RuntimeError: If any check fails.
    """

    tables = (set(_FH_FOSSEN_HULLS), set(_HEURISTIC_HULLS), set(_MODEL_SCALE_HULLS))
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            if tables[i] & tables[j]:
                raise RuntimeError(
                    f"Ship models in both preset tables: {sorted(m.name for m in tables[i] & tables[j])}")
    missing = set(ShipModel) - tables[0] - tables[1] - tables[2]
    if missing:
        raise RuntimeError(f"Ship models without a preset: {sorted(m.name for m in missing)}")

    for table, expected in ((_FH_FOSSEN_HULLS, _FH_HULL_KEYS),
                            (_HEURISTIC_HULLS, _HEURISTIC_HULL_KEYS),
                            (_MODEL_SCALE_HULLS, _MODEL_SCALE_HULL_KEYS)):
        for model, hull in table.items():
            if set(hull) != expected:
                raise RuntimeError(
                    f"{model.name}: unexpected keys {sorted(set(hull) - expected)}, "
                    f"missing keys {sorted(expected - set(hull))}")
            # "not > 0" also rejects NaN
            if not (hull["mass"] > 0.0 and hull["length"] > 0.0):
                raise RuntimeError(f"{model.name}: mass and length must be positive")

    for model, hull in _MODEL_SCALE_HULLS.items():
        if set(hull["actuator_envelope"]) != _ENVELOPE_OVERRIDE_KEYS:
            raise RuntimeError(f"{model.name}: actuator_envelope keys must be {sorted(_ENVELOPE_OVERRIDE_KEYS)}")
        # "not > 0" also rejects NaN
        if not hull["I_z"] > 0.0:
            raise RuntimeError(f"{model.name}: I_z must be positive")


_validate_preset_tables()


class MyzakoShip:
    """
    Minimal ship model with predefined specifications for simulation.

    This class defines a simple, self-contained ship model suitable
    for use in simulators and control experiments. It provides only
    the vessel specifications and does not include dynamics, actuators,
    or environmental effects.

    Attributes:
        specifications: ShipSpecifications object containing physical
            properties, actuator limits, and operational constraints.
    """

    def __init__(self, ship_model: ShipModel):
        """
        Initialize the ship model with parameters specific to *ship_model*.

        Hydrodynamic coefficients (mass, length, added-mass, damping) and the
        maximum rudder angle differ per variant:

            - Maximum rudder angle : Myzako 75.0 deg (MYZAKO_MAX_RUDDER_ANGLE_DEG),
                                     Tripoli 35.0 deg (TRIPOLI_MAX_RUDDER_ANGLE_DEG),
                                     B01 35.0 deg (B01_MAX_RUDDER_ANGLE_DEG)

        The remaining actuator and kinematic limits are shared across all
        full-scale variants (B01 overrides the rate and velocity limits with
        Froude-scaled values and sets min thrust 0.2, see _MODEL_SCALE_HULLS):

            - Maximum rudder rate  : 6.0 deg/s (≈ 0.105 rad/s)
            - Rudder jitter threshold : 0.6 deg
            - Thrust               : min 0.1, max 1.0 (normalized)
            - Maximum thrust rate  : 0.07 /s
            - Thrust jitter threshold : 0.005
            - Surge velocity limits: 0 – 5 m/s
            - Sway velocity limits : -2 – 2 m/s
            - Yaw rate limits      : -0.5 – 0.5 rad/s

        Per-variant hull parameters:

            MYZAKO_29 / MYZAKO_37 (T = 2.85 / 3.65 m):
                LPP 108 m, mass 3.13 / 4.19 × 10⁶ kg, I_z = m L² / 12,
                FH-identified Fossen hull coefficients.

            TRIPOLI_38 / TRIPOLI_45 (T = 3.78 / 4.50 m):
                LPP 106 m, mass 5.85 / 7.16 × 10⁶ kg, I_z = m L² / 12,
                FH-identified Fossen hull coefficients.

            MYZAKO_DEFAULT:
                Length 110 m, mass 3.86 × 10⁶ kg, I_z = m L² / 12,
                heuristic coefficients.

            TRIPOLI:
                Length 110 m, mass 6.16 × 10⁶ kg, I_z = m L² / 12,
                heuristic coefficients.

            B01 (model scale, lambda = 25, T = 0.146 m):
                LPP 4.32 m, mass 260.8 kg, I_zz 307.4 kg m^2 (report Table 1),
                min thrust 0.2 (B01_MIN_THRUST),
                heuristic Fossen placeholders (unused by UGentB01Surrogate).

        Rudder and thrust coefficients (Y_rudder, N_rudder, X_thrust) are the
        heuristic ShipSpecifications defaults for every variant.
        """

        # Shared limits. max_rudder_angle is deliberately NOT in here: every
        # preset must set it explicitly, so a new ship model can never
        # silently inherit another ship's rudder limit.
        default_actuator_envelope = dict(
            max_rudder_rate=np.radians(6.0),
            min_thrust=0.1,
            max_thrust=1.0,
            max_thrust_rate=0.07,
            rudder_jitter_threshold=np.radians(0.6),
            thrust_jitter_threshold=0.005,
            surge_limits=(0.0, 5.0),
            sway_limits=(-2.0, 2.0),
            yaw_rate_limits=(-0.5, 0.5),
        )

        if ship_model in _FH_FOSSEN_HULLS:
            hull = _FH_FOSSEN_HULLS[ship_model]
            coeffs = _fossen_coeffs_from_fh(hull)
            I_z = hull["I_z"]
        elif ship_model in _HEURISTIC_HULLS:
            hull = _HEURISTIC_HULLS[ship_model]
            coeffs = _HEURISTIC_FOSSEN_COEFFS
            I_z = None
        elif ship_model in _MODEL_SCALE_HULLS:
            hull = _MODEL_SCALE_HULLS[ship_model]
            coeffs = _HEURISTIC_FOSSEN_COEFFS
            I_z = hull["I_z"]
            default_actuator_envelope.update(hull["actuator_envelope"])
        else:
            raise ValueError(f"No specifications defined for ship model {ship_model}.")

        self.specifications = ShipSpecifications(
            length=hull["length"],
            mass=hull["mass"],
            I_z=I_z,
            max_rudder_angle=np.radians(hull["max_rudder_angle_deg"]),
            **coeffs,
            **default_actuator_envelope,
        )


def create_ship(ship_model: ShipModel = ShipModel.MYZAKO_DEFAULT) -> MyzakoShip:
    """
    Factory function to create a MyzakoShip instance.

    Args:
        ship_model (ShipModel, optional): Ship model to use. Defaults to MYZAKO_DEFAULT.

    Returns:
        MyzakoShip: A new ship object with default specifications
        suitable for use in simulators and experiments.

    Example:
        >>> ship = create_ship(ShipModel.MYZAKO_DEFAULT)
        >>> ship.specifications.length
        110.0
    """

    if not isinstance(ship_model, ShipModel):
        raise ValueError(f"Ship model '{ship_model}' not supported yet.")
    return MyzakoShip(ship_model)
