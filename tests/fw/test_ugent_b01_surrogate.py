"""
Unit and wiring tests for fw.dynamics.ugent_b01_surrogate (UGent B01 / Myzako, model scale).
"""

import copy
import logging
import math

import numpy as np
import pytest

from fw.dynamics.dynamics_model import Model
from fw.dynamics.ugent_b01_surrogate import (
    B01_COEFFICIENT_SETS, DEFAULT_N_MAX_RPM, TESTED_UKC_PERCENT, UGentB01Surrogate, interpolation_blend)
from fw.ships.ship_factory import (
    B01_MIN_THRUST, ShipModel, create_ship, require_fh_supported, require_full_scale_ship,
    require_hull_coefficients)
from fw.ships.vessel_state import VesselState

START = np.array([2245.0, -47.0], dtype=np.float32)
TARGET = np.array([-1559.0, 11990.0], dtype=np.float32)


@pytest.fixture
def spec():
    """
    Specifications of the model-scale B01 preset.
    """

    return create_ship(ShipModel.B01).specifications


def _state(u=0.8, v=0.0, r=0.0):
    """
    A VesselState at the origin with the given body velocities.
    """

    return VesselState(terminated=False, truncated=False, x=0.0, y=0.0, heading=0.0, u=u, v=v, r=r)


# ---------------------------------------------------------------- ship preset

def test_b01_preset_is_model_scale_with_froude_scaled_envelope(spec):
    """
    ShipModel.B01: model-scale hull, 35 deg rudder, limits Froude-scaled with sqrt(25).
    """

    assert spec.length == pytest.approx(4.32)
    assert spec.mass == pytest.approx(260.8)
    assert spec.I_z == pytest.approx(307.4)
    assert math.degrees(spec.max_rudder_angle) == pytest.approx(35.0)
    assert math.degrees(spec.max_rudder_rate) == pytest.approx(30.0)
    assert spec.min_thrust == B01_MIN_THRUST == 0.2
    assert spec.surge_limits == pytest.approx((0.0, 1.0))
    assert spec.sway_limits == pytest.approx((-0.4, 0.4))
    assert spec.yaw_rate_limits == pytest.approx((-2.5, 2.5))


def test_b01_has_no_fh_model():
    """
    The FH entry points refuse the model-scale preset.
    """

    with pytest.raises(ValueError, match="model-scale"):
        require_fh_supported(ShipModel.B01, "test")


# ---------------------------------------------------------------- coefficients and UKC

def test_tested_ukc_and_deep_water_bank_terms():
    """
    379 % bank terms are the draft's values (coeff_UKC379).
    """

    assert TESTED_UKC_PERCENT == (379.0, 153.0, 35.0)
    deep = B01_COEFFICIENT_SETS[0]
    assert deep.Y_uabsu_yB == -2.74
    assert deep.N_uabsu_yB == 2.08


@pytest.mark.parametrize("ukc, expected", [
    (500.0, 379.0), (379.0, 379.0), (35.0, 35.0), (10.0, 35.0),
])
def test_blend_outside_and_at_ends(ukc, expected):
    """
    At and beyond the tested range one set is used with weight 1.
    """

    blend = interpolation_blend(ukc)
    assert len(blend) == 1
    assert blend[0][0].ukc_percent == expected
    assert blend[0][1] == 1.0


def test_blend_is_linear_in_depth_ratio():
    """
    Between 153 and 35 % the weights are linear in T/(h-T).
    """

    (lo, w_lo), (hi, w_hi) = interpolation_blend(100.0)  # T/(h-T) = 1.0
    assert (lo.ukc_percent, hi.ukc_percent) == (153.0, 35.0)
    assert w_lo + w_hi == pytest.approx(1.0)
    assert w_hi == pytest.approx((1.0 - 100 / 153) / (100 / 35 - 100 / 153))


def test_get_ukc_in_model_scale_metres(spec):
    """
    get_ukc returns metres at model scale and follows set_ukc_percent.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153})
    assert model.get_ukc() == pytest.approx(1.53 * 0.146)
    model.set_ukc_percent(35)
    assert model.get_ukc() == pytest.approx(0.35 * 0.146)


# ---------------------------------------------------------------- golden values

# Expected framework-frame accelerations computed with UGent's draft
# B01_surrogate_model.py, unchanged (calc_accelerations + np.interp as in
# calc_next_state, its own coefficient lists), converted to the framework frame
# (v, r flipped) and to rad. Includes banks at 379 % and a bank at y' = 0.26.
# The draft hard-codes n_max = 1650 rpm, so the comparison uses that value.
# (ukc, u, v, r, rudder [deg], thrust, dist_port, dist_starboard, (du, dv, dr))
_DRAFT_N_MAX_RPM = 1650.0
_GOLDEN = [
    (379, 0.6, -0.02, 0.01, 12.0, 0.7, None, None,
     (-0.0019306847373014027, -0.008520386013294948, 0.007043900979998916)),
    (379, 0.55, 0.01, -0.02, -8.0, 0.6, 0.9, 1.6,
     (-0.0017069684447618503, 0.006938000085016343, -0.001783252444281799)),
    (250, 0.45, 0.03, -0.02, -20.0, 0.5, 1.0, 2.5,
     (-0.005549483895524197, 0.005523069359577196, -0.004563961083283947)),
    (153, 0.5, -0.01, -0.03, 30.0, 0.9, 0.9, None,
     (0.011038363844618055, 0.0005172485970625474, 0.013348129179354837)),
    (153, 0.4, -0.0, -0.01, 5.0, 0.5, 0.35, None,
     (0.0008723077081815083, 0.007169517400212031, -0.004826809071493691)),
    (80, 0.3, -0.0, 0.02, -5.0, 0.4, None, 1.2,
     (-0.004212118566981342, -0.004413615021909742, 0.00031281148640107786)),
    (35, 0.25, 0.01, -0.01, 25.0, 1.0, 0.8, 1.5,
     (0.014681722366652022, -0.005498414660434207, 0.004186099487787289)),
]


@pytest.mark.parametrize("ukc, u, v, r, rudder_deg, thrust, d_port, d_stbd, expected", _GOLDEN)
def test_matches_ugent_draft(spec, ukc, u, v, r, rudder_deg, thrust, d_port, d_stbd, expected):
    """
    Accelerations equal UGent's draft (frame and unit conversion applied).
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": ukc, "n_max_rpm": _DRAFT_N_MAX_RPM})
    model.set_bank_distances(d_port, d_stbd)
    got = model._compute_accelerations(_state(u, v, r), math.radians(rudder_deg), thrust)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-15)


# ---------------------------------------------------------------- low thrust

def test_default_n_max_is_tank_setting(spec):
    """
    n_max defaults to 1700 rpm (A. Van Damme): self-propulsion about 1 m/s at 379 % UKC.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 379})
    assert DEFAULT_N_MAX_RPM == 1700.0
    assert model.n_max_rpm == DEFAULT_N_MAX_RPM
    u, _ = model._straight_running_trim(1.0)
    assert u == pytest.approx(0.96, abs=0.01)


def test_n_max_override(spec):
    """
    n_max_rpm in the config replaces the default.
    """

    assert UGentB01Surrogate(spec, {"ukc_percent": 379, "n_max_rpm": 1000}).n_max_rpm == 1000.0


@pytest.mark.parametrize("ukc, expected", [(379, 0.176), (153, 0.150), (100, 0.256), (35, 0.585)])
def test_min_straight_running_thrust(spec, ukc, expected):
    """
    Smallest thrust whose wheel effect can be trimmed with 35 deg rudder (n_max 1700 rpm).
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": ukc})
    assert model.min_straight_running_thrust == pytest.approx(expected, abs=2e-3)


def test_min_thrust_preset_covers_deep_water(spec):
    """
    B01_MIN_THRUST is trimmable at the tested deep and intermediate UKC.
    """

    for ukc in (379, 153):
        assert UGentB01Surrogate(spec, {"ukc_percent": ukc}).min_straight_running_thrust < B01_MIN_THRUST


def test_low_thrust_warns_once(spec, caplog):
    """
    Thrust below the UKC's minimum logs one warning per instance.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 35})
    with caplog.at_level(logging.WARNING, logger="fw.dynamics.ugent_b01_surrogate"):
        model._compute_accelerations(_state(), 0.0, 0.3)
        model._compute_accelerations(_state(), 0.0, 0.3)
        model._compute_accelerations(_state(), 0.0, 0.9)
    assert sum("cannot keep a straight course" in m for m in caplog.messages) == 1


def test_no_warning_above_min_thrust(spec, caplog):
    """
    No low-thrust warning above the minimum.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 379})
    with caplog.at_level(logging.WARNING, logger="fw.dynamics.ugent_b01_surrogate"):
        model._compute_accelerations(_state(), 0.0, B01_MIN_THRUST)
    assert not any("straight course" in m for m in caplog.messages)


def test_min_thrust_ignores_bank_distances(spec):
    """
    The minimum is computed without banks and leaves the set bank distances untouched.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153})
    model.set_bank_distances(1.0, 2.0)
    before = model._compute_accelerations(_state(), 0.0, 0.5)
    model.set_ukc_percent(153)
    assert model._compute_accelerations(_state(), 0.0, 0.5) == before


# ---------------------------------------------------------------- conventions

def test_positive_rudder_turns_to_port(spec):
    """
    Framework convention: positive rudder -> r > 0 (bow to port).
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 379})
    _, _, dr_port = model._compute_accelerations(_state(), math.radians(20), 0.8)
    _, _, dr_zero = model._compute_accelerations(_state(), 0.0, 0.8)
    _, _, dr_stbd = model._compute_accelerations(_state(), math.radians(-20), 0.8)
    assert dr_port > dr_zero > dr_stbd
    assert dr_port > 0.0 > dr_stbd


def test_port_bank_attracts_and_turns_bow_away(spec):
    """
    A port bank pulls the ship to port and turns the bow to starboard.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153})
    beam = model.particulars.beam
    model.set_bank_distances(beam / 2 + 1.5 * beam, None)  # y' = 1.5
    _, dv, dr = model._compute_accelerations(_state(), 0.0, 0.0)  # no propeller: no wheel effect
    assert dv > 0.0
    assert dr < 0.0


# ---------------------------------------------------------------- bank limits

def test_far_banks_act_without_cut_off_by_default(spec):
    """
    Default: no upper cut-off (both tank walls were in the identification).
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153})
    without = model._compute_accelerations(_state(), 0.0, 0.0)
    model.set_bank_distances(10.0, None)  # y' ~ 21
    assert model._compute_accelerations(_state(), 0.0, 0.0) != without


def test_optional_cut_off(spec):
    """
    max_bank_clearance ignores banks farther than that y'.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153, "max_bank_clearance": 4.0})
    without = model._compute_accelerations(_state(), 0.0, 0.0)
    model.set_bank_distances(10.0, 10.0)
    assert model._compute_accelerations(_state(), 0.0, 0.0) == without


def test_close_banks_unclamped_by_default(spec):
    """
    By default, y' is used as in the draft: a bank at y' = 0.2 acts more strongly than at y' = 1.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153})
    beam = model.particulars.beam
    model.set_bank_distances(beam / 2 + 1.0 * beam, None)
    _, dv_1, _ = model._compute_accelerations(_state(), 0.0, 0.0)
    model.set_bank_distances(beam / 2 + 0.2 * beam, None)
    _, dv_02, _ = model._compute_accelerations(_state(), 0.0, 0.0)
    assert dv_02 == pytest.approx(5.0 * dv_1)


def test_bank_at_or_inside_ship_is_floored(spec):
    """
    y' <= 0 (ship at or inside the bank) is held at the default floor: finite, attraction kept.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153})
    beam = model.particulars.beam
    model.set_bank_distances(beam / 2 - 0.1, None)  # y' < 0
    _, dv_inside, _ = model._compute_accelerations(_state(), 0.0, 0.0)
    model.set_bank_distances(beam / 2 + 1e-3 * beam, None)  # y' = floor
    _, dv_floor, _ = model._compute_accelerations(_state(), 0.0, 0.0)
    assert math.isfinite(dv_inside) and dv_inside > 0.0
    assert dv_inside == pytest.approx(dv_floor)


def test_optional_min_clearance(spec):
    """
    min_bank_clearance = 1.0 holds closer banks at the closest validated y'.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153, "min_bank_clearance": 1.0})
    beam = model.particulars.beam
    model.set_bank_distances(beam / 2 + 1.0 * beam, None)
    at_min = model._compute_accelerations(_state(), 0.0, 0.0)
    model.set_bank_distances(beam / 2 + 0.2 * beam, None)
    assert model._compute_accelerations(_state(), 0.0, 0.0) == pytest.approx(at_min)


# ---------------------------------------------------------------- inputs and config

def test_reverse_thrust_clamped_to_zero(spec):
    """
    Negative thrust commands act as zero rpm.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 379})
    assert model._compute_accelerations(_state(), 0.0, -0.3) == \
        model._compute_accelerations(_state(), 0.0, 0.0)


@pytest.mark.parametrize("config", [
    {},
    {"ukc_percent": 0},
    {"ukc_percent": float("nan")},
    {"ukc_percent": "153"},
    {"ukc_percent": True},
    {"ukc_percent": 153, "ittc_correction": True},
    {"ukc_percent": 153, "n_max_rpm": -1},
    {"ukc_percent": 153, "min_bank_clearance": 2.0, "max_bank_clearance": 1.0},
])
def test_invalid_config_raises(spec, config):
    """
    Missing, unknown or invalid keys raise ValueError.
    """

    with pytest.raises(ValueError):
        UGentB01Surrogate(spec, config)


def test_nan_bank_distance_raises(spec):
    """
    NaN bank distances are refused.
    """

    model = UGentB01Surrogate(spec, {"ukc_percent": 153})
    with pytest.raises(ValueError):
        model.set_bank_distances(float("nan"), None)


class _BankOwner:
    """
    Holds a B01 model whose bank provider is one of its own bound methods (as PySimEnv).
    """

    def __init__(self, spec, d_port):
        """
        Create the model and bind the provider.
        """

        self.d_port = d_port
        self.dynamics = UGentB01Surrogate(spec, {"ukc_percent": 153})
        self.dynamics.set_bank_distance_provider(self.banks)

    def banks(self, state):
        """
        Bank distances [m] of this owner.
        """

        return self.d_port, None


def test_deepcopy_rebinds_bound_provider(spec):
    """
    deepcopy re-binds a bound-method provider to the copied owner.
    """

    owner = _BankOwner(spec, d_port=0.9)
    clone = copy.deepcopy(owner)
    assert clone.dynamics._bank_distance_provider.__self__ is clone
    assert clone.dynamics._compute_accelerations(_state(), 0.1, 0.5) == \
        owner.dynamics._compute_accelerations(_state(), 0.1, 0.5)
    clone.d_port = 2.0  # the copy follows its own owner, not the original
    assert clone.dynamics._compute_accelerations(_state(), 0.1, 0.5) != \
        owner.dynamics._compute_accelerations(_state(), 0.1, 0.5)


# ---------------------------------------------------------------- simulator and environments

def test_ship_simulator_refuses_wind_and_current(spec):
    """
    EnvironmentModel's full-scale wind/current are refused for the B01.
    """

    from fw.simulators.ship_simulator import ShipSimulator

    model = UGentB01Surrogate(spec, {"ukc_percent": 379})
    ShipSimulator(spec, model, dt=0.1)  # no wind/current: fine
    for wind, current in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="wind"):
            ShipSimulator(spec, model, dt=0.1, wind=wind, current=current)


def test_full_scale_guard():
    """
    require_full_scale_ship refuses only model-scale presets.
    """

    with pytest.raises(ValueError, match="model-scale"):
        require_full_scale_ship(ShipModel.B01, "test")
    for other in set(ShipModel) - {ShipModel.B01}:
        require_full_scale_ship(other, "test")


def _colav(**kwargs):
    """
    A small PySimColavEnv (no rendering).
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv

    return PySimColavEnv(time_step=1.0, max_steps=20, ship_pos=START, target_pos=TARGET, **kwargs)


def test_full_scale_envs_refuse_b01():
    """
    PySimEnv and the COLAV envs (Western Scheldt, full scale) refuse ShipModel.B01.
    """

    from fw.environments.py_sim_env import PySimEnv

    with pytest.raises(ValueError, match="model-scale"):
        _colav(dynamics_model=Model.UGENT_B01_SURROGATE, ship_model=ShipModel.B01,
               ugent_b01_surrogate_config={"ukc_percent": 153})
    with pytest.raises(ValueError, match="model-scale"):
        PySimEnv(dynamics_model=Model.UGENT_B01_SURROGATE, ship_model=ShipModel.B01,
                 ugent_b01_surrogate_config={"ukc_percent": 153})


def test_b01_surrogate_refuses_other_ships():
    """
    The B01 coefficients are for ShipModel.B01 only.
    """

    with pytest.raises(ValueError, match="B01"):
        _colav(dynamics_model=Model.UGENT_B01_SURROGATE, ship_model=ShipModel.MYZAKO_37,
               ugent_b01_surrogate_config={"ukc_percent": 379})


def test_model_scale_env_wiring():
    """
    With SUPPORTS_MODEL_SCALE_SHIPS (as a towing-tank env will set), PySimEnv wires the B01 surrogate.
    """

    from fw.environments.py_sim_env import PySimEnv

    class _ModelScalePySimEnv(PySimEnv):
        """
        PySimEnv that accepts model-scale ships (stand-in for the towing-tank env).
        """

        SUPPORTS_MODEL_SCALE_SHIPS = True

    env = _ModelScalePySimEnv(dynamics_model=Model.UGENT_B01_SURROGATE, ship_model=ShipModel.B01,
                              ugent_b01_surrogate_config={"ukc_percent": 153})
    assert isinstance(env.dynamics, UGentB01Surrogate)
    assert env.dynamics._bank_distance_provider.__self__ is env
    assert env.get_keel_clearance() == pytest.approx(1.53 * 0.146)
    env.reset(seed=0)
    obs, *_ = env.step(np.array([0.2, 0.5], dtype=np.float32))
    assert np.all(np.isfinite(obs))


def test_build_colav_config_carries_b01_config():
    """
    env_factory's COLAV config builders forward ugent_b01_surrogate_config.
    """

    from fw.configuration.config import load_config_parameters
    from fw.configuration.env_factory import build_colav_config, build_multi_policy_colav_config

    framework_config = load_config_parameters()
    framework_config["ugent_b01_surrogate_config"] = {"ukc_percent": 35}
    assert build_colav_config(framework_config)["ugent_b01_surrogate_config"] == {"ukc_percent": 35}
    assert build_multi_policy_colav_config(framework_config)["ugent_b01_surrogate_config"] == {"ukc_percent": 35}


def test_hull_coefficient_guard():
    """
    require_hull_coefficients refuses only model-scale presets.
    """

    with pytest.raises(ValueError, match="model-scale"):
        require_hull_coefficients(ShipModel.B01, "test")
    for other in set(ShipModel) - {ShipModel.B01}:
        require_hull_coefficients(other, "test")


@pytest.mark.parametrize("dynamics_model", [Model.FOSSEN, Model.NOMOTO])
@pytest.mark.parametrize("base", ["pysim", "colav"])
def test_model_scale_envs_refuse_fossen_and_nomoto(dynamics_model, base):
    """
    Even where model-scale ships are accepted, Fossen/Nomoto refuse B01 (full-scale placeholder coefficients).
    """

    if base == "pysim":
        from fw.environments.py_sim_env import PySimEnv as Base
    else:
        from fw.environments.py_sim_colav_env import PySimColavEnv as Base

    class _ModelScaleEnv(Base):
        """
        Env that accepts model-scale ships (stand-in for the towing-tank env).
        """

        SUPPORTS_MODEL_SCALE_SHIPS = True

    kwargs = {} if base == "pysim" else dict(time_step=1.0, max_steps=20, ship_pos=START, target_pos=TARGET)
    with pytest.raises(ValueError, match=dynamics_model.name):
        _ModelScaleEnv(dynamics_model=dynamics_model, ship_model=ShipModel.B01, **kwargs)
