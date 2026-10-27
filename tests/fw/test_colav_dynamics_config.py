"""
The COLAV environments (PySimColavEnv / MultiPolicyColavEnv, built on
PySimBaseEnv) pass the UGent surrogate configuration through to the dynamics
model, refuse UGent for ships other than the Tripoli, and compute the signed
cross-track error without np.cross (deprecated for 2-D vectors since NumPy 2.0;
NumPy 2.5 raises).
"""

import numpy as np
import pytest

from fw.dynamics.dynamics_model import Model
from fw.dynamics.ugent_surrogate import UGentSurrogate
from fw.ships.ship_factory import ShipModel

START = np.array([2245.0, -47.0], dtype=np.float32)
TARGET = np.array([-1559.0, 11990.0], dtype=np.float32)


def _colav(**kwargs):
    """A small PySimColavEnv (no rendering)."""

    from fw.environments.py_sim_colav_env import PySimColavEnv

    return PySimColavEnv(time_step=1.0, max_steps=20, ship_pos=START, target_pos=TARGET, **kwargs)


def test_colav_ugent_uses_the_given_config():
    """UGent in a COLAV env gets the configured UKC and runs."""

    env = _colav(dynamics_model=Model.UGENT_SURROGATE, ship_model=ShipModel.TRIPOLI,
                 ugent_surrogate_config={"ukc_percent": 125})
    assert isinstance(env.dynamics, UGentSurrogate)
    assert env.dynamics.coefficients.ukc_percent == 125.0
    env.reset(seed=0)
    obs, *_ = env.step(np.array([0.2, 0.5], dtype=np.float32))
    assert np.all(np.isfinite(obs))


def test_colav_ugent_without_config_names_the_missing_key():
    """Without a config, the UGent error says what is missing."""

    with pytest.raises(ValueError, match="ukc_percent"):
        _colav(dynamics_model=Model.UGENT_SURROGATE, ship_model=ShipModel.TRIPOLI)


def test_colav_ugent_refuses_other_ships():
    """UGent's coefficients are for the Tripoli only."""

    with pytest.raises(ValueError, match="TRIPOLI"):
        _colav(dynamics_model=Model.UGENT_SURROGATE, ship_model=ShipModel.MYZAKO_37,
               ugent_surrogate_config={"ukc_percent": 363})


def test_colav_fossen_unchanged():
    """The default (Fossen) COLAV env still builds and steps without a UGent config."""

    env = _colav(dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    env.reset(seed=0)
    obs, *_ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert np.all(np.isfinite(obs))


def test_build_colav_config_carries_ugent_config():
    """env_factory's COLAV config builder forwards ugent_surrogate_config."""

    from fw.configuration.config import load_config_parameters
    from fw.configuration.env_factory import build_colav_config, build_multi_policy_colav_config

    framework_config = load_config_parameters()
    framework_config["ugent_surrogate_config"] = {"ukc_percent": 43}
    assert build_colav_config(framework_config)["ugent_surrogate_config"] == {"ukc_percent": 43}
    assert build_multi_policy_colav_config(framework_config)["ugent_surrogate_config"] == {"ukc_percent": 43}


def test_line_intersection_cross_product():
    """tools.is_ship_heading_correct (2-D cross products) still finds a crossing ahead."""

    from fw.environments.tools import is_ship_heading_correct

    checkpoint = {"perpendicular_line": [(10.0, -5.0), (10.0, 5.0)]}
    assert is_ship_heading_correct(checkpoint, np.array([0.0, 0.0, 0.0]))          # heading East: crosses x = 10
    assert not is_ship_heading_correct(checkpoint, np.array([0.0, 0.0, np.pi]))    # heading West: never


def test_planner_env_forwards_ugent_config():
    """PySimColavPlannerEnv also passes the UGent configuration through."""

    from fw.environments.py_sim_colav_planner import PySimColavPlannerEnv

    env = PySimColavPlannerEnv(time_step=1.0, max_steps=20, ship_pos=START, target_pos=TARGET,
                               dynamics_model=Model.UGENT_SURROGATE, ship_model=ShipModel.TRIPOLI,
                               ugent_surrogate_config={"ukc_percent": 43})
    assert env.dynamics.coefficients.ukc_percent == 43.0
