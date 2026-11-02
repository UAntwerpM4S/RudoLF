"""
UKC/environment lookup: fw.dynamics.wcu_lookup.WcuLookup (nearest row,
optional max distance, .csv.gz) and how UAntwerpSurrogate uses it (nearest
row always used; far lookups reported and counted).
"""

import logging

import numpy as np
import pytest

from fw.ships.ship_factory import ShipModel, create_ship
from fw.surrogate.schema import get_scenario

from surrogate_test_helpers import make_state

SC = get_scenario("confined_ukc")


def _lookup_csv(path, points):
    """
    A confined_ukc-layout lookup CSV (raw row, 3 targets, episode_id) with the given (x, y, ukc) points.
    """

    rows = np.zeros((len(points), SC.raw_input_dim + 4))
    for i, (x, y, ukc) in enumerate(points):
        rows[i, SC.col("x")], rows[i, SC.col("y")], rows[i, SC.col("ukc")] = x, y, ukc
    rows[:, -1] = -1
    np.savetxt(path, rows, delimiter=",", fmt="%.6g")
    return path


POINTS = [(0.0, 0.0, 5.0), (10.0, 0.0, 7.0), (0.0, 10.0, 9.0)]


def _wcu(path):
    """
    WcuLookup on a confined lookup CSV (x, y, UKC only).
    """

    from fw.dynamics.wcu_lookup import WcuLookup

    return WcuLookup(str(path), x_col=SC.col("x"), y_col=SC.col("y"), ukc_col=SC.col("ukc"),
                     wind_speed_col=None, wind_dir_col=None, current_speed_col=None, current_dir_col=None)


@pytest.mark.parametrize("name", ["lookup.csv", "lookup.csv.gz"])
def test_nearest_row_and_max_distance(tmp_path, name):
    """
    Nearest row is returned (plain and gzip); with max_distance, nothing beyond it.
    """

    lk = _wcu(_lookup_csv(tmp_path / name, POINTS))
    assert lk.get_ukc(1.0, 1.0) == pytest.approx(5.0)
    assert lk.get_ukc(9.0, 1.0) == pytest.approx(7.0)
    assert lk.get_ukc(-3.0, 12.0) == pytest.approx(9.0)
    sample = lk.get_sample(10.0, 3.0)
    assert sample.env.ukc == pytest.approx(7.0) and sample.distance == pytest.approx(3.0)
    assert lk.get_ukc(5.0, 100.0) == pytest.approx(9.0)                       # no limit: nearest anyway
    assert lk.get_ukc(5.0, 100.0, max_distance=50.0) is None                  # beyond the limit


def _confined_surrogate(tiny_models, lookup):
    """
    A myzako_confined UAntwerpSurrogate on the tiny checkpoint with the given lookup CSV.
    """

    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate

    spec = create_ship(ShipModel.MYZAKO_37).specifications
    return UAntwerpSurrogate(spec, {"preset": "myzako_confined", "checkpoint": str(tiny_models["confined_ukc"]),
                                    "lookup_csv": str(lookup)}, ship_model=ShipModel.MYZAKO_37)


def test_surrogate_uses_lookup_and_reports_far_rows(tiny_models, tmp_path, caplog):
    """
    Near: lookup UKC, nothing reported. Far: the nearest row is still used, warned once, counted every time.
    """

    from fw.dynamics import uantwerp_surrogate

    model = _confined_surrogate(tiny_models, _lookup_csv(tmp_path / "lk.csv.gz", POINTS))
    state = make_state(x=1.0, y=1.0, heading=0.0, u=2.0, v=0.0, r=0.0)
    with caplog.at_level(logging.WARNING, logger=uantwerp_surrogate.__name__):
        env = model.get_env(np.array([1.0, 1.0]), state)
        assert env.ukc == pytest.approx(5.0)
        assert model.lookup_far_count == 0 and not caplog.records

        far = uantwerp_surrogate.LOOKUP_WARN_DISTANCE_M + 20.0
        for _ in range(3):
            env = model.get_env(np.array([far, 0.0]), state)
        assert env.ukc == pytest.approx(7.0)                                   # behaviour unchanged
        assert model.lookup_far_count == 3
        assert model.lookup_max_distance == pytest.approx(far - 10.0)
        warnings = [r for r in caplog.records if "lookup" in r.getMessage()]
        assert len(warnings) == 1


def test_missing_lookup_falls_back_to_default_ukc(tiny_models, tmp_path, caplog):
    """
    A missing lookup file: a warning, then the preset's default UKC.
    """

    with caplog.at_level(logging.WARNING):
        model = _confined_surrogate(tiny_models, tmp_path / "does_not_exist.csv.gz")
    assert model.wcu_lookup is None
    assert any("not found" in r.getMessage() for r in caplog.records)
    env = model.get_env(np.array([0.0, 0.0]), make_state(x=0.0, y=0.0, heading=0.0, u=1.0, v=0.0, r=0.0))
    assert env.ukc == pytest.approx(model.default_ukc)
