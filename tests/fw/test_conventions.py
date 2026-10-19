"""
Regression tests for RudoLF's sign conventions -- no FH installation needed.

Framework conventions (fw.ships.vessel_state, fw.dynamics.dynamics_model):
  * heading: 0 = East, counter-clockwise positive (ray_0 = bow, rays clockwise)
  * sway v: PORT positive;  yaw rate r: counter-clockwise (to port) positive
  * rudder command: positive = the ship turns to PORT, for every ship
  * FH WindState: direction the wind comes FROM, relative to the bow,
    positive = from starboard

The FH side of these conventions (FH's own sway and rudder signs) can only be
measured with FH itself: check_sway_sign.py and check_rudder_sign.py. These
tests pin down the framework side, so a later edit cannot silently flip it.

See fw/CONVENTIONS.md for the full list.

Run from src:
    python -m pytest fw/tests/test_conventions.py
    python -m fw.tests.test_conventions          (without pytest)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np

from fw.ships.ship_factory import ShipModel
from fw.dynamics.fh_tabular_model import FhTabularModel, fh_rudder_sign
from fw.dynamics.uantwerp_surrogate import _bow_and_starboard, apparent_wind_for_state
from fw.surrogate import schema

TOL = 1e-9


def _close(a, b, tol=TOL):
    """Assert that `a` and `b` are equal within `tol`."""
    assert np.allclose(a, b, atol=tol), f"{a} != {b}"


# ---------------------------------------------------------------- heading frame

def test_bow_and_starboard_frame():
    """Heading 0 points East with starboard South; heading +90 deg points North with starboard East."""

    bow, stbd = _bow_and_starboard(0.0)
    _close(bow, [1.0, 0.0])            # heading 0 = East
    _close(stbd, [0.0, -1.0])          # starboard of an East-going ship = South
    bow, stbd = _bow_and_starboard(np.pi / 2)
    _close(bow, [0.0, 1.0])            # counter-clockwise: +90 deg = North
    _close(stbd, [1.0, 0.0])


# ---------------------------------------------------------------- apparent wind
# A ship at rest, heading 0 (East), measures a 10 m/s wind from its bow: the
# true wind comes FROM the East. Every case below re-expresses it for another
# ship; the expected values follow from that alone.

REC = dict(app_speed=10.0, app_dir=0.0, rec_heading=0.0, rec_u=0.0, rec_v=0.0)


def _wind(heading, u=0.0, v=0.0, rec=REC):
    """
    Apparent wind (speed, direction) for a ship at `heading` with surge `u` and sway `v`, from
    the recording `rec`.
    """

    return apparent_wind_for_state(**rec, heading=heading, u=u, v=v)


def test_wind_same_ship_unchanged():
    """Re-expressing a wind for the recording ship itself returns it unchanged."""

    rec = dict(app_speed=7.0, app_dir=0.4, rec_heading=0.7, rec_u=4.0, rec_v=0.3)
    speed, direction = _wind(0.7, 4.0, 0.3, rec=rec)
    _close([speed, direction], [7.0, 0.4])


def test_wind_at_rest_rotated_heading():
    """
    A ship at rest heading North sees the East wind from starboard (+90 deg); heading South,
    from port (-90 deg).
    """

    speed, direction = _wind(np.pi / 2)          # heading North: East is starboard
    _close([speed, direction], [10.0, np.pi / 2])
    speed, direction = _wind(-np.pi / 2)         # heading South: East is port
    _close([speed, direction], [10.0, -np.pi / 2])


def test_wind_moving_into_and_away_from_wind():
    """
    Sailing at 3 m/s into the wind adds to the apparent speed (13 m/s from ahead); sailing away
    from it subtracts (7 m/s from astern).
    """

    speed, direction = _wind(0.0, u=3.0)         # into the wind: 10 + 3 from ahead
    _close([speed, direction], [13.0, 0.0])
    speed, direction = _wind(np.pi, u=3.0)       # heading West, wind from astern: 10 - 3
    _close(speed, 7.0)
    _close(abs(direction), np.pi)


def test_wind_pure_sway():
    """
    In calm air, sway to port gives a wind from port (-90 deg) and sway to starboard a wind from
    starboard (+90 deg).
    """

    calm = dict(REC, app_speed=0.0)
    speed, direction = _wind(0.0, v=2.0, rec=calm)    # moving to PORT: wind from port
    _close([speed, direction], [2.0, -np.pi / 2])
    speed, direction = _wind(0.0, v=-2.0, rec=calm)   # moving to starboard: wind from starboard
    _close([speed, direction], [2.0, np.pi / 2])


def test_wind_recording_ship_motion_removed():
    """
    The recording ship's own surge and port-positive sway are removed before the wind is
    re-expressed for another ship.
    """

    # Recording ship heading North at 4 m/s, sway 1 m/s to port (= West), in
    # the same true wind from the East. Its apparent "from" vector in
    # (ahead, starboard): true (0, 10) + (u, -v) = (4, 9).
    rec = dict(app_speed=np.hypot(4.0, 9.0), app_dir=np.arctan2(9.0, 4.0),
               rec_heading=np.pi / 2, rec_u=4.0, rec_v=1.0)
    speed, direction = _wind(0.0, rec=rec)       # a ship at rest heading East
    _close([speed, direction], [10.0, 0.0], tol=1e-9)


# ---------------------------------------------------------------- FH rudder / sway

def test_fh_rudder_sign_values():
    """fh_rudder_sign: +1 for the TRIPOLI, -1 for the MYZAKO_37, ValueError for an unknown ship."""

    assert fh_rudder_sign(ShipModel.TRIPOLI) == 1.0       # azipod: + turns to port
    assert fh_rudder_sign(ShipModel.MYZAKO_37) == -1.0    # rudder control: + turns to starboard
    try:
        fh_rudder_sign("NEW_SHIP")                         # unknown ship: no silent default
    except ValueError:
        pass
    else:
        raise AssertionError("fh_rudder_sign accepted an unknown ship")


class _Recorder:
    """Mock FH control: records every value it is given."""

    def __init__(self, max_angle=35.0):
        """Start with no recorded calls; `max_angle` is what getMaximumAngle returns."""
        self.calls = []
        self._max = max_angle

    def __getattr__(self, name):
        """
        Any control method: getMaximumAngle returns max_angle, every other call is recorded as
        (name, args).
        """

        if name == "getMaximumAngle":
            return lambda: self._max
        return lambda *a: self.calls.append((name, a))


def _mock_engine(ship_model, vog=(5.0, 0.4), yaw_deg=1.5):
    """
    Mock FH engine for FhTabularModel with fixed velocity over ground and yaw rate [deg/s].

    Returns:
        (engine, recorder of the steering control).
    """

    rudder, prop = _Recorder(), _Recorder()
    itf = SimpleNamespace(
        getAzipodControls=lambda: [rudder], getRudderControls=lambda: [rudder],
        getPropellerControls=lambda: [prop],
        getShipVelocityOverGround=lambda: SimpleNamespace(x=vog[0], y=vog[1]),
        getShipYawRate=lambda: yaw_deg)
    engine = SimpleNamespace(ship_model=ship_model, ship_interface=itf,
                             math_model=SimpleNamespace(simulateSeconds=lambda dt: None))
    return engine, rudder


def _sent(rudder, method):
    """First argument of every recorded call of `method` on the recorder."""
    return [a[0] for name, a in rudder.calls if name == method]


def test_fh_tabular_tripoli_rudder_and_sway():
    """
    TRIPOLI: the rudder goes to setDesiredDirection unchanged, in degrees; sway and yaw rate come
    back with FH's signs.
    """

    engine, rudder = _mock_engine(ShipModel.TRIPOLI)
    model = FhTabularModel(SimpleNamespace(max_rudder_angle=np.radians(35.0)), engine)
    u, v, r = model.step(None, np.radians(10.0), 0.5, 1.0)
    _close(_sent(rudder, "setDesiredDirection"), [10.0])     # sent as is, in degrees
    _close([u, v, r], [5.0, 0.4, np.radians(1.5)])           # sway and yaw rate as FH gives them


def test_fh_tabular_myzako_rudder_both_scalings():
    """
    MYZAKO_37: setControlValue gets the negated rudder, normalized by the framework maximum, with
    and without rudder angle scaling.
    """

    for scaling, framework_max in ((False, 35.0), (True, 30.0)):
        engine, rudder = _mock_engine(ShipModel.MYZAKO_37)
        model = FhTabularModel(SimpleNamespace(max_rudder_angle=np.radians(framework_max)), engine)
        model.step(None, np.radians(10.0), 0.5, 1.0, enable_rudder_angle_scaling=scaling)
        _close(_sent(rudder, "setControlValue"), [-10.0 / framework_max])   # negated for FH


# ---------------------------------------------------------------- dataset identity

class _Npz(dict):
    """dict standing in for a loaded .npz file (adds the `files` attribute)."""

    @property
    def files(self):
        """Keys, like NpzFile.files."""
        return list(self)


def _identity_npz(**override):
    """Mock .npz with the metadata of a valid full_env identity; `override` replaces or adds keys."""

    ident = schema.DatasetIdentity(schema.FULL_ENV.name, schema.RAY_LAYOUT_VERSION, "TRIPOLI", 35.0)
    data = _Npz(schema.npz_metadata(ident))
    data.update({k: np.array(v) for k, v in override.items()})
    return data


def test_npz_identity_sway_convention():
    """
    read_npz_identity accepts the current sway convention and rejects an .npz with an old or
    missing sway_convention.
    """

    assert schema.read_npz_identity(_identity_npz(), "ok.npz").sway_convention == schema.SWAY_CONVENTION
    for bad in (_identity_npz(**{schema.NPZ_SWAY_CONVENTION_KEY: "starboard_positive"}),
                {k: v for k, v in _identity_npz().items() if k != schema.NPZ_SWAY_CONVENTION_KEY}):
        try:
            schema.read_npz_identity(_Npz(bad), "old.npz")
        except ValueError:
            continue
        raise AssertionError("an .npz with an old or missing sway_convention was accepted")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
