"""
The data contract of fw.surrogate.schema: raw column order, encoded features,
angle and wind/current encoding, projection rules and dataset identity.

These lock down what the generator writes, what training encodes and what
UAntwerpSurrogate builds at inference time: a change here changes every
dataset and checkpoint, so it must be deliberate.
"""

import numpy as np
import pytest

from fw.environments.ray_features import RAY_ANGLES_DEG
from fw.surrogate import schema
from fw.surrogate.schema import (
    ENV_ENCODING_ANGLES,
    ENV_ENCODING_BODY,
    check_projection,
    encoded_feature_names,
    get_scenario,
    project_raw_columns,
    select_and_encode,
)

from surrogate_test_helpers import identity

RAYS = tuple(f"ray_{int(a)}" for a in RAY_ANGLES_DEG)
BASE = ("x", "y", "heading", "surge", "sway", "yaw_rate")


def test_ray_layout():
    """24 rays, ray_0 straight ahead, then clockwise in 15 deg steps."""

    assert len(RAY_ANGLES_DEG) == 24
    assert list(RAY_ANGLES_DEG) == [15 * i for i in range(24)]


@pytest.mark.parametrize("name, expected", [
    ("open_water", BASE + ("rudder_action", "thrust_action")),
    ("confined_ukc", BASE + ("ukc", "rudder_action", "thrust_action") + RAYS),
    ("full_env", BASE + ("ukc", "wind_speed", "wind_direction", "current_speed", "current_direction",
                         "rudder_action", "thrust_action") + RAYS),
])
def test_raw_column_order(name, expected):
    """The raw row layout (= dataset X columns = UAntwerpSurrogate's input row)."""

    sc = get_scenario(name)
    assert sc.raw_columns == expected
    assert sc.raw_input_dim == len(expected)


@pytest.mark.parametrize("name, encoding, width", [
    ("open_water", ENV_ENCODING_BODY, 7), ("confined_ukc", ENV_ENCODING_BODY, 32),
    ("full_env", ENV_ENCODING_BODY, 36), ("full_env", ENV_ENCODING_ANGLES, 38),
])
def test_encoded_width_and_names(name, encoding, width):
    """Encoded feature count; x/y are never inputs; heading is encoded as sin/cos."""

    sc = get_scenario(name)
    names = list(encoded_feature_names(sc, encoding))
    assert len(names) == width
    assert "x" not in names and "y" not in names and "heading" not in names
    x = np.zeros((3, sc.raw_input_dim), dtype=np.float32)
    assert select_and_encode(x, sc, encoding).shape == (3, width)


def test_heading_encoding_is_continuous_across_pi():
    """Headings just below +pi and just above -pi are almost the same direction: nearly equal features."""

    sc = get_scenario("open_water")
    x = np.zeros((2, sc.raw_input_dim), dtype=np.float32)
    x[0, sc.col("heading")] = np.pi - 1e-3
    x[1, sc.col("heading")] = -np.pi + 1e-3
    enc = select_and_encode(x, sc, ENV_ENCODING_BODY)
    assert np.max(np.abs(enc[0] - enc[1])) < 3e-3


def _body(heading, wind_speed=0.0, wind_dir=0.0, current_speed=0.0, current_dir=0.0):
    """Encoded body components (wind_from_ahead, wind_from_stbd, current_to_ahead, current_to_stbd)."""

    sc = get_scenario("full_env")
    x = np.zeros((1, sc.raw_input_dim), dtype=np.float32)
    for c, v in (("heading", heading), ("wind_speed", wind_speed), ("wind_direction", wind_dir),
                 ("current_speed", current_speed), ("current_direction", current_dir)):
        x[0, sc.col(c)] = v
    names = list(encoded_feature_names(sc, ENV_ENCODING_BODY))
    enc = select_and_encode(x, sc, ENV_ENCODING_BODY)[0]
    return {n: float(enc[names.index(n)]) for n in
            ("wind_from_ahead", "wind_from_stbd", "current_to_ahead", "current_to_stbd")}


def test_current_body_components():
    """Current: earth-fixed direction it flows TOWARD; components in the ship frame (starboard +)."""

    # Heading East (0), current flowing East: straight ahead.
    b = _body(heading=0.0, current_speed=1.0, current_dir=0.0)
    assert b["current_to_ahead"] == pytest.approx(1.0, abs=1e-6)
    assert b["current_to_stbd"] == pytest.approx(0.0, abs=1e-6)
    # Heading East, current flowing North (counter-clockwise 90 deg = to PORT): starboard component -1.
    b = _body(heading=0.0, current_speed=1.0, current_dir=np.pi / 2)
    assert b["current_to_ahead"] == pytest.approx(0.0, abs=1e-6)
    assert b["current_to_stbd"] == pytest.approx(-1.0, abs=1e-6)
    # Heading North, current flowing East: to starboard.
    b = _body(heading=np.pi / 2, current_speed=1.0, current_dir=0.0)
    assert b["current_to_stbd"] == pytest.approx(1.0, abs=1e-6)


def test_wind_body_components():
    """Wind: apparent, direction it comes FROM relative to the bow, positive = from starboard."""

    b = _body(heading=1.0, wind_speed=2.0, wind_dir=0.0)            # head wind
    assert b["wind_from_ahead"] == pytest.approx(2.0, abs=1e-6)
    assert b["wind_from_stbd"] == pytest.approx(0.0, abs=1e-6)
    b = _body(heading=1.0, wind_speed=2.0, wind_dir=np.pi / 2)      # from starboard
    assert b["wind_from_stbd"] == pytest.approx(2.0, abs=1e-6)
    # Relative to the bow: the ship's heading does not change it.
    assert _body(heading=-2.0, wind_speed=2.0, wind_dir=np.pi / 2) == pytest.approx(b, abs=1e-6)


@pytest.mark.parametrize("source, target, allowed", [
    ("open_water", "open_water", True), ("confined_ukc", "confined_ukc", True), ("full_env", "full_env", True),
    ("confined_ukc", "open_water", True),
    ("full_env", "confined_ukc", False), ("full_env", "open_water", False),
    ("open_water", "confined_ukc", False), ("confined_ukc", "full_env", False),
])
def test_projection_rules(source, target, allowed):
    """A poorer model only from richer data, and never dropping wind/current (their effect is in the targets)."""

    src, tgt = get_scenario(source), get_scenario(target)
    if allowed:
        check_projection(src, tgt)
    else:
        with pytest.raises(ValueError):
            check_projection(src, tgt)


def test_projection_keeps_values():
    """confined_ukc -> open_water keeps exactly the shared columns."""

    src, tgt = get_scenario("confined_ukc"), get_scenario("open_water")
    x = np.arange(2 * src.raw_input_dim, dtype=np.float32).reshape(2, -1)
    out = project_raw_columns(x, src, tgt)
    for c in tgt.raw_columns:
        assert np.array_equal(out[:, tgt.col(c)], x[:, src.col(c)])


def test_identity_roundtrip_and_checks(tmp_path):
    """The identity written with the data is read back; a wrong sway convention or mixed identities are refused."""

    from fw.surrogate.dataset import check_same_identity, load_raw, read_identity

    ident = identity("confined_ukc")
    sc = get_scenario("confined_ukc")
    x = np.zeros((4, sc.raw_input_dim), dtype=np.float32)
    y = np.zeros((4, 3), dtype=np.float32)
    good = tmp_path / "good.npz"
    np.savez(good, X=x, Y=y, episode_id=np.zeros(4, dtype=np.int64), **schema.npz_metadata(ident))
    assert read_identity(good) == ident
    x2, y2, ident2 = load_raw(good)
    assert ident2 == ident and x2.shape == x.shape

    bad = tmp_path / "bad_sway.npz"
    meta = schema.npz_metadata(ident)
    meta[schema.NPZ_SWAY_CONVENTION_KEY] = np.array("starboard_positive")
    np.savez(bad, X=x, Y=y, episode_id=np.zeros(4, dtype=np.int64), **meta)
    with pytest.raises(ValueError):
        read_identity(bad)

    other = tmp_path / "other_ship.npz"
    np.savez(other, X=x, Y=y, episode_id=np.zeros(4, dtype=np.int64),
             **schema.npz_metadata(identity("confined_ukc", ship_model="TRIPOLI", rudder_max_deg=35.0)))
    with pytest.raises(ValueError):
        check_same_identity([good, other], [read_identity(good), read_identity(other)])


def test_provenance_roundtrip(tmp_path):
    """Provenance is stored as JSON text (no pickle) and is optional."""

    from fw.surrogate.dataset import read_provenance

    sc = get_scenario("open_water")
    base = dict(X=np.zeros((1, sc.raw_input_dim), np.float32), Y=np.zeros((1, 3), np.float32),
                episode_id=np.zeros(1, np.int64), **schema.npz_metadata(identity("open_water")))
    with_prov = tmp_path / "with.npz"
    np.savez(with_prov, **base, **schema.provenance_metadata({"seed": 1, "thrust_range": [0.0, 1.0]}))
    without = tmp_path / "without.npz"
    np.savez(without, **base)
    assert read_provenance(with_prov) == {"seed": 1, "thrust_range": [0.0, 1.0]}
    assert read_provenance(without) is None
