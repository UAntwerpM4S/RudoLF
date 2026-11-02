"""
fw.surrogate.episode_time: rebuilding the order of the (shuffled) rows within
each episode, and the whole-step time helpers.

Must: order normal, slow and turning episodes exactly; skip (never mis-order)
episodes whose order is ambiguous, e.g. a fully stopped ship.
"""

import numpy as np
import pytest

from fw.surrogate.episode_time import step_index, step_threshold, steps_from_times
from fw.surrogate.schema import SURROGATE_DT


def _episode(n, u0, v0=0.0, r0=0.0, x0=100.0, y0=200.0, h0=0.3, decel=0.0, sub=10):
    """
    FH-like episode: integrated with `sub` substeps; returns float32 (state (n, 6), delta (n, 3)).
    """

    x, y, h, u, v, r = x0, y0, h0, u0, v0, r0
    states, deltas = [], []
    for _ in range(n):
        s = (x, y, h, u, v, r)
        for _ in range(sub):
            dt = SURROGATE_DT / sub
            x += dt * (u * np.cos(h) - v * np.sin(h))
            y += dt * (u * np.sin(h) + v * np.cos(h))
            h += dt * r
            u += dt * (-decel * u)
        states.append(s)
        deltas.append((u - s[3], v - s[4], r - s[5]))
    st = np.array(states)
    st[:, 2] = (st[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return st.astype(np.float32), np.array(deltas).astype(np.float32)


def _shuffled(episodes, seed=0):
    """
    Concatenate episodes, shuffle all rows; returns (state, delta, episode_id, true step).
    """

    st = np.concatenate([e[0] for e in episodes])
    de = np.concatenate([e[1] for e in episodes])
    ep = np.concatenate([np.full(len(e[0]), i) for i, e in enumerate(episodes)])
    true = np.concatenate([np.arange(len(e[0])) for e in episodes])
    perm = np.random.default_rng(seed).permutation(len(st))
    return st[perm], de[perm], ep[perm], true[perm]


@pytest.mark.parametrize("name, kwargs", [
    ("normal", dict(n=300, u0=4.0, v0=0.1, r0=0.02)),
    ("turning", dict(n=300, u0=5.0, v0=-0.3, r0=0.12)),
    ("slow_0.8", dict(n=200, u0=0.8)),
    ("decelerating", dict(n=600, u0=3.0, decel=0.05)),
    ("two_rows_slow", dict(n=2, u0=0.5)),
    ("one_row", dict(n=1, u0=3.0)),
])
def test_order_is_rebuilt_exactly(name, kwargs):
    """
    Every row gets its true step index (several episodes mixed and shuffled).
    """

    st, de, ep, true = _shuffled([_episode(**kwargs), _episode(n=150, u0=3.0, x0=900.0), _episode(**kwargs)])
    steps, n_ok, n_bad = step_index(st, de, ep)
    assert n_bad == 0 and n_ok == 3
    assert np.array_equal(steps, true)


def test_full_turning_circle_is_ordered_or_skipped_never_misordered():
    """
    A steady circle that closes on itself: exact order, or skipped (-1) -- never a wrong order.
    """

    r0 = 0.1
    n = int(round(3 * 2 * np.pi / r0 / SURROGATE_DT))         # three full loops
    st, de, ep, true = _shuffled([_episode(n=n, u0=3.0, r0=r0)])
    steps, n_ok, n_bad = step_index(st, de, ep)
    assert (n_ok == 1 and np.array_equal(steps, true)) or (n_bad == 1 and np.all(steps == -1))


def test_closed_circles_at_large_coordinates_are_never_misordered():
    """
    Steady turning circles that close on themselves, far from the origin
    (float32 position rounding ~0.01 m): each episode is ordered exactly or
    skipped -- never accepted in a wrong order. Some are expected to be skipped
    (the ambiguity guard), the others ordered.
    """

    rng = np.random.default_rng(7)
    outcomes = {"ordered": 0, "skipped": 0}
    for _ in range(40):
        r0 = float(rng.uniform(0.03, 0.15))
        n = int(round(2 * np.pi / r0 / SURROGATE_DT)) + int(rng.integers(-2, 3))
        ep = _episode(n=n, u0=float(rng.uniform(1.0, 5.0)), v0=float(rng.normal() * 0.2), r0=r0,
                      x0=1.5e5, y0=2.0e5, h0=float(rng.uniform(-np.pi, np.pi)))
        st, de, eid, true = _shuffled([ep], seed=int(rng.integers(1 << 30)))
        steps, n_ok, n_bad = step_index(st, de, eid)
        if n_ok == 1:
            assert np.array_equal(steps, true)
            outcomes["ordered"] += 1
        else:
            assert n_bad == 1 and np.all(steps == -1)
            outcomes["skipped"] += 1
    assert outcomes["ordered"] > 0


def test_slow_ship_far_from_origin_is_ordered():
    """
    A very slow, slowly turning, decelerating ship far from the origin: a row's
    predicted next state lies close to the row itself, so a row must never be
    taken as its own successor (the k=2 query) -- with that guard the order is exact.
    """

    for seed in range(4):
        st, de, ep, true = _shuffled([_episode(n=200, u0=0.3, r0=0.002, decel=0.02, x0=1.5e5, y0=1.5e5)],
                                     seed=seed)
        steps, n_ok, n_bad = step_index(st, de, ep)
        assert n_ok == 1 and n_bad == 0
        assert np.array_equal(steps, true)


def test_stopped_ship_is_skipped():
    """
    Identical rows (a ship at rest): the order is ambiguous -> skipped, all -1.
    """

    st, de, ep, _ = _shuffled([_episode(n=50, u0=0.0)])
    steps, n_ok, n_bad = step_index(st, de, ep)
    assert n_bad == 1 and n_ok == 0 and np.all(steps == -1)


def test_whole_step_helpers():
    """
    Stored float32 times map back to exact steps; thresholds are robust to float rounding.
    """

    t = (np.arange(5000) * SURROGATE_DT).astype(np.float32)
    assert np.array_equal(steps_from_times(t), np.arange(5000))
    assert steps_from_times(np.array([-1.0], np.float32))[0] == -1
    assert [step_threshold(v) for v in (0.0, 0.7, 0.75, 1.3, 20.0, 32.1)] == [0, 7, 8, 13, 200, 321]
