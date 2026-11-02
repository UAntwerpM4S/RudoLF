"""
fw.surrogate.validation.plot_environment_coverage: the UKC / wind / current
distributions of a dataset CSV.

Must: reconstruct the true wind exactly (checked against the framework's own
apparent_wind_for_state), use the compass and off-the-bow conventions of the
figures, give percentiles within one histogram bin and exact shares, and run
end to end on full and confined CSVs (plain and .gz).
"""

import csv

import numpy as np
import pytest

from fw.dynamics.uantwerp_surrogate import apparent_wind_for_state
from fw.surrogate.schema import get_scenario
from fw.surrogate.validation import plot_environment_coverage as M


def test_true_wind_matches_the_framework_conversion():
    """
    Apparent wind made with apparent_wind_for_state turns back into the true wind it came from.
    """

    rng = np.random.default_rng(0)
    for _ in range(200):
        speed, compass_from = rng.uniform(0.5, 20.0), rng.uniform(0.0, 360.0)
        heading, u, v = rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 6.0), rng.normal(0.0, 0.5)
        # A ship at rest heading North sees the true wind at its compass angle off the bow.
        app_s, app_d = apparent_wind_for_state(speed, np.radians(compass_from), np.pi / 2, 0.0, 0.0,
                                               heading, u, v)
        got_s, got_d = M.true_wind(np.array([app_s]), np.array([app_d]), np.array([heading]),
                                   np.array([u]), np.array([v]))
        assert got_s[0] == pytest.approx(speed, abs=1e-9)
        assert (got_d[0] - compass_from + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-6)


def test_direction_conventions():
    """
    Compass bearings (N = 0, clockwise) and current off the bow (0 = following, 90 = to starboard).
    """

    assert M.compass_from_math(np.radians([0.0, 90.0, 180.0, -90.0])).tolist() == \
        pytest.approx([90.0, 0.0, 270.0, 180.0])
    # Ship heading East (0 rad): current flowing East = following, South = to starboard, North = to port.
    rel = M.current_relative_deg(np.radians([0.0, -90.0, 90.0, 180.0]), np.zeros(4))
    assert rel.tolist() == pytest.approx([0.0, 90.0, 270.0, 180.0])


def test_scalar_statistics():
    """
    Percentiles within one bin, exact moments, min / max and shares below thresholds.
    """

    rng = np.random.default_rng(1)
    values = rng.gamma(2.0, 2.0, 50_000)
    s = M.Scalar("x", "m", M.UKC_EDGES, (1.0, 5.0))
    for part in np.array_split(values, 7):
        s.add(np.append(part, np.nan))                      # NaN is ignored
    assert s.n == len(values)
    assert s.mean == pytest.approx(values.mean()) and s.std == pytest.approx(values.std())
    assert (s.lo, s.hi) == (values.min(), values.max())
    for q in M.PERCENTILES:
        assert abs(s.percentile(q) - np.percentile(values, q)) <= 0.0101
    assert s.share_below(1.0) == pytest.approx(100.0 * (values < 1.0).mean())
    assert s.share_below(5.0) == pytest.approx(100.0 * (values < 5.0).mean())


def test_direction_sectors_are_centred():
    """
    Rose sectors are centred on the compass points: 355 deg and 5 deg both count as N.
    """

    d = M.Direction("d", "test")
    d.add(np.array([355.0, 5.0, 89.0, -90.0]))              # -90 wraps to 270 (W)
    centres, shares = d.sector_shares(30.0)
    total = dict(zip(centres.tolist(), shares.sum(axis=1).tolist()))
    assert total[0.0] == pytest.approx(50.0)
    assert total[90.0] == pytest.approx(25.0) and total[270.0] == pytest.approx(25.0)


def _write_full_csv(path, rng, n_episodes=40, rows=50):
    """
    Shuffled full_env CSV: per episode one true wind (from compass 270 +- 20) and current (toward East).
    """

    sc = get_scenario("full_env")
    data = np.zeros((n_episodes * rows, sc.raw_input_dim + 4))
    i = 0
    for e in range(n_episodes):
        speed, compass_from = rng.uniform(2.0, 12.0), rng.normal(270.0, 20.0)
        for _ in range(rows):
            h, u, v = rng.uniform(-np.pi, np.pi), rng.uniform(0.0, 5.0), rng.normal(0.0, 0.2)
            app_s, app_d = apparent_wind_for_state(speed, np.radians(compass_from), np.pi / 2, 0.0, 0.0, h, u, v)
            row = data[i]
            row[[sc.col("heading"), sc.col("surge"), sc.col("sway")]] = h, u, v
            row[sc.col("ukc")] = rng.uniform(2.0, 8.0)
            row[[sc.col("wind_speed"), sc.col("wind_direction")]] = app_s, app_d
            row[[sc.col("current_speed"), sc.col("current_direction")]] = 0.8, 0.0
            row[-1] = e
            i += 1
    np.savetxt(path, data[rng.permutation(len(data))], delimiter=",", fmt="%.8g")


def _read(path):
    """
    CSV file as a list of dicts.
    """

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_full_dataset_end_to_end(tmp_path):
    """
    Tables and figures for a full CSV; the wind rose and current direction land where they should.
    """

    pytest.importorskip("matplotlib")
    path = tmp_path / "full.csv"
    _write_full_csv(path, np.random.default_rng(2))
    assert M.main([str(path), "--chunk-rows", "300", "--dpi", "40"]) == 0
    out = tmp_path / "full_coverage"
    for name in ("overview.png", "ukc.png", "wind_rose_true.png", "current_relative.png",
                 "summary.csv", "thresholds.csv", "directions.csv"):
        assert (out / name).is_file(), name
    summary = {r["quantity"]: r for r in _read(out / "summary.csv")}
    assert int(summary["UKC"]["rows"]) == 2000
    assert 2.0 <= float(summary["UKC"]["min"]) and float(summary["UKC"]["max"]) <= 8.0
    assert float(summary["current speed"]["p50"]) == pytest.approx(0.8, abs=0.0011)
    directions = {r["quantity"]: r for r in _read(out / "directions.csv")}
    wind = directions["true wind direction"]
    assert float(wind["270"]) > 50.0                       # mostly from the West
    assert float(directions["current direction"]["090"]) == pytest.approx(100.0)   # flows toward East


def test_confined_gz_and_open_water(tmp_path):
    """
    A confined .csv.gz gives the UKC only; open-water data is refused.
    """

    sc = get_scenario("confined_ukc")
    data = np.zeros((500, sc.raw_input_dim + 4))
    data[:, sc.col("ukc")] = np.linspace(1.0, 6.0, 500)
    data[:, -1] = np.arange(500) // 50
    np.savetxt(tmp_path / "conf.csv.gz", data, delimiter=",", fmt="%.6g")
    assert M.main([str(tmp_path / "conf.csv.gz"), "--no-figures"]) == 0
    out = tmp_path / "conf_coverage"
    assert [r["quantity"] for r in _read(out / "summary.csv")] == ["UKC"]
    assert not (out / "directions.csv").exists()

    ow = get_scenario("open_water")
    np.savetxt(tmp_path / "ow.csv", np.zeros((10, ow.raw_input_dim + 4)), delimiter=",")
    with pytest.raises(ValueError, match="no UKC"):
        M.main([str(tmp_path / "ow.csv"), "--no-figures"])


def test_compare_slide_from_two_coverage_folders(tmp_path):
    """
    compare_environment_coverage draws one slide from the tables of two coverage folders.
    """

    pytest.importorskip("matplotlib")
    from fw.surrogate.validation import compare_environment_coverage as C

    rng = np.random.default_rng(3)
    for name in ("a", "b"):
        _write_full_csv(tmp_path / f"{name}.csv", rng, n_episodes=10, rows=20)
        assert M.main([str(tmp_path / f"{name}.csv"), "--no-figures"]) == 0
    out = tmp_path / "cmp.png"
    assert C.main([str(tmp_path / "a_coverage"), str(tmp_path / "b_coverage"),
                   "--labels", "A", "B", "--out", str(out), "--dpi", "40"]) == 0
    assert out.is_file() and out.stat().st_size > 0
