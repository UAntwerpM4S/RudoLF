"""
fw.surrogate.make_ukc_lookup: grid-reduced lookups.

confined_ukc: one point per cell (mean position, median UKC).
full_env: one REAL row per cell (the one closest to the cell's mean position),
only the columns the lookup reads kept, the others 0; .gz output deterministic.
"""

import gzip

import numpy as np

from fw.surrogate import make_ukc_lookup as M
from fw.surrogate.schema import get_scenario

CELL = 5.0


def _write_csv(path, scenario_name, seed=0):
    """
    Write a dataset-layout CSV (raw row, 3 targets, episode_id) with many rows
    per 5 m cell and per-row varying UKC/wind/current, so a wrong median or row
    choice shows. Returns the array as written.
    """

    sc = get_scenario(scenario_name)
    rng = np.random.default_rng(seed)
    n = 600
    rows = np.zeros((n, sc.raw_input_dim + 4))
    rows[:, sc.col("x")] = rng.uniform(0.0, 20.0, n)           # 4 x 2 cells, ~75 rows each
    rows[:, sc.col("y")] = rng.uniform(0.0, 10.0, n)
    rows[:, sc.col("heading")] = rng.uniform(-np.pi, np.pi, n)
    rows[:, sc.col("surge")] = rng.uniform(0.0, 5.0, n)
    rows[:, sc.col("ukc")] = rng.uniform(2.0, 20.0, n)
    if sc.includes_wind_current:
        for c in ("wind_speed", "current_speed"):
            rows[:, sc.col(c)] = rng.uniform(0.0, 10.0, n)
        for c in ("wind_direction", "current_direction"):
            rows[:, sc.col(c)] = rng.uniform(-np.pi, np.pi, n)
    rows[:, -1] = np.arange(n) // 20                              # episodes of 20 rows
    np.savetxt(path, rows, delimiter=",", fmt="%.8g")
    return np.loadtxt(path, delimiter=",")


def _cells(rows, sc):
    """
    Row indices per occupied CELL x CELL square.
    """

    cells = {}
    for i, (x, y) in enumerate(zip(rows[:, sc.col("x")], rows[:, sc.col("y")])):
        cells.setdefault((int(np.floor(x / CELL)), int(np.floor(y / CELL))), []).append(i)
    return cells


def _cell_of(row, sc):
    """
    The cell key of an output row.
    """

    return int(np.floor(row[sc.col("x")] / CELL)), int(np.floor(row[sc.col("y")] / CELL))


def test_confined_grid_median(tmp_path):
    """
    Confined: per occupied cell the mean position and the MEDIAN UKC of its rows.
    """

    sc = get_scenario("confined_ukc")
    rows = _write_csv(tmp_path / "c.csv", "confined_ukc")
    assert M.main(["--csv", str(tmp_path / "c.csv"), "--cell", str(CELL), "--holdout-frac", "0",
                   "--out", str(tmp_path / "c_lk.csv")]) == 0
    out = np.loadtxt(tmp_path / "c_lk.csv", delimiter=",", ndmin=2)
    cells = _cells(rows, sc)
    assert len(out) == len(cells)
    for r in out:
        idx = cells[_cell_of(r, sc)]
        assert abs(r[sc.col("ukc")] - np.median(rows[idx, sc.col("ukc")])) < 1e-3
        assert abs(r[sc.col("x")] - rows[idx, sc.col("x")].mean()) < 1e-2
        assert abs(r[sc.col("y")] - rows[idx, sc.col("y")].mean()) < 1e-2


def test_full_env_keeps_the_real_row_closest_to_the_cell_mean(tmp_path):
    """
    Full: per cell the input row closest to the cell's mean position (lookup columns), the rest 0; .gz stable.
    """

    sc = get_scenario("full_env")
    rows = _write_csv(tmp_path / "f.csv", "full_env")
    for name in ("f1.csv.gz", "f2.csv.gz"):
        assert M.main(["--csv", str(tmp_path / "f.csv"), "--cell", str(CELL), "--holdout-frac", "0",
                       "--out", str(tmp_path / name)]) == 0
    assert (tmp_path / "f1.csv.gz").read_bytes() == (tmp_path / "f2.csv.gz").read_bytes()
    with gzip.open(tmp_path / "f1.csv.gz", "rt") as f:
        out = np.loadtxt(f, delimiter=",", ndmin=2)
    keep = [sc.col(c) for c in M.ENV_LOOKUP_COLUMNS]
    other = [c for c in range(sc.raw_input_dim + 3) if c not in keep]
    assert np.all(out[:, other] == 0.0)
    cells = _cells(rows, sc)
    assert len(out) == len(cells)
    for r in out:
        idx = np.array(cells[_cell_of(r, sc)])
        cx, cy = rows[idx, sc.col("x")].mean(), rows[idx, sc.col("y")].mean()
        d2 = (rows[idx, sc.col("x")] - cx) ** 2 + (rows[idx, sc.col("y")] - cy) ** 2
        expected = rows[idx[np.argmin(d2)]]
        assert np.allclose(r[keep], expected[keep], rtol=1e-5, atol=1e-3)
