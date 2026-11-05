"""
Dataset provenance: the generator records how a dataset was made (in every
.npz and next to the CSV); training logs it and stores it in the checkpoint.
Informational only: no effect on the data or the model.
"""

import torch
import logging
import argparse
import numpy as np

from fw.surrogate.dataset import read_provenance
from surrogate_test_helpers import identity, synthetic_rows, train_tiny


def test_split_and_save_writes_provenance(tmp_path):
    """
    split_and_save stores the provenance in all three splits; the data itself is unchanged.
    """

    from fw.surrogate.generate_ship_data import split_and_save

    x, y, ep = synthetic_rows("confined_ukc", n_episodes=10, steps=20, seed=1)
    prov = {"seed": 1, "cli_settings": {"thrust_range": [0.0, 1.0]}}
    split_and_save(x, y, ep, str(tmp_path / "with"), identity("confined_ukc"), seed=1, provenance=prov)
    split_and_save(x, y, ep, str(tmp_path / "without"), identity("confined_ukc"), seed=1)
    for name in ("train", "val", "test"):
        assert read_provenance(tmp_path / f"with_{name}.npz") == prov
        assert read_provenance(tmp_path / f"without_{name}.npz") is None
        with np.load(tmp_path / f"with_{name}.npz") as a, np.load(tmp_path / f"without_{name}.npz") as b:
            for key in b.files:                                  # the data itself is identical
                assert np.array_equal(a[key], b[key])


def test_generation_provenance_contents():
    """
    generation_provenance records the CLI settings, ship, exercises and generator data version.
    """

    from fw.ships.ship_factory import ShipModel
    from fw.surrogate.generate_ship_data import generation_provenance
    from fw.surrogate.schema import provenance_metadata

    args = argparse.Namespace(samples=10, horizon=80.0, save="data/x.csv", workers=4, split=True,
                              bank_fraction=0.3, bank_band=[11.0, 40.0], straight_fraction=0.25,
                              bank_straight_fraction=0.15, bank_straight_band=[11.0, 40.0],
                              thrust_range=[0.0, 1.0], ship_model="TRIPOLI", configuration="full", seed=1)
    prov = generation_provenance(args, ShipModel.TRIPOLI, 35.0, n_episodes=9, n_rows=700)
    assert prov["ship_model"] == "TRIPOLI" and prov["configuration"] == "full"
    assert prov["cli_settings"]["thrust_range"] == [0.0, 1.0]
    assert prov["cli_settings"]["bank_straight_fraction"] == 0.15
    assert prov["rows"] == 700 and prov["episodes_with_rows"] == 9
    assert prov["exercise"].endswith(".tab")
    provenance_metadata(prov)                                    # JSON-serializable


def test_training_logs_and_stores_provenance(tmp_path, caplog):
    """
    Training stores the provenance of every file in the checkpoint and logs differing settings.
    """

    prov = {"created_utc": "x", "ship_model": "MYZAKO_37", "configuration": "confined",
            "cli_settings": {"seed": 1, "bank_straight_fraction": 0.0}}
    with caplog.at_level(logging.INFO):
        ckpt, (train, val, test) = train_tiny(tmp_path, "confined_ukc", provenance=prov)
    stored = torch.load(ckpt, weights_only=False)["data_provenance"]
    assert stored == {str(train): prov, str(val): prov, str(test): prov}
    assert any("generated" in r.getMessage() for r in caplog.records)


def test_combining_datasets_lists_differing_settings(tmp_path, caplog):
    """
    Two training datasets generated with different settings: training runs and names the differences.
    """

    from fw.surrogate import train_surrogate
    from surrogate_test_helpers import write_splits

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pa = {"configuration": "confined", "cli_settings": {"seed": 1, "bank_straight_fraction": 0.0}}
    pb = {"configuration": "confined", "cli_settings": {"seed": 2, "bank_straight_fraction": 1.0}}
    sa = write_splits(a, "confined_ukc", seed=1, provenance=pa)
    sb = write_splits(b, "confined_ukc", seed=2, provenance=pb)
    with caplog.at_level(logging.INFO):
        train_surrogate.main(["--train-file", str(sa[0]), str(sb[0]), "--val-file", str(sa[1]), str(sb[1]),
                              "--test-file", str(sa[2]), str(sb[2]), "--epochs", "1", "--device", "cpu",
                              "--checkpoint", str(tmp_path / "m.pth")])
    msg = [r.getMessage() for r in caplog.records if "different settings" in r.getMessage()]
    assert msg and "bank_straight_fraction" in msg[0] and "seed" in msg[0]


def test_training_survives_malformed_provenance(tmp_path, caplog):
    """
    Provenance that is not a JSON object (or cli_settings that is not one) never stops training.
    """

    from fw.surrogate import train_surrogate
    from fw.surrogate.schema import NPZ_PROVENANCE_KEY
    from surrogate_test_helpers import write_splits

    train, val, test = write_splits(tmp_path, "confined_ukc", seed=3, provenance={"cli_settings": None})
    for path, raw in ((val, "[1, 2]"), (test, "{not json")):
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays[NPZ_PROVENANCE_KEY] = np.array(raw)
        np.savez(path, **arrays)
    with caplog.at_level(logging.WARNING):
        train_surrogate.main(["--train-file", str(train), "--val-file", str(val), "--test-file", str(test),
                              "--epochs", "1", "--device", "cpu", "--checkpoint", str(tmp_path / "m.pth")])
    assert (tmp_path / "m.pth").is_file()
    assert any("provenance" in r.getMessage() for r in caplog.records)
