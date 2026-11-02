"""
Named UAntwerp surrogate presets: ship, scenario, checkpoint and UKC lookup data
of one surrogate slot.

Selected via ua_surrogate_config = {"preset": <name>, ...} (see config.py). The
preset states what is expected; the checkpoint states what the model is (scenario,
ship, rudder scale, env encoding). Both are compared when the surrogate is built,
so a wrong checkpoint fails immediately.

A new ship or scenario variant needs a dataset, a trained checkpoint and one
PRESETS entry; no code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fw.ships.ship_factory import ShipModel
from fw.surrogate.schema import CONFINED_UKC, FULL_ENV, OPEN_WATER, ScenarioSchema

# Base folder for relative lookup-CSV paths (next to the surrogate dynamics module).
LOOKUP_CSV_DIR = Path(__file__).resolve().parent.parent / "dynamics"

# Base folder for relative checkpoint paths (src/fw/checkpoints), independent of
# the working directory.
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


@dataclass(frozen=True)
class SurrogatePreset:
    """
    One surrogate slot.

    Attributes:
        name: Preset name, used as ua_surrogate_config["preset"].
        ship_model: Ship the checkpoint must have been trained for.
        scenario: Scenario the checkpoint must have been trained for.
        checkpoint: Default checkpoint file, relative to CHECKPOINT_DIR
            unless absolute.
        lookup_csv: Default CSV (.csv or .csv.gz) for the nearest-neighbor UKC
            lookup (fw.dynamics.wcu_lookup.WcuLookup), relative to LOOKUP_CSV_DIR
            unless absolute; None = no lookup. Only for scenarios with a UKC input.
        default_ukc: UKC [m] used without a lookup or when it finds nothing.
    """

    name: str
    ship_model: ShipModel
    scenario: ScenarioSchema
    checkpoint: str
    lookup_csv: Optional[str]
    default_ukc: float

    def __post_init__(self):
        """
        Reject a lookup_csv for a scenario without a UKC input (ValueError).
        """

        if self.lookup_csv is not None and not self.scenario.includes_ukc:
            raise ValueError(f"Preset '{self.name}': lookup_csv given, but scenario "
                             f"'{self.scenario.name}' has no UKC input.")


# Fallback UKC [m] for presets with a UKC input (named water depth for historical
# reasons; the models receive it as UKC).
_DEFAULT_WATER_DEPTH_M = 18.5

# Median UKC of the Myzako confined training data (static water level, 2.2-25 m).
# Currently unused: myzako_confined uses _DEFAULT_WATER_DEPTH_M.
_MYZAKO_CONFINED_DEFAULT_UKC_M = 12.6

PRESETS = {p.name: p for p in (
    SurrogatePreset(
        name="tripoli_clean",
        ship_model=ShipModel.TRIPOLI,
        scenario=OPEN_WATER,
        checkpoint="best_surrogate_tripoli_clean.pth",
        lookup_csv=None,
        default_ukc=0.0,        # UKC is not a model input; informational only
    ),
    SurrogatePreset(
        name="tripoli_confined",
        ship_model=ShipModel.TRIPOLI,
        scenario=CONFINED_UKC,
        checkpoint="best_surrogate_tripoli_confined.pth",
        lookup_csv="ukc_lookup_tripoli_confined.csv.gz",
        default_ukc=_DEFAULT_WATER_DEPTH_M,
    ),
    SurrogatePreset(
        name="tripoli_full",
        ship_model=ShipModel.TRIPOLI,
        scenario=FULL_ENV,
        checkpoint="best_surrogate_tripoli_full.pth",
        # One dataset row per 5 m cell (UKC, wind, current and the recording ship's
        # heading/speed; fw.surrogate.make_ukc_lookup). The full dataset CSV also
        # works but loads much slower.
        lookup_csv="env_lookup_tripoli_full.csv.gz",
        default_ukc=_DEFAULT_WATER_DEPTH_M,
    ),
    SurrogatePreset(
        name="myzako_clean",
        ship_model=ShipModel.MYZAKO_37,
        scenario=OPEN_WATER,
        checkpoint="best_surrogate_myzako_clean.pth",
        lookup_csv=None,
        default_ukc=0.0,        # UKC is not a model input; informational only
    ),
    SurrogatePreset(
        name="myzako_confined",
        ship_model=ShipModel.MYZAKO_37,
        scenario=CONFINED_UKC,
        checkpoint="best_surrogate_myzako_confined.pth",
        # Grid-reduced copy of the dataset (make_ukc_lookup); loads much faster.
        lookup_csv="ukc_lookup_myzako_confined.csv.gz",
        default_ukc=_DEFAULT_WATER_DEPTH_M,
    ),
    SurrogatePreset(
        name="myzako_full",
        ship_model=ShipModel.MYZAKO_37,
        scenario=FULL_ENV,
        checkpoint="best_surrogate_myzako_full.pth",
        # As tripoli_full.
        lookup_csv="env_lookup_myzako_full.csv.gz",
        default_ukc=_DEFAULT_WATER_DEPTH_M,
    ),
)}


def get_preset(name: str) -> SurrogatePreset:
    """
    Look up a preset by name.

    Raises:
        ValueError: If there is no such preset.
    """

    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown UAntwerp surrogate preset {name!r}. Available: {sorted(PRESETS)}.") from None


def resolve_checkpoint(path: str) -> Path:
    """
    Absolute path of a checkpoint (relative paths against CHECKPOINT_DIR).
    """

    p = Path(path)
    return p if p.is_absolute() else CHECKPOINT_DIR / p


def resolve_lookup_csv(path: Optional[str]) -> Optional[Path]:
    """
    Absolute path of a lookup CSV (relative paths against LOOKUP_CSV_DIR); None stays None.
    """

    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else LOOKUP_CSV_DIR / p
