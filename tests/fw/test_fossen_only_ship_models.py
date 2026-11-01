"""Temporary guard: TRIPOLI_38 / TRIPOLI_45 are rejected by the FH entry points (until the ShipModel refactor)."""

import pytest

from fw.dynamics.fh_tabular_model import FhTabularModel
from fw.ships.ship_factory import FOSSEN_ONLY_SHIP_MODELS, ShipModel, create_ship, require_fh_supported


class _FakeEngine:
    """Minimal stand-in for an FH engine: only exposes ship_model."""

    def __init__(self, ship_model):
        """Store the ship model."""

        self.ship_model = ship_model


@pytest.mark.parametrize("ship_model", sorted(FOSSEN_ONLY_SHIP_MODELS, key=lambda m: m.name))
def test_fossen_only_models_rejected_for_fh(ship_model):
    """require_fh_supported and FhTabularModel raise at construction for Fossen-only presets."""

    with pytest.raises(ValueError, match=ship_model.name):
        require_fh_supported(ship_model, "test")
    with pytest.raises(ValueError, match="FhTabularModel"):
        FhTabularModel(create_ship(ship_model).specifications, _FakeEngine(ship_model))


@pytest.mark.parametrize("ship_model", [m for m in ShipModel if m not in FOSSEN_ONLY_SHIP_MODELS])
def test_other_models_accepted_for_fh(ship_model):
    """All other presets pass the guard unchanged."""

    require_fh_supported(ship_model, "test")
    FhTabularModel(create_ship(ship_model).specifications, _FakeEngine(ship_model))
