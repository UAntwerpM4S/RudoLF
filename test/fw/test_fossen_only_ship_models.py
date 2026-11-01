"""Temporary guard: TRIPOLI_38 / TRIPOLI_45 (and the model-scale B01) are rejected by the FH entry points (until the ShipModel refactor)."""

import pytest

from fw.dynamics.fh_tabular_model import FhTabularModel
from fw.ships.ship_factory import (FH_UNSUPPORTED_SHIP_MODELS, FOSSEN_ONLY_SHIP_MODELS, MODEL_SCALE_SHIP_MODELS,
                                   ShipModel, create_ship, require_fh_supported)


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


@pytest.mark.parametrize("ship_model", sorted(MODEL_SCALE_SHIP_MODELS, key=lambda m: m.name))
def test_model_scale_models_rejected_for_fh(ship_model):
    """Model-scale presets (B01) have no FH ship model: refused at construction."""

    with pytest.raises(ValueError, match="model-scale"):
        require_fh_supported(ship_model, "test")
    with pytest.raises(ValueError, match="FhTabularModel"):
        FhTabularModel(create_ship(ship_model).specifications, _FakeEngine(ship_model))


@pytest.mark.parametrize("ship_model", [m for m in ShipModel if m not in FH_UNSUPPORTED_SHIP_MODELS])
def test_other_models_accepted_for_fh(ship_model):
    """All other presets pass the guard unchanged."""

    require_fh_supported(ship_model, "test")
    FhTabularModel(create_ship(ship_model).specifications, _FakeEngine(ship_model))
