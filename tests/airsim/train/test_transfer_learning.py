"""
Unit tests for AirSim transfer learning utilities (T-055 — dimension validation).

Covers:
    - TopologyReport construction and formatting
    - validate_transfer_compatibility: accepts valid configs, rejects mismatches
    - apply_transfer_weights: correct keys transferred, shapes preserved, freeze logic
    - Cross-device weight transfer (CPU source → CUDA target shape parity)
"""

from __future__ import annotations

import pytest
import torch

from fw.policies.ppo_policy import PPOPolicy
from airsim.train.transfer_learning import (
    TopologyReport,
    TransferIncompatibilityError,
    apply_transfer_weights,
    compare_network_topology,
    validate_transfer_compatibility,
    _TRANSFERABLE_KEYS,
    _BACKBONE_KEYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_policy(input_dim: int, output_dim: int = 2, device: str = "cpu") -> PPOPolicy:
    """
    Create a freshly initialised PPOPolicy without touching disk.
    """

    return PPOPolicy(input_dim, output_dim, device=device)


# ---------------------------------------------------------------------------
# T-054 — topology comparison
# ---------------------------------------------------------------------------


class TestCompareNetworkTopology:
    """
    Tests for compare_network_topology — 2D vs 3D input/output dimension reporting.
    """

    def test_extra_dims_computed_correctly(self):
        """
        extra_input_dims must equal target_input_dim minus source_input_dim.
        """

        src = _make_policy(input_dim=10)
        report = compare_network_topology(src, target_input_dim=18, target_output_dim=2)
        assert report.extra_input_dims == 8

    def test_source_dims_captured(self):
        """
        Both source and target input dims must be stored on the report.
        """

        src = _make_policy(input_dim=10)
        report = compare_network_topology(src, target_input_dim=18, target_output_dim=2)
        assert report.source_input_dim == 10
        assert report.target_input_dim == 18

    def test_output_compatible_when_matching(self):
        """
        output_compatible must be True when action dimensions are equal.
        """

        src = _make_policy(input_dim=10, output_dim=2)
        report = compare_network_topology(src, target_input_dim=18, target_output_dim=2)
        assert report.output_compatible is True

    def test_output_incompatible_when_mismatched(self):
        """
        output_compatible must be False when action dimensions differ.
        """

        src = _make_policy(input_dim=10, output_dim=3)
        report = compare_network_topology(src, target_input_dim=18, target_output_dim=2)
        assert report.output_compatible is False

    def test_shared_layer_shapes_include_256_to_128(self):
        """
        The transferable hidden layer (256→128) must appear in shared_layer_shapes.
        """

        src = _make_policy(input_dim=10)
        report = compare_network_topology(src, target_input_dim=18, target_output_dim=2)
        assert (256, 128) in report.shared_layer_shapes

    def test_format_contains_key_labels(self):
        """
        Formatted report must mention mapping layer and both input dimensions.
        """

        src = _make_policy(input_dim=10)
        report = compare_network_topology(src, target_input_dim=18, target_output_dim=2)
        text = report.format()
        assert "mapping layer" in text.lower()
        assert "10" in text
        assert "18" in text


# ---------------------------------------------------------------------------
# T-055 — incompatibility reporting
# ---------------------------------------------------------------------------


class TestValidateTransferCompatibility:
    """
    Tests for validate_transfer_compatibility — error messages and pass/fail logic.
    """

    def test_valid_config_does_not_raise(self):
        """
        A compatible topology report must not raise any exception.
        """

        report = TopologyReport(
            source_input_dim=10,
            target_input_dim=18,
            source_output_dim=2,
            target_output_dim=2,
            extra_input_dims=8,
            shared_layer_shapes=[(256, 128)],
            output_compatible=True,
        )
        validate_transfer_compatibility(report)

    def test_output_mismatch_raises(self):
        """
        Mismatched action dimensions must raise TransferIncompatibilityError.
        """

        report = TopologyReport(
            source_input_dim=10,
            target_input_dim=18,
            source_output_dim=3,
            target_output_dim=2,
            extra_input_dims=8,
            shared_layer_shapes=[(256, 128)],
            output_compatible=False,
        )
        with pytest.raises(TransferIncompatibilityError, match="Action dimension mismatch"):
            validate_transfer_compatibility(report)

    def test_target_smaller_than_source_raises(self):
        """
        Target observation space smaller than source must raise TransferIncompatibilityError.
        """

        report = TopologyReport(
            source_input_dim=18,
            target_input_dim=10,
            source_output_dim=2,
            target_output_dim=2,
            extra_input_dims=-8,
            shared_layer_shapes=[(256, 128)],
            output_compatible=True,
        )
        with pytest.raises(TransferIncompatibilityError, match="smaller than"):
            validate_transfer_compatibility(report)

    def test_both_errors_reported_together(self):
        """
        Both action and observation dimension errors must appear in a single exception.
        """

        report = TopologyReport(
            source_input_dim=18,
            target_input_dim=10,
            source_output_dim=3,
            target_output_dim=2,
            extra_input_dims=-8,
            shared_layer_shapes=[(256, 128)],
            output_compatible=False,
        )
        with pytest.raises(TransferIncompatibilityError) as exc_info:
            validate_transfer_compatibility(report)
        message = str(exc_info.value)
        assert "Action dimension mismatch" in message
        assert "smaller than" in message

    def test_error_message_mentions_source_and_target_dims(self):
        """
        The error message must include the concrete dimension numbers.
        """

        report = TopologyReport(
            source_input_dim=10,
            target_input_dim=18,
            source_output_dim=5,
            target_output_dim=2,
            extra_input_dims=8,
            shared_layer_shapes=[(256, 128)],
            output_compatible=False,
        )
        with pytest.raises(TransferIncompatibilityError) as exc_info:
            validate_transfer_compatibility(report)
        assert "5" in str(exc_info.value)
        assert "2" in str(exc_info.value)

    def test_exact_same_dims_valid(self):
        """
        Zero extra input dims (same-size obs space) must be accepted.
        """

        report = TopologyReport(
            source_input_dim=18,
            target_input_dim=18,
            source_output_dim=2,
            target_output_dim=2,
            extra_input_dims=0,
            shared_layer_shapes=[(256, 128)],
            output_compatible=True,
        )
        validate_transfer_compatibility(report)


# ---------------------------------------------------------------------------
# T-056 — weight loading with layer matching and freezing
# ---------------------------------------------------------------------------


class TestApplyTransferWeights:
    """
    Tests for apply_transfer_weights — key selection, device safety, and freeze logic.
    """

    def test_transferable_keys_are_copied(self):
        """
        Only keys whose shapes match must be copied; skipped keys must not raise.
        """

        src = _make_policy(input_dim=10, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        transferred = apply_transfer_weights(src, tgt, freeze_shared=False)

        src_sd = src.network.state_dict()
        tgt_sd = tgt.network.state_dict()
        # Only verify keys that were actually transferred (shape-matched).
        for key in transferred:
            assert torch.allclose(src_sd[key], tgt_sd[key]), f"Key {key!r} not copied correctly"

    def test_first_layer_not_transferred_when_dims_differ(self):
        """
        When input dims differ the first linear layer values must be unchanged.
        """

        src = _make_policy(input_dim=10, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        tgt_original_w = tgt.network.state_dict()["shared_network.0.weight"].clone()
        apply_transfer_weights(src, tgt, freeze_shared=False)
        tgt_after_w = tgt.network.state_dict()["shared_network.0.weight"]

        assert torch.allclose(tgt_original_w, tgt_after_w), (
            "First layer weights must not be overwritten when input dims differ"
        )

    def test_freeze_disables_grad_only_for_backbone(self):
        """
        Only the backbone (shared_network.2) must be frozen; actor/critic heads stay trainable.
        """

        src = _make_policy(input_dim=10, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        apply_transfer_weights(src, tgt, freeze_shared=True)

        for name, param in tgt.network.named_parameters():
            if name in _BACKBONE_KEYS:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should remain trainable"

    def test_no_freeze_keeps_all_grads_enabled(self):
        """
        All parameters must remain trainable when freeze_shared=False.
        """

        src = _make_policy(input_dim=10, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        apply_transfer_weights(src, tgt, freeze_shared=False)

        for name, param in tgt.network.named_parameters():
            assert param.requires_grad, f"{name} should remain trainable"

    def test_first_layer_remains_trainable_after_freeze(self):
        """
        The mapping-layer first linear must stay trainable even when shared layers are frozen.
        """

        src = _make_policy(input_dim=10, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        apply_transfer_weights(src, tgt, freeze_shared=True)

        first_layer_weight = dict(tgt.network.named_parameters())["shared_network.0.weight"]
        assert first_layer_weight.requires_grad, "Mapping layer must remain trainable"

    def test_returns_list_of_transferred_key_names(self):
        """
        Return value must be a non-empty list of valid state-dict key strings.
        """

        src = _make_policy(input_dim=10, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        transferred = apply_transfer_weights(src, tgt)

        assert len(transferred) > 0
        for key in transferred:
            assert key in _TRANSFERABLE_KEYS

    def test_same_input_dim_transfers_all_keys_including_first_layer(self):
        """
        When input dims match all keys including shared_network.0 must be transferred.
        """

        src = _make_policy(input_dim=18, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        transferred = apply_transfer_weights(src, tgt, freeze_shared=False)

        assert "shared_network.0.weight" in transferred
        assert "shared_network.0.bias" in transferred
        assert "shared_network.2.weight" in transferred
        assert "actor_mean.weight" in transferred
        assert "critic.weight" in transferred

    def test_first_layer_values_transferred_when_same_dim(self):
        """
        First layer weight values must match source when input dimensions are equal.
        """

        src = _make_policy(input_dim=18, output_dim=2)
        tgt = _make_policy(input_dim=18, output_dim=2)

        src_w = src.network.state_dict()["shared_network.0.weight"].clone()
        apply_transfer_weights(src, tgt, freeze_shared=False)
        tgt_w = tgt.network.state_dict()["shared_network.0.weight"]

        assert torch.allclose(src_w, tgt_w), (
            "First layer values must be copied when source and target input dims match"
        )

    def test_output_dim_mismatch_skips_actor_keys(self):
        """
        Actor head keys must not be transferred when output dimensions differ.
        """

        src = _make_policy(input_dim=10, output_dim=3)
        tgt = _make_policy(input_dim=18, output_dim=2)

        transferred = apply_transfer_weights(src, tgt, freeze_shared=False)

        assert "actor_mean.weight" not in transferred
        assert "actor_mean.bias" not in transferred

    def test_actor_log_std_is_not_transferred(self):
        """
        `actor_log_std` must keep the target's fresh init so exploration is preserved.

        A converged 2D policy can collapse log_std to highly negative values for
        some action dimensions, producing near-deterministic actions. Copying
        that into the target freezes exploration and PPO cannot recover. The
        target must retain its default ``log_std = -0.5``.
        """

        src = _make_policy(input_dim=18, output_dim=2)
        src.network.actor_log_std.data.copy_(torch.tensor([-3.3, 0.06]))
        tgt = _make_policy(input_dim=18, output_dim=2)
        tgt_initial = tgt.network.actor_log_std.detach().clone()

        transferred = apply_transfer_weights(src, tgt, freeze_shared=False)

        assert "actor_log_std" not in transferred
        assert torch.allclose(tgt.network.actor_log_std.detach(), tgt_initial)

    def test_cross_device_transfer_preserves_target_device(self):
        """
        Transferred weights must land on the target network device, not the source device.
        """

        src = _make_policy(input_dim=18, output_dim=2, device="cpu")
        tgt = _make_policy(input_dim=18, output_dim=2, device="cpu")

        transferred = apply_transfer_weights(src, tgt, freeze_shared=False)

        tgt_sd = tgt.network.state_dict()
        for key in transferred:
            assert tgt_sd[key].device.type == "cpu", (
                f"{key} ended up on wrong device after transfer"
            )
