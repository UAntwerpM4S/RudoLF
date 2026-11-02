"""
Model compatibility error detection and error message quality tests.

Covers R4 (error type hierarchy, TopologyReport immutability) and R10 (error
messages name the concrete dimensions). The pass/fail routing itself is tested
in test_transfer_learning.py.
"""

import pytest

from airsim.train.transfer_learning import (
    TopologyReport,
    TransferIncompatibilityError,
    validate_transfer_compatibility,
)


def _make_report(**overrides) -> TopologyReport:
    """
    Return a valid TopologyReport with optional field overrides.

    Args:
        overrides: Fields to override from the compatible baseline.

    Returns:
        TopologyReport: Baseline-compatible or intentionally broken report.
    """

    defaults = dict(
        source_input_dim=10,
        target_input_dim=18,
        source_output_dim=2,
        target_output_dim=2,
        extra_input_dims=8,
        shared_layer_shapes=[(256, 128)],
        output_compatible=True,
    )
    defaults.update(overrides)
    return TopologyReport(**defaults)


class TestTransferIncompatibilityErrorContract:
    """
    R4: TransferIncompatibilityError is a ValueError subclass, so generic
    ValueError handlers catch it.
    """

    def test_transfer_incompatibility_error_is_value_error_subclass(self):
        """
        TransferIncompatibilityError subclasses ValueError.
        """

        assert issubclass(TransferIncompatibilityError, ValueError)

    def test_transfer_incompatibility_error_can_be_caught_as_value_error(self):
        """
        A raised TransferIncompatibilityError is caught by `except ValueError`.
        """

        report = _make_report(output_compatible=False, source_output_dim=3)
        caught = False
        try:
            validate_transfer_compatibility(report)
        except ValueError:
            caught = True
        assert caught, "TransferIncompatibilityError was not caught by except ValueError"


class TestActionMismatchErrorMessageContent:
    """
    R10: action-mismatch messages name both output dimensions.
    """

    def test_action_mismatch_message_names_source_output_dim(self):
        """
        Message contains the source output dim (5 vs target 2).
        """

        report = _make_report(
            source_output_dim=5,
            target_output_dim=2,
            output_compatible=False,
        )
        with pytest.raises(TransferIncompatibilityError) as exc_info:
            validate_transfer_compatibility(report)
        assert "5" in str(exc_info.value)

    def test_action_mismatch_message_names_target_output_dim(self):
        """
        Message contains the target output dim (2 vs source 5).
        """

        report = _make_report(
            source_output_dim=5,
            target_output_dim=2,
            output_compatible=False,
        )
        with pytest.raises(TransferIncompatibilityError) as exc_info:
            validate_transfer_compatibility(report)
        assert "2" in str(exc_info.value)


class TestTargetSmallerErrorMessageContent:
    """
    R10: target-smaller-than-source messages name both input dimensions.
    """

    def test_target_smaller_message_names_source_input_dim(self):
        """
        Message contains the source input dim (18 vs target 10).
        """

        report = _make_report(
            source_input_dim=18,
            target_input_dim=10,
            extra_input_dims=-8,
            output_compatible=True,
        )
        with pytest.raises(TransferIncompatibilityError) as exc_info:
            validate_transfer_compatibility(report)
        assert "18" in str(exc_info.value)

    def test_target_smaller_message_names_target_input_dim(self):
        """
        Message contains the target input dim (10 vs source 18).
        """

        report = _make_report(
            source_input_dim=18,
            target_input_dim=10,
            extra_input_dims=-8,
            output_compatible=True,
        )
        with pytest.raises(TransferIncompatibilityError) as exc_info:
            validate_transfer_compatibility(report)
        assert "10" in str(exc_info.value)


class TestTopologyReportDataclassContract:
    """
    R4: TopologyReport is a frozen dataclass that stores what it is given.
    """

    def test_topology_report_is_frozen(self):
        """
        Assigning a field raises AttributeError or TypeError.
        """

        report = _make_report()
        with pytest.raises((AttributeError, TypeError)):
            # noinspection PyDataclass
            report.source_input_dim = 99  # type: ignore[misc]

    def test_topology_report_extra_input_dims_constructed_directly(self):
        """
        extra_input_dims is stored as supplied.
        """

        report = _make_report(source_input_dim=10, target_input_dim=18, extra_input_dims=8)
        assert report.extra_input_dims == 8

    def test_topology_report_output_compatible_true_when_action_dims_match(self):
        """
        output_compatible=True is stored as supplied.
        """

        report = _make_report(source_output_dim=2, target_output_dim=2, output_compatible=True)
        assert report.output_compatible is True
