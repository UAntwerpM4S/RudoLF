"""Model compatibility error detection and error message quality tests.

Covers R4 (incompatible model file — class hierarchy, immutability, dataclass contract)
and R10 (actionable error messages — concrete dimension values in validation output).

These tests complement test_transfer_learning.py by focusing on the error type
contract and the exact dimension values present in error messages, rather than
repeating the pass/fail routing logic that is already verified there.
"""

import pytest

from airsim.train.transfer_learning import (
    TopologyReport,
    TransferIncompatibilityError,
    validate_transfer_compatibility,
)


def _make_report(**overrides) -> TopologyReport:
    """Return a valid TopologyReport with optional field overrides.

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
    """R4 — TransferIncompatibilityError must be a proper ValueError subclass.

    Verifies the error-type hierarchy so callers that catch ValueError generically
    can still handle transfer failures without importing the specific exception type.
    """

    def test_transfer_incompatibility_error_is_value_error_subclass(self):
        """TransferIncompatibilityError must be a subclass of ValueError at class level.

        Verifies:
            - issubclass(TransferIncompatibilityError, ValueError) is True.
        """
        assert issubclass(TransferIncompatibilityError, ValueError)

    def test_transfer_incompatibility_error_can_be_caught_as_value_error(self):
        """A raised TransferIncompatibilityError must be catchable as ValueError.

        Verifies:
            - except ValueError intercepts the exception without re-raising.
        """
        report = _make_report(output_compatible=False, source_output_dim=3)
        caught = False
        try:
            validate_transfer_compatibility(report)
        except ValueError:
            caught = True
        assert caught, "TransferIncompatibilityError was not caught by except ValueError"


class TestActionMismatchErrorMessageContent:
    """R10 — Action-mismatch error messages must name both concrete output dimensions."""

    def test_action_mismatch_message_names_source_output_dim(self):
        """Error message for action mismatch must contain the source output dim.

        Verifies:
            - When source_output_dim=5 and target_output_dim=2 differ,
              the error message contains '5'.
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
        """Error message for action mismatch must contain the target output dim.

        Verifies:
            - When source_output_dim=5 and target_output_dim=2 differ,
              the error message contains '2'.
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
    """R10 — Target-smaller-than-source error messages must name both input dimensions."""

    def test_target_smaller_message_names_source_input_dim(self):
        """Error message for target-smaller-than-source must contain the source input dim.

        Verifies:
            - When source_input_dim=18 and target_input_dim=10 (extra_input_dims=-8),
              the error message contains '18'.
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
        """Error message for target-smaller-than-source must contain the target input dim.

        Verifies:
            - When source_input_dim=18 and target_input_dim=10 (extra_input_dims=-8),
              the error message contains '10'.
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
    """R4 — TopologyReport must be an immutable dataclass with correct computed fields."""

    def test_topology_report_is_frozen(self):
        """TopologyReport must be immutable — attribute assignment must raise an exception.

        Verifies:
            - Attempting to set a field on a constructed TopologyReport raises
              AttributeError or TypeError (frozen dataclass behavior).
        """
        report = _make_report()
        with pytest.raises((AttributeError, TypeError)):
            # noinspection PyDataclass
            report.source_input_dim = 99  # type: ignore[misc]

    def test_topology_report_extra_input_dims_constructed_directly(self):
        """TopologyReport must store extra_input_dims exactly as supplied to the constructor.

        Verifies:
            - extra_input_dims=8 supplied at construction is available as-is on the instance.
        """
        report = _make_report(source_input_dim=10, target_input_dim=18, extra_input_dims=8)
        assert report.extra_input_dims == 8

    def test_topology_report_output_compatible_true_when_action_dims_match(self):
        """TopologyReport.output_compatible must be True when supplied True and dims match.

        Verifies:
            - Constructing a report with output_compatible=True reflects that value
              on the instance unchanged (the dataclass stores what is supplied).
        """
        report = _make_report(source_output_dim=2, target_output_dim=2, output_compatible=True)
        assert report.output_compatible is True
