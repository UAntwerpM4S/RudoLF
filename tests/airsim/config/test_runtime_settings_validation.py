"""
Runtime settings validation, merge behavior, and config error messages.

Covers R8 (configuration drift — defaults and merge contract) and
R10 (actionable error messages — field names in validation output).
"""

import pytest

from airsim.config.airsim_config import (
    AirSimConfig,
    get_default_runtime_settings,
    merge_runtime_settings,
)


class TestDefaultRuntimeSettingsKeys:
    """
    R8 — built-in defaults must expose all expected top-level sections and keys.
    """

    def test_visualization_section_contains_mode_flags(self):
        """
        Visualization section must expose model_visualise and inference_visualise flags.

        Verifies:
            - 'model_visualise' key is present in the visualization defaults.
            - 'inference_visualise' key is present in the visualization defaults.
        """

        vis = get_default_runtime_settings()["visualization"]
        assert "model_visualise" in vis
        assert "inference_visualise" in vis

    def test_visualization_section_contains_path_keys(self):
        """
        Visualization section must expose model_path and inference_model_path keys.

        Verifies:
            - 'model_path' key is present in the visualization defaults.
            - 'inference_model_path' key is present in the visualization defaults.
        """

        vis = get_default_runtime_settings()["visualization"]
        assert "model_path" in vis
        assert "inference_model_path" in vis

    def test_training_section_contains_stop_condition_keys(self):
        """
        Training section must expose max_steps for episode capping.

        Verifies:
            - 'max_steps' key is present in the training defaults.
        """

        training = get_default_runtime_settings()["training"]
        assert "max_steps" in training

    def test_training_section_contains_speed_profile_key(self):
        """
        Training section must expose speed_profile for timing preset selection.

        Verifies:
            - 'speed_profile' key is present in the training defaults.
        """

        training = get_default_runtime_settings()["training"]
        assert "speed_profile" in training

    def test_airsim_section_present_in_defaults(self):
        """
        Default settings must contain an 'airsim' section with at least port and ip.

        Verifies:
            - 'airsim' section is present.
            - 'port' and 'ip' keys are present within it.
        """

        airsim = get_default_runtime_settings().get("airsim", {})
        assert "port" in airsim
        assert "ip" in airsim


class TestMergeRuntimeSettings:
    """
    R8 — merge_runtime_settings must preserve defaults on partial or unknown overrides.
    """

    def test_unknown_top_level_section_does_not_crash_merge(self):
        """
        Merging an override with an unknown top-level key must not raise.

        Verifies:
            - merge_runtime_settings returns a non-None result when given a dict
              containing a key that is not part of the default sections.
        """

        result = merge_runtime_settings({"custom_section": {"foo": 1}})
        assert result is not None

    def test_partial_visualization_override_keeps_unspecified_defaults(self):
        """
        A partial visualization override must leave untouched keys at their defaults.

        Verifies:
            - Providing only model_visualise=True in the override does not erase
              the default model_path value from the merged result.
        """

        defaults = get_default_runtime_settings()
        default_path = defaults["visualization"]["model_path"]
        result = merge_runtime_settings({"visualization": {"model_visualise": True}})
        assert result["visualization"]["model_path"] == default_path

    def test_explicit_value_wins_over_default(self):
        """
        An explicitly supplied value must override its corresponding default.

        Verifies:
            - episodes=99 in the override results in episodes=99 in the merged output.
        """

        result = merge_runtime_settings({"visualization": {"episodes": 99}})
        assert result["visualization"]["episodes"] == 99

    def test_non_dict_override_returns_pure_defaults(self):
        """
        A non-dict override must return the built-in defaults unchanged.

        Verifies:
            - Passing None as the override returns the same structure as
              get_default_runtime_settings().
        """

        result = merge_runtime_settings(None)
        assert result == get_default_runtime_settings()


class TestAirSimConfigErrorMessages:
    """
    R10 — AirSimConfig validation must produce field-naming, actionable error messages.
    """

    def test_invalid_port_message_names_the_port_field(self):
        """
        A ValueError from an out-of-range port must mention 'port' in the message.

        Verifies:
            - AirSimConfig(port=0) raises ValueError.
            - The error message contains 'Port' or 'port'.
        """

        with pytest.raises(ValueError, match="[Pp]ort"):
            AirSimConfig(port=0)

    def test_invalid_timeout_message_names_the_timeout_field(self):
        """
        A ValueError from a non-positive timeout must mention 'timeout' in the message.

        Verifies:
            - AirSimConfig(timeout=0.0) raises ValueError.
            - The error message contains 'Timeout' or 'timeout'.
        """

        with pytest.raises(ValueError, match="[Tt]imeout"):
            AirSimConfig(timeout=0.0)

    def test_invalid_ip_message_mentions_address(self):
        """
        A ValueError from a bad IP string must mention 'IP' or 'address'.

        Verifies:
            - AirSimConfig(ip='not-an-ip') raises ValueError.
            - The error message contains 'IP' or 'address' to guide the user.
        """

        with pytest.raises(ValueError, match="[Ii][Pp]|[Aa]ddress"):
            AirSimConfig(ip="not-an-ip")
