from copy import deepcopy
from enum import Enum, auto

from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.configuration.profile_applier import apply_profile
from fw.configuration.config_profile import ProfileRegistry
from fw.common.default_positions import (
    DEFAULT_FRAMEWORK_SHIP_POS,
    DEFAULT_FRAMEWORK_TARGET_POS,
)


class VisualizationMode(Enum):
    """
    Visualization modes for rendering simulation results.

    Attributes:
        SIMPLE: Vessel trajectory only.
        COMPLEX: Trajectory with diagnostics (cross-track error, rudder input,
            UKC, and performance metrics).
    """

    SIMPLE = auto()
    COMPLEX = auto()


class PolicyStrategy(Enum):
    """
    Defines strategies for handling policies during environment switching.

    Attributes:
        RESET_POLICY: Start with a fresh policy.
        REUSE_CURRENT_POLICY: Continue training with the current policy.
        REUSE_OTHER_POLICY: Load and use a policy trained on another environment.
    """

    RESET_POLICY = auto()
    REUSE_CURRENT_POLICY = auto()
    REUSE_OTHER_POLICY = auto()


PPO_POLICY_NAME = "PPO"
PPO2_POLICY_NAME = "PPO2"
MULTI_PPO_POLICY_NAME = "MPPO"
MULTI_POLICY_SINGLE_NAME = "MPSA"  # Multi-Policy Single Agent

AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE = "{session_type}_session.csv"
AIRSIM_BRIDGE_LOG_FILENAME = "airsim_telemetry.csv"
AIRSIM_METADATA_FILENAME = "metadata.json"
AIRSIM_INFERENCE_TRAJECTORY_PLOT_FILENAME = "inference_trajectory.png"
AIRSIM_COMPARISON_TRAJECTORY_PLOT_FILENAME = "trajectory_2d_vs_3d.png"


"""
Following (hyper)parameters are for fine-tuning the agent. The key of the dictionary
that they are linked to, is the name of the environment that they're associated with.
In the long term this should be moved to a config-file which will then be read.
"""

DEFAULT_HYPERPARAMETERS = {
    PPO_POLICY_NAME: {
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.05,
        'eval_frequency': 100,
        'num_envs': 1,
        'num_epochs': 10,
        'normalize': False,
        'max_nbr_iterations': 200000,
        'batch_size': 64,
        'device': "cpu",
        'model_dir': "models"
    },
    PPO2_POLICY_NAME: {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.2,
        'eval_frequency': 10000,
        'num_envs': 1,
        'total_time_steps': 200000,
        'log_interval': 10,
        'n_steps': 2048,
        'batch_size': 64,
        'num_epochs': 10,
        'reward_threshold': 100000.8,
        'device': "cpu",
        'model_dir': "models",
        'log_dir': "logs",
        'tensorboard_log': "logs/tensorboard",
        'verbose': True
    },
    MULTI_PPO_POLICY_NAME: {
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.05,
        'eval_frequency': 100,
        'num_envs': 1,
        'num_epochs': 10,
        'normalize': False,
        'max_nbr_iterations': 200000,
        'batch_size': 64,
        'device': "cpu",
        'model_dir': "models"
    },
    MULTI_POLICY_SINGLE_NAME: {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.025,
        'value_loss_coef': 0.1,
        'max_grad_norm': 0.5,
        'clip_range': 0.2,
        'eval_frequency': 10,
        'num_envs': 12,
        'num_epochs': 8,
        'normalize': False,
        'max_nbr_iterations': 5000,
        'batch_size': 2048,
        'device': "cpu",
        'model_dir': "models",
        'steps_per_policy_update': 8192,
    },
}

AIRSIM_HYPERPARAMETER_OVERRIDES = {
    PPO_POLICY_NAME: {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_range': 0.2,
        'device': 'cuda', # AirSim training benefits from GPU acceleration, can be switched to 'cpu' if CUDA is not available or not applicable.
    },
}

CONFIG_PARAMETERS = {
    # Valid env_type values: "PySimEnv", "FhSimEnv", "MultiSimEnv",
    # "MultiPolicyColavEnv", "AirSimEnv", "LunarLander"
    # Can also be a list of environment names, e.g. ["PySimEnv", "FhSimEnv"],
    # to train or visualize across multiple environments simultaneously.
    'env_type': ["PySimEnv"],
    'profile': "default_profile",       # Possible options: "default_profile",
                                        # "myzako_pysim_to_fhsim_transfer_profile",
                                        # "tripoli_pysim_to_fhsim_transfer_profile",
                                        # "ugent_surrogate_training_profile", ...
    'action': "path_following",  # Valid action values: "train", "visualize", "evaluate_reward", "path_following"
    'log_level': "INFO",  # Valid: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    'wind': False,
    'current': False,
    'enable_ukc_check': True,
    'enable_bank_collision_check': True,
    'visualize_best_trained': False,
    'viz_mode': VisualizationMode.COMPLEX,
    'start_pos': DEFAULT_FRAMEWORK_SHIP_POS,    # (2245.0, -47.0)
    'target_pos': DEFAULT_FRAMEWORK_TARGET_POS, # (-1559.0, 11990.0)
    'overall_contour': "env_Sche_no_scale.csv",
    'depth_contour': "env_Sche_250cm_no_scale.csv",
    'trajectory': "trajectory_points_no_scale.csv",
    'scenario': "DDShip_scenario1_nowindnocurrent",
    'ua_surrogate_config': {"wcu_enabled": False,
                            "file_name": "checkpoints/best_surrogate_clean.pth"},
    'ship_model': ShipModel.MYZAKO_37,
    'dynamics_model': Model.FOSSEN,
    'live_animation': True,

    # Classical/nonlinear path-following baseline settings used by
    # fw.control.path_following when action == "path_following".
    #
    # controller options:
    #   "pid"                    -> classical LOS guidance + heading PID/PD
    #   "feedback_linearization" -> uses the configured simulation model to
    #                                locally invert yaw acceleration dynamics
    #   "sliding_mode"           -> Norrbin-based sliding-mode heading control
    #   "backstepping_dynamic_smc"
    #                             -> path-error backstepping with dynamic SMC
    #
    # guidance options:
    #   "los"          -> line-of-sight guidance based on cross-track error
    #   "vector_field" -> time-varying vector field guidance:
    #                     chi_d = gamma_p - sgn(y_e) atan((|y_e|/Delta)^theta)
    'path_following_controller': "backstepping_dynamic_smc",
    'path_following_guidance': "vector_field",
    # Reduce dense route CSV points before guidance/controller simulation.
    # Ramer-Douglas-Peucker keeps the simplified polyline within the specified
    # perpendicular tolerance from the original route while connecting retained
    # points with straight segments. Increase tolerance to use fewer waypoints.
    'path_following_simplify_path': True,
    'path_following_simplification_tolerance_m': 50.0,
    'path_following_guidance_parameters': {
        "los": {
            "lookahead_distance": None,             # None -> multiplier * ship length
            "lookahead_ship_length_multiplier": 2.0,
            "switch_distance": 65.0,
        },
        "vector_field": {
            "delta": 150.0,                          # None -> effective lookahead distance
            "theta_base": 0.8,
            "theta_gain": 0.16,
            "theta_error_scale": 150,              # None -> delta
            "theta_max": 2.0,
            "switch_distance": 55.0,
        },
    },
    'path_following_surge_speed_control': {
        "target_speed": 4.0,
        "kp": 0.5,
        "ki": 0.01,
        "kd": 0.0,
        "integral_limit": 20.0,
        "feedforward_thrust": None,
        "adaptive_turn_slowdown_enabled": False,
        "min_turn_speed": 3.3,
        "turn_slowdown_heading_error_deg": 15.0,
        "turn_slowdown_max_reduction": 0.20,
    },
    'path_following_output_limits': {
        "rudder_angle_limit_deg": 70,              # None -> ship specification
        "min_thrust": 0.0,
        "max_thrust": None,                          # None -> ship specification
    },
    'path_following_rudder_response': {
        # A command-shaping rudder response is applied before the command is
        # sent to ShipSimulator.  Small command changes are filtered strongly
        # to remove chatter, while large steering commands are allowed through
        # with a faster response so path tracking is not delayed excessively.
        "enabled": True,
        "time_constant_s": 0.15,
        "small_signal_time_constant_s": 1.0,
        "large_signal_threshold_deg": 1.5,

        # Additional rate limiting.  None means use the ship specification.
        "max_rate_deg_s": None,

        # Ignore tiny command changes after filtering.  This avoids numerical
        # jitter when the controller is already close to zero rudder.
        "deadband_deg": 0.1,

        # If both the raw controller command and the filtered rudder angle are
        # very close to zero, snap the rudder back to exactly neutral.  This is
        # different from deadband_deg: deadband_deg suppresses tiny increments,
        # while neutral_deadband_deg removes residual nonzero steering after the
        # vessel has converged to the path.
        "neutral_deadband_deg": 1,
    },
    'path_following_controller_parameters': {
        "pid": {
            "heading_kp": 0.9,
            "heading_ki": 0.0,
            "heading_kd": 0.0,
            "yaw_rate_damping": 10.0,
            "nomoto_feedforward_enabled": True,
            "nomoto_feedforward_gain": 0.05,
            "desired_yaw_time_constant": 10.0,
            "desired_heading_rate_limit_deg_s": 5.0,
            "heading_reference_filter_alpha": 0.6,
            "integral_limit": 1.0,
        },
        "feedback_linearization": {
            # "norrbin" follows the feedback-linearization structure in
            # Fossen and Paulsen's ship-steering paper:
            #     delta = m_hat*a_psi + d1_hat*r + d3_hat*r^3
            # Use "numerical_model_inverse" to recover the previous local
            # finite-difference inversion of the active simulator model.
            "mode": "norrbin",

            # Pole-placement tuning.  If lambda is positive, the controller
            # uses the paper's critically damped gains:
            #     heading_kp = lambda^2, heading_kd = 2*lambda
            # for non-integral control, or the integral version when
            # use_integral_action is enabled.
            "lambda": 0.07,
            "heading_kp": None,
            "heading_kd": None,
            "use_integral_action": False,
            "integral_limit": 1.0,

            # Smooth the guidance heading before differentiating it.  Feedback
            # linearization needs psi_d_dot and psi_d_ddot, so feeding a
            # discontinuous LOS waypoint switch directly into the controller
            # produces large rudder spikes.
            "desired_heading_rate_limit_deg_s": 5.0,
            "heading_reference_filter_alpha": 0.50,
            "heading_error_deadband_deg": 0.05,
            "yaw_rate_error_deadband_deg_s": 0.010,

            # Parameter estimates for the Norrbin steering equation.  Leave
            # these as None to initialize them from the selected simulation
            # model at the first control step.
            "use_model_estimates": True,
            "m_hat": None,
            "d1_hat": None,
            "d3_hat": 0.0,
            "min_abs_m_hat": 1e-3,
            "max_abs_m_hat": 50000.0,
            "max_abs_d1_hat": 50000.0,
            "max_abs_d3_hat": 50000.0,

            # Optional direct adaptive update from the paper.  It is disabled
            # by default because aggressive adaptation can destabilize the
            # path-following run when LOS switches segments frequently.
            "adaptive_enabled": False,
            "gamma_m": 0.0,
            "gamma_d1": 0.0,
            "gamma_d3": 0.0,
            "min_adaptation_speed": 0.5,

            # Numerical safeguards used both by model initialization and by
            # the optional previous finite-difference inversion mode.
            "model_gain_epsilon": 1e-6,
            "model_probe_rudder_deg": 1.0,
        },
        "sliding_mode": {
            # Norrbin-based second-order sliding-mode steering controller:
            #   m*psi_ddot + d1*psi_dot + d3*psi_dot^3 = delta
            #   s = (psi_dot - psi_d_dot) + lambda*(psi - psi_d)
            #       + integral_gain * integral(psi - psi_d)
            # The integral channel is intentionally weak.  It removes slow
            # residual heading bias without turning the SMC into an aggressive
            # rudder-saturating integral controller.
            "lambda": 0.30,
            "eta": 0.00008,
            "boundary_layer": 0.02,
            "heading_error_acceleration_gain": 0.00018,  # [1/s^2], direct large-error convergence term
            "large_error_boost_gain": 0.00012,      # [rad/s^2], extra bounded acceleration for large errors
            "large_error_boost_width_deg": 15.0,    # [deg], error scale where tanh boost becomes active
            "direct_heading_rudder_gain_deg": 12.0,  # [deg], bounded direct rudder feedback for heading error
            "direct_heading_rudder_width_deg": 10.0,
            "direct_heading_integral_gain_deg_per_rad_s": 0.45,
            "direct_heading_integral_limit_rad_s": 20.0,
            "direct_heading_integral_leak_time_s": 60.0,
            "integral_gain": 0.00004,              # [1/s^2], multiplies accumulated heading error [rad*s]
            "integral_limit": 0.20,                # [rad*s], anti-windup clamp for heading-error integral
            "desired_heading_rate_limit_deg_s": 6.0,
            "heading_reference_filter_alpha": 0.45,
            "desired_acceleration_limit_deg_s2": 0.35,
            "use_raw_heading_for_error": False,
            "use_model_estimates": True,
            "update_model_estimates_each_step": True,
            "m_hat": None,
            "d1_hat": None,
            "d3_hat": 0.0,
            "min_abs_m_hat": 1e-3,
            "max_abs_m_hat": 50000.0,
            "max_abs_d1_hat": 50000.0,
            "max_abs_d3_hat": 50000.0,
            "model_gain_epsilon": 1e-6,
            "model_probe_rudder_deg": 1.0,
        },
        "backstepping_dynamic_smc": {
            # Path-error controller:
            #   1. Compensate sideslip beta = atan2(v, u) in the desired heading.
            #   2. Use cross-track error in the sliding surface.
            #   3. Invert the local Norrbin yaw model to obtain rudder.
            #
            # The current tuned default keeps use_path_tangent_input=False and
            # virtual_course_from_cross_track=False.  In that mode, the selected
            # guidance law still generates the desired course/heading, while
            # this controller acts as a robust dynamic heading controller with
            # optional sideslip compensation.  Setting both flags to True turns
            # on the more explicit backstepping outer loop:
            #     chi_v = chi_path + atan2(-cross_track_error, virtual_lookahead)
            # That form is useful for experiments, but it can be more sensitive
            # to sharp route geometry and waypoint switching.
            "lambda": 0.30,
            "eta": 0.00010,
            "boundary_layer": 0.025,
            "use_path_tangent_input": False,
            "virtual_course_from_cross_track": False,
            "virtual_lookahead_m": 150.0,
            "cross_track_gain": 0.0,
            "cross_track_scale_m": 80.0,
            "cross_track_term_rate_limit": 0.01,
            "cross_track_rudder_gain_deg": 0.0,
            "sideslip_compensation_enabled": True,
            "sideslip_compensation_gain": 0.4,
            "sideslip_min_speed": 0.5,
            "sideslip_limit_deg": 12.0,
            "heading_error_acceleration_gain": 0.00018,
            "large_error_boost_gain": 0.00008,
            "large_error_boost_width_deg": 15.0,
            "direct_heading_rudder_gain_deg": 9.0,
            "direct_heading_rudder_width_deg": 10.0,
            "direct_heading_integral_gain_deg_per_rad_s": 0.35,
            "direct_heading_integral_limit_rad_s": 15.0,
            "direct_heading_integral_leak_time_s": 45.0,
            "integral_gain": 0.00003,
            "integral_limit": 0.20,
            "desired_heading_rate_limit_deg_s": 6.0,
            "heading_reference_filter_alpha": 0.15,
            "desired_acceleration_limit_deg_s2": 0.03,
            "use_raw_heading_for_error": True,
            "use_model_estimates": True,
            "update_model_estimates_each_step": True,
            "use_damping_compensation": True,
            "m_hat": None,
            "d1_hat": None,
            "d3_hat": 0.0,
            "min_abs_m_hat": 1e-3,
            "max_abs_m_hat": 50000.0,
            "max_abs_d1_hat": 50000.0,
            "max_abs_d3_hat": 50000.0,
            "model_gain_epsilon": 1e-6,
            "model_probe_rudder_deg": 1.0,
        },
    },

    # Evaluate reward configurations
    'reward_configs': [
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -1.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": 0.0,
            },
        },
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -1.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": 0.0,
            },
        },
    ],
}


def get_hyperparameters(agent_name, env_name=None):
    """
    Get validated hyperparameters.

    AirSim uses the base PPO tuning set. Other environments get the legacy
    PPO settings through the override map so their training behavior stays
    separate from the AirSim tuning.
    """

    try:
        params = DEFAULT_HYPERPARAMETERS[agent_name].copy()
        if env_name == "AirSimEnv":
            params.update(AIRSIM_HYPERPARAMETER_OVERRIDES.get(agent_name, {}))
        return params
    except KeyError:
        raise ValueError(f"Unknown agent type: {agent_name}")


def load_config_parameters() -> dict:
    """
    Return a raw copy of CONFIG_PARAMETERS without applying any profile.

    Use this when you need the plain configuration data and will handle
    profile application yourself (e.g. in get_config_parameters below, or
    in tests that want to control the profile step explicitly).
    """

    return deepcopy(CONFIG_PARAMETERS)


def get_config_parameters() -> dict:
    """
    Return configuration parameters with the configured profile applied.

    Resolves the profile name from the config, validates it, applies it to
    the config dict, and returns the result. This is the single authoritative
    place where profile application onto the config happens — callers should
    not re-apply the profile to the config dict after calling this function.

    Returns:
        dict: A copy of CONFIG_PARAMETERS with the profile's overrides applied.
    """

    framework_config = load_config_parameters()
    profile_name = framework_config.get("profile", "default_profile")
    profile = ProfileRegistry.get(profile_name)
    profile.validate()
    apply_profile(profile, config=framework_config)

    return framework_config
