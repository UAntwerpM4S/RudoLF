import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from contextlib import contextmanager
from fw.environments.base_env import BaseEnv
from fw.environments.multi_sim_env import MultiSimEnv
from fw.common.tools.fh_sim_utils import fh_sim_cleanup
from fw.configuration.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, PolicyStrategy
from fw.configuration.profile_applier import apply_profile, ActionAdapter, SimulatorApplier
from airsim.communication.airsim_bridge import AirSimBridge
from fw.configuration.config_profile import VesselProfile
from fw.configuration.stop_condition import StopCondition
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.base_agent import BaseAgent
from fw.policies.ppo2_agent import PPO2Agent

logger = logging.getLogger(__name__)

MIN_GRID_POS = -11700
MAX_GRID_POS = 14500


class AgentManagerSimple:
    """
    Creates the agent for an environment, trains it and visualizes evaluation paths.
    """

    def render(
        self,
        env,
        start_pos: np.ndarray,
        all_paths: list,
        total_rewards: list,
        show_plots: bool = True,
    ) -> None:
        """
        Plot the evaluation paths with checkpoints and waterway and save
        <model_dir>/trajectory.png.

        Args:
            env: Environment with checkpoints, obstacles, overall and target_pos.
            start_pos (np.ndarray): Start position.
            all_paths (list): One path array per episode.
            total_rewards (list): Reward per episode (mean step reward, see
                visualize_trained_agent).
            show_plots (bool): Show the figure window.
        """

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Grid boundary
            ax.plot(
                [MIN_GRID_POS, MAX_GRID_POS, MAX_GRID_POS, MIN_GRID_POS, MIN_GRID_POS],
                [MIN_GRID_POS, MIN_GRID_POS, MAX_GRID_POS, MAX_GRID_POS, MIN_GRID_POS],
                'k-', label='Boundary'
            )

            checkpoint_positions = np.array([checkpoint['pos'] for checkpoint in env.checkpoints])
            for checkpoint in env.checkpoints:
                circle = plt.Circle(
                    (checkpoint['pos'][0], checkpoint['pos'][1]),
                    radius=10,
                    color='gray',
                    alpha=0.5,
                    fill=True
                )
                ax.add_patch(circle)

            ax.plot(
                checkpoint_positions[:, 0], checkpoint_positions[:, 1],
                'g--', alpha=0.5, label='Ideal Path'
            )

            ax.add_patch(
                patches.Polygon(
                    env.obstacles,
                    closed=True,
                    edgecolor='r',
                    facecolor='none',
                    lw=2,
                    label='Waterway'
                )
            )

            ax.add_patch(
                patches.Polygon(
                    env.overall,
                    closed=True,
                    edgecolor='brown',
                    facecolor='none',
                    lw=2,
                    label='Western Scheldt'
                )
            )

            colors = plt.cm.rainbow(np.linspace(0, 1, len(all_paths)))
            for i, path in enumerate(all_paths):
                ax.plot(
                    path[:, 0], path[:, 1],
                    '-', color=colors[i], alpha=0.7,
                    label=f'Episode {i + 1} (reward: {total_rewards[i]:.1f})'
                )

            ax.scatter([start_pos[0]], [start_pos[1]], c='blue', s=100, label='Start')
            ax.scatter([env.target_pos[0]], [env.target_pos[1]], c='red', s=100, label='Target')

            ax.set_title('Paths Across Multiple Episodes')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

            ax.set_aspect('equal', adjustable='box')

            model_dir = self._agent.model_dir or 'models'
            os.makedirs(model_dir, exist_ok=True)
            trajectory_file_name = f'{model_dir}/trajectory.png'

            plt.savefig(trajectory_file_name, bbox_inches='tight', dpi=300)
            logger.info("Path visualization saved as 'trajectory.png'.")

            if show_plots:
                plt.show()
        finally:
            plt.close('all')


    def __init__(self, agent_type: str, profile: VesselProfile, hyperparameters: dict):
        """
        Args:
            agent_type (str): Agent type (e.g. 'PPO').
            profile (VesselProfile): Profile for training or visualization.
            hyperparameters (dict): Agent hyperparameters.
        """

        self._agent = None
        self._profile = profile
        self._agent_type = agent_type
        self._hyperparameters = hyperparameters
        self.prev_render_mode = None


    @property
    def agent_type(self) -> str:
        """
        str: Agent type.
        """

        return self._agent_type


    @property
    def profile(self) -> "VesselProfile":
        """
        VesselProfile: Active profile.
        """

        return self._profile


    @property
    def agent(self) -> BaseAgent:
        """
        BaseAgent: Current agent.
        """

        return self._agent


    def _build_single_agent(self, env):
        """
        Create a SinglePPOAgent for env, or for its only sub-environment.

        Raises:
            RuntimeError: If env has more than one sub-environment.
        """

        sub_envs = getattr(env, "sub_environments", None)

        if sub_envs is None or len(sub_envs) < 1:
            target_env = env
        elif len(sub_envs) == 1:
            target_env = next(iter(sub_envs.values()))
        else:
            raise RuntimeError("SinglePPOAgent requires exactly one sub-environment.")

        return SinglePPOAgent(self._profile, environment=target_env, **self._hyperparameters)


    def _build_multi_agent(self, env: BaseEnv):
        """
        Create a MultiPPOAgent with one SinglePPOAgent per sub-environment.

        Raises:
            TypeError: If env is not a MultiSimEnv.
            ValueError: If it has no sub-environments.
        """

        if not isinstance(env, MultiSimEnv):
            raise TypeError("MultiPPOAgent requires a MultiSimEnv instance.")

        if not env.sub_environments:
            raise ValueError("MultiSimEnv must contain at least one sub-environment.")

        agents = [
            SinglePPOAgent(self._profile, environment=sub_env, **self._hyperparameters)
            for sub_env in env.sub_environments.values()
        ]

        return MultiPPOAgent(self.profile, environment=env, agents=agents, **self._hyperparameters)


    def set_environment(self, env: BaseEnv, policy_strategy: PolicyStrategy = PolicyStrategy.RESET_POLICY,
                        other_env_name: str = None, require_policy: bool = False) -> None:
        """
        Build a new agent for env and optionally load an existing policy.

        Without require_policy, falls back to a fresh policy (with a warning) if
        the policy file is missing or incompatible.

        Args:
            env (BaseEnv): Environment to train on.
            policy_strategy (PolicyStrategy): Reset, reuse the current policy, or
                reuse another environment's policy.
            other_env_name (str, optional): Environment whose policy to reuse.
            require_policy (bool, optional): Raise instead of falling back
                (visualization).

        Raises:
            RuntimeError: For an unknown agent type, or (require_policy) if the
                policy cannot be loaded.
            ValueError: require_policy and the policy does not fit the env.
            FileNotFoundError: require_policy and the policy file does not exist.
        """

        policy_file_name = None

        if policy_strategy == PolicyStrategy.REUSE_CURRENT_POLICY and self._agent:
            policy_file_name = f'{self.get_env().training_name}_{self._agent_type}_policy'
            self._agent.save_policy(policy_file_name)
        elif policy_strategy == PolicyStrategy.REUSE_OTHER_POLICY and other_env_name:
            policy_file_name = f'{other_env_name}_{self._agent_type}_policy'

        # Release resources of the replaced agent (e.g. PPO2Agent's SB3 VecEnv).
        if self._agent is not None and callable(getattr(self._agent, "close", None)):
            try:
                self._agent.close()
            except Exception:
                logger.exception("Closing the previous agent failed; continuing.")

        if PPO_POLICY_NAME == self._agent_type:
            self._agent = self._build_single_agent(env)
        elif PPO2_POLICY_NAME == self._agent_type:
            self._agent = PPO2Agent(environment=env, **self._hyperparameters)
        elif MULTI_PPO_POLICY_NAME == self._agent_type:
            self._agent = self._build_multi_agent(env)
        else:
            raise RuntimeError(f"Unknown {self._agent_type} policy type.")

        # Propagate profile env settings (e.g. enable_smoothing).
        apply_profile(self._profile, env=env)

        if policy_file_name:
            full_policy_file_name = f"{policy_file_name}.zip"
            if self.agent.model_dir:
                full_policy_file_name = os.path.join(self.agent.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                logger.info("Reusing %s.", policy_file_name)
                try:
                    self._agent.load_policy(policy_file_name)
                except (ValueError, RuntimeError) as e:
                    if require_policy:
                        raise
                    logger.warning("[Fallback] Incompatible model: %s", e)
                    logger.warning("[Fallback] Reverting to RESET_POLICY for %s.", self._agent_type)
            elif require_policy:
                raise FileNotFoundError(f"The policy file {full_policy_file_name} does not exist.")
            else:
                logger.warning("[Fallback] The policy file %s does not exist.", policy_file_name)
                logger.warning("[Fallback] Reverting to RESET_POLICY for %s.", self._agent_type)


    @contextmanager
    def rendering(self, env):
        """
        Context manager: set env.render_mode to 'human' and restore it afterward
        (no-op if env has no render_mode).
        """

        if hasattr(env, "render_mode"):
            try:
                self.prev_render_mode = env.render_mode
                env.render_mode = "human"
                yield
            finally:
                env.render_mode = self.prev_render_mode
                self.prev_render_mode = None
        else:
            yield


    def train(self, stop_condition: StopCondition, dispose_env: bool = True) -> None:
        """
        Train the agent until stop_condition is met.

        Args:
            stop_condition (StopCondition): When to stop training.
            dispose_env (bool): Dispose the FH Simulator bridge (if any)
                afterward, also on exceptions. Pass False when
                visualize_trained_agent() follows on the same environment, to
                avoid a needless re-initialization; it disposes itself.

        Raises:
            RuntimeError: If no agent has been created.
        """

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        with fh_sim_cleanup(self.get_env, dispose=dispose_env):
            self._agent.learn(
                stop_condition=stop_condition,
            )


    def visualize_trained_agent(
            self,
            num_episodes: int = 5,
            live_animation: bool = False,
            env_color_index: int = 0,
            airsim_bridge: AirSimBridge = None,
            return_summary: bool = False,
            show_plots: bool = True,
        ) -> dict | None:
        """
        Run the trained agent deterministically and plot the paths.

        Ship environments (with ship_pos) record and plot one path per episode;
        the episode reward is the mean step reward. Other environments are only
        rendered for one episode.

        Args:
            num_episodes (int): Number of episodes.
            live_animation (bool): Render every step.
            env_color_index (int): Environment color index (unused here).
            airsim_bridge (AirSimBridge, optional): Connected Cosys-AirSim bridge;
                the ship pose is mirrored to UE5 and the bridge is disconnected
                afterward.
            return_summary (bool): Return average/best/worst reward per environment.
            show_plots (bool): Show the plot window.

        Returns:
            dict | None: Summary if return_summary and a ship environment, else None.

        Raises:
            RuntimeError: If no agent has been created.
        """

        # self._agent.policy = PPOPolicy.load("PySimEnv_PPO_policy", device="cpu")

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        logger.info("Running %d episodes...", num_episodes)

        env = self.get_env()
        with fh_sim_cleanup(lambda: env):
            self.agent.set_policy_eval()

            coord_transform = None
            if airsim_bridge and airsim_bridge.is_connected:
                from airsim.communication.coord_transform import CoordTransform
                coord_transform = CoordTransform()

            with self.rendering(env):
                if hasattr(env, "ship_pos"):
                    all_paths = []
                    total_rewards = []
                    start_pos = None

                    try:
                        for episode in range(num_episodes):
                            state, _ = env.reset()

                            if self._profile is not None:
                                _rollout_sim = getattr(env, "phys_sim", None) or getattr(env, "_phys_sim", None)
                                SimulatorApplier.apply(self._profile, _rollout_sim)

                            start_pos = np.copy(env.ship_pos)
                            episode_reward = 0
                            done = False
                            steps = 0

                            path_positions = [np.copy(env.ship_pos)]

                            while not done:
                                action, _, _ = self._agent.predict(state=state, deterministic=True)

                                if self._profile is not None:
                                    action = ActionAdapter.transform(self._profile, action)

                                state, reward, terminated, truncated, _ = env.step(action)
                                done = terminated or truncated

                                # Running mean of the step rewards, terminal step included
                                steps += 1
                                episode_reward += (reward - episode_reward) / steps

                                path_positions.append(np.copy(env.ship_pos))

                                if not done:
                                    if live_animation:
                                        env.render()

                                    # Mirror the ship pose to UE5
                                    if coord_transform and hasattr(env, 'ship_pos') and hasattr(env, 'phys_sim'):
                                        ax, ay, _ = coord_transform.framework_to_airsim(
                                            float(env.ship_pos[0][0]),
                                            float(env.ship_pos[0][1])
                                        )
                                        ayaw = coord_transform.framework_heading_to_airsim_yaw(
                                            float(env.phys_sim.state.heading)
                                        )
                                        airsim_bridge.set_pose(ax, ay, 0.0, ayaw)
                                else:
                                    logger.info("Episode %d finished after %d steps with reward %.2f", episode + 1, steps, episode_reward)
                                    break

                            all_paths.append(np.array(path_positions))
                            total_rewards.append(episode_reward)
                    finally:
                        if airsim_bridge is not None:
                            airsim_bridge.disconnect()

                    avg_reward = np.mean(total_rewards)
                    logger.info("Evaluation completed! Average reward: %.2f", avg_reward)
                    logger.info("Best episode reward: %.2f", max(total_rewards))
                    logger.info("Worst episode reward: %.2f", min(total_rewards))

                    self.render(
                        env,
                        start_pos,
                        all_paths,
                        total_rewards,
                        show_plots=show_plots,
                    )

                    if return_summary:
                        env_name = getattr(env, "type_name", "environment")
                        return {
                            env_name: {
                                "average_reward": float(avg_reward),
                                "best_episode_reward": float(max(total_rewards)),
                                "worst_episode_reward": float(min(total_rewards)),
                            }
                        }
                else:
                    # Non-ship environment: render a single episode
                    obs, _ = env.reset()

                    env.render()

                    done = False
                    while not done:
                        prediction = self._agent.predict(state=obs, deterministic=True)
                        action = prediction[0]
                        obs, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        env.render()

                return None


    def get_env(self):
        """
        Return the agent's environment.
        """

        return self._agent.get_env()
