import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from contextlib import contextmanager
from fw.config_profile import VesselProfile
from fw.environments.base_env import BaseEnv
from fw.environments.multi_sim_env import MultiSimEnv
from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, PolicyStrategy
from fw.profile_applier import apply_profile, ActionAdapter, SimulatorApplier
from airsim.communication.airsim_bridge import AirSimBridge
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.base_agent import BaseAgent
from fw.policies.ppo2_agent import PPO2Agent
from fw.stop_condition import StopCondition

MIN_GRID_POS = -11700
MAX_GRID_POS = 14500


class AgentManagerSimple:
    """
    AgentManagerSimple class for managing agents and environments during training.

    This class includes methods for training, saving/loading policies, and
    visualizing paths taken by the agent during evaluation.
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
        Render and save a visualization of paths taken by the agent.

        Args:
            env: Environment object containing boundaries, obstacles, and checkpoints.
            start_pos (np.ndarray): Starting position of the agent.
            all_paths (list): List of paths taken over multiple episodes.
            total_rewards (list): List of total rewards corresponding to each episode.
            show_plots (bool): Whether to show plots or not.
        """

        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Draw environment boundaries
            ax.plot(
                [MIN_GRID_POS, MAX_GRID_POS, MAX_GRID_POS, MIN_GRID_POS, MIN_GRID_POS],
                [MIN_GRID_POS, MIN_GRID_POS, MAX_GRID_POS, MAX_GRID_POS, MIN_GRID_POS],
                'k-', label='Boundary'
            )

            # Plot all checkpoints as semi-transparent circles
            checkpoint_positions = np.array([checkpoint['pos'] for checkpoint in env.checkpoints])
            for checkpoint in env.checkpoints:
                circle = plt.Circle(
                    (checkpoint['pos'][0], checkpoint['pos'][1]),
                    radius=10,  # Radius can be tuned to match environment scale
                    color='gray',
                    alpha=0.5,
                    fill=True
                )
                ax.add_patch(circle)

            # Draw ideal straight-line connections between checkpoints
            ax.plot(
                checkpoint_positions[:, 0], checkpoint_positions[:, 1],
                'g--', alpha=0.5, label='Ideal Path'
            )

            # Plot environment obstacles (e.g., waterway)
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

            # Plot additional environment features (e.g., general area)
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

            # Plot all agent paths from different episodes
            colors = plt.cm.rainbow(np.linspace(0, 1, len(all_paths)))
            for i, path in enumerate(all_paths):
                ax.plot(
                    path[:, 0], path[:, 1],
                    '-', color=colors[i], alpha=0.7,
                    label=f'Episode {i + 1} (reward: {total_rewards[i]:.1f})'
                )

            # Mark start and target positions
            ax.scatter([start_pos[0]], [start_pos[1]], c='blue', s=100, label='Start')
            ax.scatter([env.target_pos[0]], [env.target_pos[1]], c='red', s=100, label='Target')

            # Final plot adjustments
            ax.set_title('Paths Across Multiple Episodes')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

            # Set aspect ratio to be equal
            ax.set_aspect('equal', adjustable='box')

            model_dir = self._agent.model_dir or 'models'
            os.makedirs(model_dir, exist_ok=True)
            trajectory_file_name = f'{model_dir}/trajectory.png'

            # Save the figure
            plt.savefig(trajectory_file_name, bbox_inches='tight', dpi=300)
            print("\nPath visualization saved as 'trajectory.png'.")

            if show_plots:
                plt.show()
        finally:
            plt.close('all')


    def __init__(self, agent_type: str, profile: VesselProfile, hyperparameters: dict):
        """
        Initialize AgentManagerSimple.

        Args:
            agent_type (str): Type of agent to use (e.g., 'PPO').
            profile (VesselProfile): Profile that will be used for training or visualization.
            hyperparameters (dict): Hyperparameters for configuring the agent.
        """

        self._agent = None
        self._profile = profile
        self._agent_type = agent_type
        self._hyperparameters = hyperparameters
        self.prev_render_mode = None


    @property
    def agent_type(self) -> str:
        """
        Get the type of agent being used by the agent manager.

        Returns:
            str: The agent type.
        """

        return self._agent_type


    @property
    def profile(self) -> "VesselProfile":
        """
        Get the active VesselProfile held by this manager.

        Returns:
            VesselProfile: The profile used for training or visualization.
        """

        return self._profile


    @property
    def agent(self) -> BaseAgent:
        """
        Get the agent that is used by the agent manager.

        Returns:
            BaseAgent: The agent.
        """

        return self._agent


    def _build_single_agent(self, env):
        """
        Create a `SinglePPOAgent` from the given environment.

        If the environment exposes a `sub_environments` attribute, this method
        ensures that exactly one sub-environment is used. Otherwise, the environment
        itself is used directly.

        Args:
            env: Environment instance, optionally containing `sub_environments`.

        Returns:
            SinglePPOAgent: Initialized PPO agent for a single environment.

        Raises:
            RuntimeError: If more than one sub-environment is provided.
        """

        sub_envs = getattr(env, "sub_environments", None)

        if sub_envs is None or len(sub_envs) < 1:
            target_env = env
        elif len(sub_envs) == 1:
            target_env = next(iter(sub_envs.values()))
        else:
            raise RuntimeError("SinglePPOAgent requires exactly one sub-environment.")

        return SinglePPOAgent(self._profile, environment=target_env, **self._hyperparameters)


    def _build_multi_agent(self, env):
        """
        Create a `MultiPPOAgent` composed of multiple `SinglePPOAgent` instances.

        Each sub-environment in the provided `MultiSimEnv` is assigned its own
        `SinglePPOAgent`. These agents are then combined into a `MultiPPOAgent`
        for coordinated training.

        Args:
            env (MultiSimEnv): Multi-environment containing sub-environments.

        Returns:
            MultiPPOAgent: Initialized multi-agent PPO trainer.

        Raises:
            TypeError: If `env` is not an instance of `MultiSimEnv`.
            ValueError: If no sub-environments are defined.
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
                        other_env_name: str = None) -> None:
        """
        Assign an environment to the agent and configure the policy strategy.

        Args:
            env (BaseEnv): The environment to set.
            policy_strategy (PolicyStrategy, optional): Strategy for handling policy (reset, reuse current, or reuse other). Default is PolicyStrategy.RESET_POLICY.
            other_env_name (str, optional): Name of the environment whose policy should be reused (if applicable).

        Raises:
            RuntimeError: If an unknown policy type is specified.
        """

        policy_file_name = None

        if policy_strategy == PolicyStrategy.REUSE_CURRENT_POLICY and self._agent:
            policy_file_name = f'{self.get_env().type_name}_{self._agent_type}_policy'
            self._agent.save_policy(policy_file_name)
        elif policy_strategy == PolicyStrategy.REUSE_OTHER_POLICY and other_env_name:
            policy_file_name = f'{other_env_name}_{self._agent_type}_policy'

        if PPO_POLICY_NAME == self._agent_type:
            self._agent = self._build_single_agent(env)
        elif PPO2_POLICY_NAME == self._agent_type:
            self._agent = PPO2Agent(environment=env, **self._hyperparameters)
        elif MULTI_PPO_POLICY_NAME == self._agent_type:
            self._agent = self._build_multi_agent(env)
        else:
            raise RuntimeError(f"Unknown {self._agent_type} policy type.")

        # Always propagate profile env settings (e.g. enable_smoothing) to the
        # environment, regardless of which entry point called set_environment.
        apply_profile(self._profile, env=env)

        if policy_file_name:
            full_policy_file_name = f"{policy_file_name}.zip"
            if self.agent.model_dir:
                full_policy_file_name = os.path.join(self.agent.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                print(f"Reusing {policy_file_name}.")
                try:
                    self._agent.load_policy(policy_file_name)
                except ValueError as e:
                    print(f"[Fallback] Incompatible model: {e}")
                    print(f"[Fallback] Reverting to RESET_POLICY for {self._agent_type}.")
            else:
                print(f"[Fallback] The policy file {policy_file_name} does not exist.")
                print(f"[Fallback] Reverting to RESET_POLICY for {self._agent_type}.")


    @contextmanager
    def rendering(self, env):
        """
        Temporarily force an environment's render_mode to 'human' for the
        duration of the `with` block, restoring its previous value afterward.

        Args:
            env: The environment instance to toggle render_mode on. If it
                 has no `render_mode` attribute, this is a no-op (the block
                 still runs, just without forcing rendering on).
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


    def train(self, stop_condition: StopCondition) -> None:
        """
        Train the agent in the assigned environment.

        Args:
            stop_condition (StopCondition): The condition that determines when training should stop.

        Raises:
            RuntimeError: If no agent is created.
        """

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        # Start training
        self._agent.learn(
            stop_condition=stop_condition,
        )

        # Cleanup the environment
        if hasattr(self.get_env(), "fh_sim"):
            self.get_env().fh_sim.dispose()


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
        Run and visualize a trained agent in the environment over multiple episodes.

        Args:
            num_episodes (int, optional): Number of episodes to visualize (default is 5).
            live_animation (bool, optional): Enable live animation of the agent in the environment.
            env_color_index (int, optional): Index of the environment color.
            airsim_bridge (AirSimBridge, optional): AirSimBridge to create a connection with the Cosys AirSim.
            return_summary (bool, optional): Return evaluation summary metrics when available.
            show_plots (bool, optional): Show plot windows when visualization is generated.

        Raises:
            RuntimeError: If no agent is created.
        """

        # Load checkpoint
        # self._agent.policy = PPOPolicy.load("PySimEnv_PPO_policy", device="cpu")

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        print(f"\nRunning {num_episodes} episodes...")

        self.agent.set_policy_eval()
        env = self.get_env()

        # Prepare framework->AirSim coordinate transform if bridge is active
        coord_transform = None
        if airsim_bridge and airsim_bridge.is_connected:
            from airsim.communication.coord_transform import CoordTransform
            coord_transform = CoordTransform()

        with self.rendering(env):
            if hasattr(env, "ship_pos"):
                # Store results for each episode
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

                        # Store positions for this episode
                        path_positions = [np.copy(env.ship_pos)]

                        while not done:
                            # Select action
                            action, _, _ = self._agent.predict(state=state, deterministic=True)

                            if self._profile is not None:
                                action = ActionAdapter.transform(self._profile, action)

                            # Take step in environment
                            state, reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated

                            # Include every reward, including the terminal transition
                            steps += 1
                            episode_reward += (reward - episode_reward) / steps

                            # Store the new position (also the terminal one)
                            path_positions.append(np.copy(env.ship_pos))

                            if not done:
                                # Uncomment for animated evaluation
                                if live_animation:
                                    env.render()

                                # Bridge ship position and heading to UE5 via coordinate transform
                                if coord_transform and hasattr(env, 'ship_pos') and hasattr(env, 'phys_sims'):
                                    ax, ay, _ = coord_transform.framework_to_airsim(
                                        float(env.ship_pos[0][0]),
                                        float(env.ship_pos[0][1])
                                    )
                                    ayaw = coord_transform.framework_heading_to_airsim_yaw(
                                        float(env.phys_sims[0].state.heading)
                                    )
                                    airsim_bridge.set_pose(ax, ay, 0.0, ayaw)
                            else:
                                print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                                break

                        all_paths.append(np.array(path_positions))
                        total_rewards.append(episode_reward)
                finally:
                    if airsim_bridge is not None:
                        airsim_bridge.disconnect()

                # Print summary statistics
                avg_reward = np.mean(total_rewards)
                print(f"\nEvaluation completed! Average reward: {avg_reward:.2f}")
                print(f"Best episode reward: {max(total_rewards):.2f}")
                print(f"Worst episode reward: {min(total_rewards):.2f}")

                self.render(
                    env,
                    start_pos,
                    all_paths,
                    total_rewards,
                    show_plots=show_plots,
                )

                # Cleanup the environment
                if hasattr(env, "fh_sim"):
                    env.fh_sim.dispose()

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
                # Reset the environment
                obs, _ = env.reset()

                # Render the environment before starting the loop
                env.render()

                # Run the agent in the environment
                done = False
                while not done:
                    # The agent chooses an action based on the current observation
                    prediction = self._agent.predict(state=obs, deterministic=True)
                    action = prediction[0]
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Render the environment each step to visualize the agent's action
                    env.render()

            return None


    def get_env(self):
        """
        Retrieve the environment associated with the agent.

        Returns:
            The environment object.
        """

        return self._agent.get_env()
