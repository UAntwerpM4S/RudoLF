"""
viz_refactor.py
---------------
Refactored multi-ship visualization pipeline for autonomous ship RL training.

Structure:
  ShipEpisodeData   – dataclass holding one ship's data for one episode
  EpisodeCollector  – runs N episodes, returns ship_data[ship_id][episode]
  MetricsComputer   – computes per-ship metrics from collected data
  CollisionAnnotator– detects near-miss / collision events across ships
  MatplotlibRenderer– saves static multi-ship trajectory PNGs
  PlotlyRenderer    – produces interactive HTML dashboard (optional)

Drop-in usage in AgentManager.visualize_trained_agent():

    collector = EpisodeCollector(agent, env)
    ship_data = collector.collect(num_episodes=5)

    metrics = MetricsComputer.compute(ship_data)
    CollisionAnnotator.annotate(ship_data, threshold=50.0)

    MatplotlibRenderer.render(ship_data, metrics, env, save_path="trajectory.png")
    PlotlyRenderer.render(ship_data, metrics, save_path="dashboard.html")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataclasses import dataclass, field
from typing import Optional
from matplotlib.colors import to_rgb


# ---------------------------------------------------------------------------
# Helpers (unchanged from original agent_manager.py)
# ---------------------------------------------------------------------------

def normalized_time_axis(length: int) -> np.ndarray:
    if length <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 100.0, length)


def shade_color(base_color, n, lo=0.45, hi=1.0):
    rgb = np.array(to_rgb(base_color))
    factors = np.linspace(hi, lo, n)
    return [tuple(np.clip(rgb * f, 0, 1)) for f in factors]


def resample_to_common_time(signal: np.ndarray, common_time: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.shape[0] == 1:
        return np.repeat(signal, len(common_time), axis=0)
    original_time = np.linspace(0.0, 1.0, signal.shape[0])
    if signal.ndim == 1:
        return np.interp(common_time, original_time, signal)
    T = signal.shape[0]
    rest_shape = signal.shape[1:]
    flat_signal = signal.reshape(T, -1)
    resampled = np.stack(
        [np.interp(common_time, original_time, flat_signal[:, i])
         for i in range(flat_signal.shape[1])],
        axis=1,
    )
    return resampled.reshape(len(common_time), *rest_shape)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CollisionEvent:
    """Records a near-miss or collision between two ships at a given timestep."""
    timestep: int
    ship_a: str
    ship_b: str
    distance: float
    is_collision: bool  # True if distance < hard collision threshold


@dataclass
class ShipEpisodeData:
    """
    All recorded data for a single ship over a single episode.

    Attributes
    ----------
    ship_id     : identifier matching the key in MultiSimEnv.sub_environments
    episode     : zero-based episode index
    path        : (T, 2) array of (x, y) positions
    cross_errors: cross-track errors per step (length may be < T due to
                  checkpoint gating in the original code)
    rudder_actions     : rudder angle at each step
    rudder_changes     : absolute change in rudder angle per step
    heading_errors     : heading error (rad) per step
    velocities         : surge velocity per step
    ukc_measurements   : under-keel clearance per step
    reward             : scalar episode reward
    collision_events   : populated later by CollisionAnnotator
    """
    ship_id: str
    episode: int
    path: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    cross_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    rudder_actions: np.ndarray = field(default_factory=lambda: np.array([]))
    rudder_changes: np.ndarray = field(default_factory=lambda: np.array([]))
    heading_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    ukc_measurements: np.ndarray = field(default_factory=lambda: np.array([]))
    reward: float = 0.0
    collision_events: list[CollisionEvent] = field(default_factory=list)


# Type alias for the full data structure:
#   ship_data["ship_0"][episode_index] → ShipEpisodeData
ShipData = dict[str, list[ShipEpisodeData]]


# ---------------------------------------------------------------------------
# Episode collector
# ---------------------------------------------------------------------------

class EpisodeCollector:
    """
    Runs evaluation episodes and returns structured per-ship data.

    Replaces the inline episode loop in AgentManager.visualize_trained_agent().
    Works for both single-ship (one entry in ship_data) and multi-ship setups.

    Parameters
    ----------
    agent       : trained agent (SinglePPOAgent or MultiPPOAgent)
    env         : the environment to evaluate in
    ship_ids    : list of ship identifiers; defaults to ["ship_0"] for
                  single-ship setups, or the keys of env.sub_environments
    """

    def __init__(self, agent, env, ship_ids: Optional[list[str]] = None):
        self._agent = agent
        self._env = env

        if ship_ids is not None:
            self._ship_ids = ship_ids
        elif hasattr(env, "sub_environments") and self._env.sub_environments:  # <-- add truthiness check
            self._ship_ids = list(env.sub_environments.keys())
        else:
            self._ship_ids = ["ship_0"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, num_episodes: int = 5) -> ShipData:
        """
        Run `num_episodes` evaluation episodes.

        Returns
        -------
        ship_data : dict mapping each ship_id to a list of ShipEpisodeData,
                    one entry per episode.
        """
        ship_data: ShipData = {sid: [] for sid in self._ship_ids}

        for episode in range(num_episodes):
            episode_data = self._run_episode(episode)
            for sid, data in episode_data.items():
                ship_data[sid].append(data)

        # Diagnose empty collection early
        total_steps = sum(
            sum(len(ep.path) for ep in episodes)
            for episodes in ship_data.values()
        )
        if total_steps == 0:
            raise RuntimeError(
                f"EpisodeCollector collected 0 steps across all ships. "
                f"ship_ids={self._ship_ids}, env type={type(self._env).__name__}. "
                f"Check that _get_ship_env returns the correct env object."
            )

        return ship_data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_episode(self, episode: int) -> dict[str, ShipEpisodeData]:
        """Run a single episode and return per-ship data."""
        env = self._env

        # Initialise per-ship accumulators
        accumulators: dict[str, dict] = {
            sid: self._empty_accumulator(sid, episode)
            for sid in self._ship_ids
        }

        state, _ = env.reset()
        done = np.zeros(len(self._ship_ids), dtype=bool)
        steps = 0

        # Capture initial state for each ship
        for sid in self._ship_ids:
            ship_env = self._get_ship_env(sid)
            if ship_env is not None and hasattr(ship_env, "ship_pos"):
                accumulators[sid]["path"].append(np.copy(ship_env.ship_pos))
            if ship_env is not None and hasattr(ship_env, "cross_error"):
                accumulators[sid]["cross_errors"].append(ship_env.cross_error)
            if ship_env is not None and hasattr(ship_env, "velocities"):
                accumulators[sid]["velocities"].append(ship_env.velocities)
            if ship_env is not None and hasattr(ship_env, "phys_sim"):
                accumulators[sid]["heading_errors"].append(
                    self._compute_heading_error(ship_env)
                )

        while not np.all(done):
            action, _, _ = self._agent.predict(state=state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            env_done = np.logical_or(terminated, truncated)
            done = np.logical_or(done, env_done)

            if not np.all(done):
                steps += 1
                self._record_step(accumulators, reward, steps)

        # Finalise reward (running mean stored in accumulator)
        for sid in self._ship_ids:
            accumulators[sid]["reward"] = accumulators[sid].get("_running_reward", 0.0)

        print(f"Episode {episode + 1} finished after {steps} steps")

        return {
            sid: self._accumulator_to_episode_data(acc)
            for sid, acc in accumulators.items()
        }

    def _record_step(self, accumulators: dict, reward: float, steps: int) -> None:
        """Append one timestep of observations to each ship's accumulator."""
        env = self._env

        for sid in self._ship_ids:
            ship_env = self._get_ship_env(sid)
            acc = accumulators[sid]

            if ship_env is None:
                continue

            if hasattr(ship_env, "ship_pos"):
                acc["path"].append(np.copy(ship_env.ship_pos))

            # Cross-track error: gated on checkpoint index (mirrors original logic)
            if hasattr(ship_env, "cross_error") and hasattr(ship_env, "checkpoint_index"):
                if ship_env.checkpoint_index >= 2:
                    acc["cross_errors"].append(ship_env.cross_error)

            if hasattr(ship_env, "heading_error"):
                acc["heading_errors"].append(ship_env.heading_error)
            if hasattr(ship_env, "velocities"):
                acc["velocities"].append(ship_env.velocities)
            if hasattr(ship_env, "performed_action"):
                prev = acc["rudder_actions"][-1] if acc["rudder_actions"] else 0.0
                acc["rudder_actions"].append(ship_env.performed_action[0])
                acc["rudder_changes"].append(abs(ship_env.performed_action[0] - prev))
            if hasattr(ship_env, "ukc"):
                acc["ukc_measurements"].append(ship_env.ukc)

            # Running mean reward (simple incremental average)
            prev_reward = acc.get("_running_reward", 0.0)
            acc["_running_reward"] = prev_reward + (reward - prev_reward) / steps

    def _get_ship_env(self, ship_id: str):
        """Return the sub-environment for a ship, or the env itself for single-ship."""
        if hasattr(self._env, "sub_environments") and self._env.sub_environments:
            return self._env.sub_environments.get(ship_id)
        return self._env  # single-ship: env IS the ship env

    @staticmethod
    def _empty_accumulator(ship_id: str, episode: int) -> dict:
        return {
            "ship_id": ship_id,
            "episode": episode,
            "path": [],
            "cross_errors": [],
            "rudder_actions": [],
            "rudder_changes": [],
            "heading_errors": [],
            "velocities": [],
            "ukc_measurements": [],
            "reward": 0.0,
        }

    @staticmethod
    def _accumulator_to_episode_data(acc: dict) -> ShipEpisodeData:
        return ShipEpisodeData(
            ship_id=acc["ship_id"],
            episode=acc["episode"],
            path=np.array(acc["path"]) if acc["path"] else np.empty((0, 2)),
            cross_errors=np.array(acc["cross_errors"]),
            rudder_actions=np.array(acc["rudder_actions"]),
            rudder_changes=np.array(acc["rudder_changes"]),
            heading_errors=np.array(acc["heading_errors"]),
            velocities=np.array(acc["velocities"]),
            ukc_measurements=np.array(acc["ukc_measurements"]),
            reward=acc.get("_running_reward", 0.0),
        )

    @staticmethod
    def _compute_heading_error(ship_env) -> float:
        """Mirrors AgentManager.compute_heading_error for a sub-environment."""
        checkpoints = ship_env.checkpoints
        idx = ship_env.checkpoint_index
        curr = checkpoints[idx]
        nxt = checkpoints[min(idx + 1, len(checkpoints) - 1)]
        desired = np.arctan2(
            nxt["pos"][1] - curr["pos"][1],
            nxt["pos"][0] - curr["pos"][0],
        )
        return (ship_env.phys_sim.state.heading - desired + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ShipMetrics:
    """Per-ship, per-run aggregate metrics."""
    ship_id: str
    mTEI: float = 0.0   # Mean Track Error Integral
    MTE: float = 0.0    # Maximum Track Error
    mRA: float = 0.0    # Mean Rudder Angle
    MRA: float = 0.0    # Max Rudder Angle
    mRAC: float = 0.0   # Mean Rudder Angle Change
    mHE: float = 0.0    # Mean Heading Error
    mu: float = 0.0     # Mean surge velocity
    Mu: float = 0.0     # Max surge velocity
    avg_reward: float = 0.0
    best_reward: float = 0.0
    worst_reward: float = 0.0


class MetricsComputer:
    """
    Computes per-ship aggregate metrics from ShipData.

    Usage
    -----
        metrics: dict[str, ShipMetrics] = MetricsComputer.compute(ship_data)
    """

    @staticmethod
    def compute(ship_data: ShipData) -> dict[str, ShipMetrics]:
        result = {}
        for ship_id, episodes in ship_data.items():
            result[ship_id] = MetricsComputer._compute_ship(ship_id, episodes)
        return result

    @staticmethod
    def _compute_ship(ship_id: str, episodes: list[ShipEpisodeData]) -> ShipMetrics:
        m = ShipMetrics(ship_id=ship_id)
        if not episodes:
            return m

        rewards = [ep.reward for ep in episodes]
        m.avg_reward = float(np.mean(rewards))
        m.best_reward = float(np.max(rewards))
        m.worst_reward = float(np.min(rewards))

        per_ep_mTEI, per_ep_MTE = [], []
        per_ep_mRA, per_ep_MRA, per_ep_mRAC = [], [], []
        per_ep_mHE, per_ep_mu, per_ep_Mu = [], [], []

        for ep in episodes:
            if ep.cross_errors.size:
                per_ep_mTEI.append(np.mean(np.abs(ep.cross_errors)))
                per_ep_MTE.append(np.max(np.abs(ep.cross_errors)))
            if ep.rudder_actions.size:
                per_ep_mRA.append(np.mean(np.abs(ep.rudder_actions)))
                per_ep_MRA.append(np.max(np.abs(ep.rudder_actions)))
            if ep.rudder_changes.size:
                per_ep_mRAC.append(np.mean(ep.rudder_changes))
            if ep.heading_errors.size:
                per_ep_mHE.append(np.mean(np.abs(ep.heading_errors)))
            if ep.velocities.size:
                per_ep_mu.append(np.mean(np.abs(ep.velocities)))
                per_ep_Mu.append(np.max(np.abs(ep.velocities)))

        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        def safe_max(lst):  return float(np.max(lst))  if lst else 0.0

        m.mTEI = safe_mean(per_ep_mTEI)
        m.MTE  = safe_max(per_ep_MTE)
        m.mRA  = safe_mean(per_ep_mRA)
        m.MRA  = safe_max(per_ep_MRA)
        m.mRAC = safe_mean(per_ep_mRAC)
        m.mHE  = safe_mean(per_ep_mHE)
        m.mu   = safe_mean(per_ep_mu)
        m.Mu   = safe_max(per_ep_Mu)

        return m


# ---------------------------------------------------------------------------
# Collision annotator
# ---------------------------------------------------------------------------

class CollisionAnnotator:
    """
    Detects near-miss and collision events between ships and attaches them
    to the relevant ShipEpisodeData objects.

    Parameters
    ----------
    near_miss_threshold : distance (same units as ship_pos) below which
                          a near-miss event is recorded
    collision_threshold : distance below which a hard collision is recorded
    """

    def __init__(self, near_miss_threshold: float = 50.0, collision_threshold: float = 15.0):
        self.near_miss_threshold = near_miss_threshold
        self.collision_threshold = collision_threshold

    def annotate(self, ship_data: ShipData) -> None:
        """
        Annotate ship_data in-place with CollisionEvent objects.

        Iterates over all pairs of ships and compares positions at each
        shared timestep within each episode.
        """
        ship_ids = list(ship_data.keys())

        if len(ship_ids) < 2:
            return  # nothing to compare in single-ship setup

        num_episodes = len(next(iter(ship_data.values())))

        for episode_idx in range(num_episodes):
            for i in range(len(ship_ids)):
                for j in range(i + 1, len(ship_ids)):
                    sid_a = ship_ids[i]
                    sid_b = ship_ids[j]
                    ep_a = ship_data[sid_a][episode_idx]
                    ep_b = ship_data[sid_b][episode_idx]
                    self._compare_paths(ep_a, ep_b)

    def _compare_paths(self, ep_a: ShipEpisodeData, ep_b: ShipEpisodeData) -> None:
        """Compare two ships' paths and record proximity events on both."""
        min_len = min(len(ep_a.path), len(ep_b.path))
        if min_len == 0:
            return

        for t in range(min_len):
            dist = float(np.linalg.norm(ep_a.path[t] - ep_b.path[t]))
            if dist < self.near_miss_threshold:
                is_collision = dist < self.collision_threshold
                event = CollisionEvent(
                    timestep=t,
                    ship_a=ep_a.ship_id,
                    ship_b=ep_b.ship_id,
                    distance=dist,
                    is_collision=is_collision,
                )
                ep_a.collision_events.append(event)
                ep_b.collision_events.append(event)


# ---------------------------------------------------------------------------
# Matplotlib renderer
# ---------------------------------------------------------------------------

# One color per ship (cycles if more ships than colors)
SHIP_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # cyan
]


class MatplotlibRenderer:
    """
    Produces a static multi-ship trajectory figure.

    Layout (per ship column):
        Row 0 – Trajectory map
        Row 1 – Cross-track error over time
        Row 2 – Rudder actions over time
        Row 3 – UKC over time

    Each ship gets its own color; episodes within a ship are shaded variants.
    Collision events are annotated on the trajectory map.

    Usage
    -----
        MatplotlibRenderer.render(ship_data, metrics, env, save_path="trajectory.png")
    """

    COMMON_TIME = np.linspace(0.0, 1.0, 200)

    @classmethod
    def render(
        cls,
        ship_data: ShipData,
        metrics: dict[str, ShipMetrics],
        env,
        save_path: str = "trajectory.png",
        fig_width_per_ship: int = 10,
    ) -> None:
        ship_ids = [sid for sid in ship_data if ship_data[sid]]  # only ships with data
        n_ships = len(ship_ids)
        n_rows = 4

        if n_ships == 0:
            print("[MatplotlibRenderer] No episode data to render — skipping.")
            return

        try:
            fig, axes = plt.subplots(
                n_rows, n_ships,
                figsize=(fig_width_per_ship * n_ships, 16),
                squeeze=False,   # always (n_rows, n_ships) shape
            )

            for col, ship_id in enumerate(ship_ids):
                episodes = ship_data[ship_id]
                ship_color = SHIP_COLORS[col % len(SHIP_COLORS)]
                colors = shade_color(ship_color, len(episodes))
                m = metrics[ship_id]

                ax_traj  = axes[0][col]
                ax_error = axes[1][col]
                ax_rudder= axes[2][col]
                ax_ukc   = axes[3][col]

                # -- Trajectory map --
                cls._plot_trajectory(ax_traj, episodes, colors, cls._get_ship_env(env, ship_id), ship_id, m)

                # -- Cross-track error --
                cls._plot_signal(
                    ax_error, episodes, colors,
                    signal_attr="cross_errors",
                    title=f"Cross-track error – {ship_id}",
                    ylabel="Cross-track error",
                    ylim=(0, 45),
                )

                # -- Rudder actions --
                cls._plot_rudder(ax_rudder, episodes, colors, ship_id)

                # -- UKC --
                cls._plot_signal(
                    ax_ukc, episodes, colors,
                    signal_attr="ukc_measurements",
                    title=f"UKC – {ship_id}",
                    ylabel="UKC",
                    ylim=(0, 45),
                )

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.show()
            print(f"Trajectory figure saved to '{save_path}'")
        finally:
            plt.close('all')

    @classmethod
    def _get_ship_env(cls, env, ship_id: str):
        """Return the sub-environment for a ship, or the env itself for single-ship."""
        if hasattr(env, "sub_environments") and env.sub_environments:
            return env.sub_environments.get(ship_id)
        return env  # single-ship: env IS the ship env

    # ------------------------------------------------------------------
    # Sub-plot helpers
    # ------------------------------------------------------------------

    @classmethod
    def _plot_trajectory(cls, ax, episodes, colors, env, ship_id, metrics):
        """Trajectory map with waterway, checkpoints, paths, and collision markers."""

        # Waterway geometry
        if hasattr(env, "obstacles"):
            ax.add_patch(patches.Polygon(
                env.obstacles, closed=True,
                edgecolor="r", facecolor="none", lw=2, label="Waterway",
            ))
        if hasattr(env, "overall"):
            ax.add_patch(patches.Polygon(
                env.overall, closed=True,
                edgecolor="brown", facecolor="none", lw=2, label="Western Scheldt",
            ))

        # Checkpoints
        if hasattr(env, "checkpoints"):
            cp_pos = np.array([cp["pos"] for cp in env.checkpoints])
            ax.plot(cp_pos[:, 0], cp_pos[:, 1], "g--", alpha=0.5, label="Ideal path")
            for cp in env.checkpoints:
                ax.add_patch(plt.Circle(
                    cp["pos"], radius=10, color="gray", alpha=0.5, fill=True,
                ))

        # Episode paths
        for i, ep in enumerate(episodes):
            if ep.path.shape[0] < 2:
                continue
            ax.plot(
                ep.path[:, 0], ep.path[:, 1],
                color=colors[i], alpha=0.5,
                label=f"Ep {i + 1} (R={float(np.mean(ep.reward)):.1f})",
            )

            # Start / end markers
            ax.scatter(*ep.path[0],  c="blue", s=80, zorder=5)
            ax.scatter(*ep.path[-1], c="red",  s=80, zorder=5)

            # Collision events
            # for ev in ep.collision_events:
            #     t = min(ev.timestep, len(ep.path) - 1)
            #     marker = "x" if ev.is_collision else "^"
            #     color  = "red" if ev.is_collision else "orange"
            #     ax.scatter(*ep.path[t], marker=marker, c=color, s=120, zorder=6,
            #                label="Collision" if ev.is_collision else "Near-miss")

        ax.set_title(
            f"Trajectory – {ship_id}\n"
            f"mTEI={metrics.mTEI:.1f}  MTE={metrics.MTE:.1f}  avg R={metrics.avg_reward:.1f}"
        )
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True)
        ax.axis("equal")

    @classmethod
    def _plot_signal(cls, ax, episodes, colors, signal_attr, title, ylabel, ylim=None):
        """Generic time-series plot (cross-track error, UKC, …)."""
        all_signals = []

        for i, ep in enumerate(episodes):
            signal = getattr(ep, signal_attr)
            if signal.size == 0:
                continue
            t = normalized_time_axis(len(signal))
            ax.plot(t, signal, color=colors[i], alpha=0.2)
            all_signals.append(signal)

        # Average across episodes (resampled to common time)
        if all_signals:
            resampled = [
                resample_to_common_time(s, cls.COMMON_TIME) for s in all_signals
            ]
            avg = np.mean(resampled, axis=0)
            t_common = normalized_time_axis(len(avg))
            ax.plot(t_common, avg, color="black", alpha=0.7, lw=1.5, label="Average")

        ax.set_title(title)
        ax.set_xlabel("Episode progress [%]")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
            ax.axhline(ylim[0], color="k", lw=0.8, ls="--", alpha=0.4)
            ax.axhline(ylim[1], color="k", lw=0.8, ls="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True)

    @classmethod
    def _plot_rudder(cls, ax, episodes, colors, ship_id):
        """Rudder actions resampled to common time."""
        for i, ep in enumerate(episodes):
            if ep.rudder_actions.size == 0:
                continue
            resampled = resample_to_common_time(ep.rudder_actions, cls.COMMON_TIME)
            ax.plot(cls.COMMON_TIME, resampled, color=colors[i], alpha=0.2)

        ax.set_title(f"Rudder actions – {ship_id}")
        ax.set_xlabel("Episode progress [%]")
        ax.set_ylabel("Rudder action")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True)


# ---------------------------------------------------------------------------
# Plotly renderer (interactive HTML dashboard)
# ---------------------------------------------------------------------------

class PlotlyRenderer:
    """
    Produces an interactive HTML dashboard for multi-ship comparison.

    Requires: pip install plotly

    Features
    --------
    - Toggle individual ships and episodes via the legend
    - Hover to see exact position, timestep, and reward
    - Zoom into collision events
    - Cross-track error, rudder, and UKC panels below the map

    Usage
    -----
        PlotlyRenderer.render(ship_data, metrics, save_path="dashboard.html")
    """

    @classmethod
    def render(
        cls,
        ship_data: ShipData,
        metrics: dict[str, ShipMetrics],
        save_path: str = "dashboard.html",
    ) -> None:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Run: pip install plotly")
            return

        ship_ids = list(ship_data.keys())
        n_ships = len(ship_ids)

        # 4 rows: trajectory, cross-track error, rudder, UKC
        fig = make_subplots(
            rows=4, cols=n_ships,
            subplot_titles=[
                f"{metric} – {sid}"
                for metric in ["Trajectory", "Cross-track error", "Rudder", "UKC"]
                for sid in ship_ids
            ],
            shared_xaxes=False,
        )

        for col, ship_id in enumerate(ship_ids, start=1):
            episodes = ship_data[ship_id]
            ship_color_hex = SHIP_COLORS[(col - 1) % len(SHIP_COLORS)]
            m = metrics[ship_id]

            for ep_idx, ep in enumerate(episodes):
                opacity = max(0.3, 1.0 - ep_idx * 0.12)
                name = f"{ship_id} ep{ep_idx + 1} R={ep.reward:.1f}"
                show_legend = col == 1  # avoid duplicate legend entries

                # Trajectory
                if ep.path.shape[0] >= 2:
                    fig.add_trace(go.Scatter(
                        x=ep.path[:, 0], y=ep.path[:, 1],
                        mode="lines",
                        name=name,
                        line=dict(color=ship_color_hex, width=1.5),
                        opacity=opacity,
                        legendgroup=name,
                        showlegend=show_legend,
                        hovertemplate="x=%{x:.1f}, y=%{y:.1f}<br>step=%{pointNumber}<extra>" + name + "</extra>",
                    ), row=1, col=col)

                    # Collision markers
                    for ev in ep.collision_events:
                        t = min(ev.timestep, len(ep.path) - 1)
                        fig.add_trace(go.Scatter(
                            x=[ep.path[t, 0]], y=[ep.path[t, 1]],
                            mode="markers",
                            marker=dict(
                                symbol="x" if ev.is_collision else "triangle-up",
                                size=12,
                                color="red" if ev.is_collision else "orange",
                            ),
                            name="Collision" if ev.is_collision else "Near-miss",
                            legendgroup="events",
                            showlegend=(ep_idx == 0 and col == 1),
                            hovertemplate=f"d={ev.distance:.1f}m<extra>{'collision' if ev.is_collision else 'near-miss'}</extra>",
                        ), row=1, col=col)

                # Cross-track error
                if ep.cross_errors.size:
                    t = normalized_time_axis(len(ep.cross_errors))
                    fig.add_trace(go.Scatter(
                        x=t, y=ep.cross_errors,
                        mode="lines", name=name,
                        line=dict(color=ship_color_hex, width=1),
                        opacity=opacity * 0.6,
                        legendgroup=name, showlegend=False,
                    ), row=2, col=col)

                # Rudder actions
                if ep.rudder_actions.size:
                    ct = np.linspace(0, 100, 200)
                    resampled = resample_to_common_time(ep.rudder_actions, ct / 100)
                    fig.add_trace(go.Scatter(
                        x=ct, y=resampled,
                        mode="lines", name=name,
                        line=dict(color=ship_color_hex, width=1),
                        opacity=opacity * 0.6,
                        legendgroup=name, showlegend=False,
                    ), row=3, col=col)

                # UKC
                if ep.ukc_measurements.size:
                    t = normalized_time_axis(len(ep.ukc_measurements))
                    fig.add_trace(go.Scatter(
                        x=t, y=ep.ukc_measurements,
                        mode="lines", name=name,
                        line=dict(color=ship_color_hex, width=1),
                        opacity=opacity * 0.6,
                        legendgroup=name, showlegend=False,
                    ), row=4, col=col)

        fig.update_layout(
            height=1200,
            title_text="Multi-ship RL evaluation dashboard",
            legend=dict(groupclick="toggleitem"),
        )

        fig.write_html(save_path)
        print(f"Interactive dashboard saved to '{save_path}'")


# ---------------------------------------------------------------------------
# Convenience: drop-in replacement for visualize_trained_agent's plot block
# ---------------------------------------------------------------------------

def run_visualization(
    agent,
    env,
    num_episodes: int = 5,
    save_dir: str = ".",
    ship_ids: Optional[list[str]] = None,
    near_miss_threshold: float = 50.0,
    collision_threshold: float = 15.0,
) -> tuple[ShipData, dict[str, ShipMetrics]]:
    """
    Full pipeline: collect → annotate → compute metrics → render.

    Returns (ship_data, metrics) so callers can do further analysis.

    Example
    -------
        ship_data, metrics = run_visualization(
            agent=manager.agent,
            env=manager.get_env(),
            num_episodes=5,
            save_dir=manager.agent.model_dir or ".",
        )
    """
    import os

    # 1. Collect
    collector = EpisodeCollector(agent, env, ship_ids=ship_ids)
    ship_data = collector.collect(num_episodes=num_episodes)

    # 2. Annotate collisions
    # annotator = CollisionAnnotator(
    #     near_miss_threshold=near_miss_threshold,
    #     collision_threshold=collision_threshold,
    # )
    # annotator.annotate(ship_data)

    # 3. Compute metrics
    metrics = MetricsComputer.compute(ship_data)

    # 4. Print summary
    for sid, m in metrics.items():
        print(f"\n[{sid}]  avg_R={m.avg_reward:.2f}  mTEI={m.mTEI:.2f}  MTE={m.MTE:.2f}"
              f"  mRA={m.mRA:.3f}  mHE={m.mHE:.3f} rad  mu={m.mu:.2f}")

    # 5. Render
    traj_path = os.path.join(save_dir, "trajectory.png")
    dash_path = os.path.join(save_dir, "dashboard.html")

    MatplotlibRenderer.render(ship_data, metrics, env, save_path=traj_path)
    PlotlyRenderer.render(ship_data, metrics, save_path=dash_path)

    return ship_data, metrics
