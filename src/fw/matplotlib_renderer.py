from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fw.ship_episode_data import ShipData
from fw.metrics_computer import ShipMetrics
from fw.output_helpers import SHIP_COLORS, shade_color, normalized_time_axis, resample_to_common_time


# ---------------------------------------------------------------------------
# Matplotlib renderer
# ---------------------------------------------------------------------------


class MatplotlibRenderer:
    """Renders multi-ship episode data as a static matplotlib figure.

    Produces a grid of subplots where each column corresponds to a ship and
    each row visualizes a different aspect of the episode data:
        - Row 0: Trajectory map
        - Row 1: Cross-track error over time
        - Row 2: Rudder actions over time
        - Row 3: Under-keel clearance (UKC) over time

    Episodes for a given ship are shown as color variations of a base ship color.
    Aggregate (average) signals are overlaid where applicable.

    Attributes:
        COMMON_TIME: Normalized time axis used for resampling signals to a
            common length for averaging.
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
        """Generates and saves a multi-ship trajectory visualization.

        Args:
            ship_data: Mapping of ship IDs to lists of `ShipEpisodeData`.
            metrics: Mapping of ship IDs to computed `ShipMetrics`.
            env: Environment instance (single-ship or multi-ship).
            save_path: File path where the generated figure is saved.
            fig_width_per_ship: Width (in inches) allocated per ship column.

        Returns:
            None.

        Notes:
            - Ships without episode data are skipped.
            - The figure is displayed and saved to disk.
            - All matplotlib figures are closed after rendering to avoid leaks.
        """

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
        """Retrieves the environment corresponding to a specific ship.

        Args:
            env: Environment instance, possibly containing sub-environments.
            ship_id: Identifier of the ship.

        Returns:
            The sub-environment for the given ship in multi-ship mode,
            otherwise the main environment.
        """

        if hasattr(env, "sub_environments") and env.sub_environments:
            return env.sub_environments.get(ship_id)
        return env  # single-ship: env IS the ship env

    # ------------------------------------------------------------------
    # Sub-plot helpers
    # ------------------------------------------------------------------

    @classmethod
    def _plot_trajectory(cls, ax, episodes, colors, env, ship_id, metrics):
        """Plots the trajectory map for a single ship.

        Includes waterway geometry, checkpoints, episode paths, and start/end
        markers.

        Args:
            ax: Matplotlib axis to draw on.
            episodes: List of `ShipEpisodeData` for the ship.
            colors: List of colors corresponding to episodes.
            env: Ship-specific environment instance.
            ship_id: Identifier of the ship.
            metrics: `ShipMetrics` instance for summary statistics.

        Notes:
            - Only episodes with at least two path points are plotted.
            - Collision visualization hooks are present but currently disabled.
        """

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
        """Plots a generic time-series signal across episodes.

        Each episode is plotted individually, along with an averaged signal
        computed after resampling to a common time axis.

        Args:
            ax: Matplotlib axis to draw on.
            episodes: List of `ShipEpisodeData`.
            colors: List of colors corresponding to episodes.
            signal_attr: Name of the attribute to extract from each episode.
            title: Plot title.
            ylabel: Label for the y-axis.
            ylim: Optional tuple specifying y-axis limits.

        Returns:
            None.

        Notes:
            - Signals are normalized to episode progress [0, 1].
            - Average signal is computed only if at least one valid signal exists.
        """

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
        """Plots rudder actions over normalized episode time.

        Rudder signals are resampled to a common time axis to allow comparison
        across episodes.

        Args:
            ax: Matplotlib axis to draw on.
            episodes: List of `ShipEpisodeData`.
            colors: List of colors corresponding to episodes.
            ship_id: Identifier of the ship.

        Returns:
            None.

        Notes:
            - Episodes without rudder data are skipped.
            - No average curve is plotted for rudder actions.
        """

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
