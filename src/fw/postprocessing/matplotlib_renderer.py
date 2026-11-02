from __future__ import annotations

import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely import Polygon, MultiPolygon
from fw.postprocessing.ship_episode_data import ShipData
from fw.postprocessing.metrics_computer import ShipMetrics
from fw.postprocessing.output_helpers import SHIP_COLORS, shade_color, normalized_time_axis, resample_to_common_time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matplotlib renderer
# ---------------------------------------------------------------------------


class MatplotlibRenderer:
    """
    Renders multi-ship episode data as a static matplotlib figure.

    Two figures with one column per ship: a trajectory map, and a metrics
    figure with rows for cross-track error, rudder, UKC, surge velocity and
    thrust. Episodes of a ship are shades of the ship's base color; averages
    are overlaid where applicable.

    Attributes:
        COMMON_TIME: Normalized time axis in [0, 1] for resampling signals to
            a common length before averaging.
    """

    COMMON_TIME = np.linspace(0.0, 1.0, 200)

    @classmethod
    def render(
        cls,
        ship_data: ShipData,
        metrics: dict[str, ShipMetrics],
        env,
        save_path: str = "image.png",
        fig_width_per_ship: int = 10,
        show_plots: bool = True,
    ) -> None:
        """
        Generate and save the trajectory and metrics figures.

        Args:
            ship_data: Mapping of ship IDs to lists of `ShipEpisodeData`.
            metrics: Mapping of ship IDs to computed `ShipMetrics`.
            env: Environment instance (single-ship or multi-ship).
            save_path: Base path; figures are saved as `<base>_trajectory.<ext>`
                and `<base>_metrics.<ext>`.
            fig_width_per_ship: Width (in inches) allocated per ship column.
            show_plots: Whether to show the figures (blocking); otherwise
                they are closed.

        Raises:
            ValueError: If fig_width_per_ship is not positive.

        Ships without episode data are skipped.
        """

        if fig_width_per_ship <= 0:
            raise ValueError(
                f"fig_width_per_ship must be positive, got {fig_width_per_ship}."
            )

        ship_ids = [sid for sid in ship_data if ship_data[sid]]
        n_ships = len(ship_ids)
        # Metrics rows: cross-track error, rudder, UKC, velocity, thrust
        n_rows = 5
        if n_ships == 0:
            logger.warning("No episode data to render - skipping.")
            return

        try:
            # --- Trajectory figure (one window) ---
            fig_traj, axes_traj = plt.subplots(
                1, n_ships,
                figsize=(fig_width_per_ship * n_ships, 6),
                squeeze=False,
            )

            for col, ship_id in enumerate(ship_ids):
                episodes = ship_data[ship_id]
                ship_color = SHIP_COLORS[col % len(SHIP_COLORS)]
                colors = shade_color(ship_color, len(episodes))
                m = metrics[ship_id]

                ax_traj = axes_traj[0][col]

                ship_env = cls._get_ship_env(env, ship_id)

                if ship_env is None:
                    warnings.warn(
                        f"[MatplotlibRenderer] Could not resolve environment for ship '{ship_id}'. "
                        f"Trajectory and obstacle overlays will be skipped.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    # Add hatched pattern to first subplot
                    difference_coords = cls.create_hashed_area(ship_env.obstacles, ship_env.overall)
                    for i, coords in enumerate(difference_coords):
                        ax_traj.add_patch(patches.Polygon(
                            coords,
                            facecolor='none',
                            edgecolor='gray',
                            hatch='///',
                            alpha=0.3,
                            label='Low water level area' if i == 0 else "",
                        ))

                cls._plot_trajectory(ax_traj, episodes, colors, ship_env, ship_id, m)

            plt.tight_layout()
            # Save trajectory figure
            base, ext = (save_path.rsplit('.', 1) + ["png"])[:2]
            traj_path = f"{base}_trajectory.{ext}"
            plt.savefig(traj_path, bbox_inches="tight", dpi=300)
            logger.info("Trajectory figure saved to '%s'", traj_path)

            # --- Metrics figure (separate window) ---
            fig_metrics, axes = plt.subplots(
                n_rows, n_ships,
                figsize=(fig_width_per_ship * n_ships, 16),
                squeeze=False,
            )

            for col, ship_id in enumerate(ship_ids):
                episodes = ship_data[ship_id]
                ship_color = SHIP_COLORS[col % len(SHIP_COLORS)]
                colors = shade_color(ship_color, len(episodes))

                ax_error  = axes[0][col]
                ax_rudder = axes[1][col]
                ax_ukc    = axes[2][col]
                ax_vel    = axes[3][col]
                ax_thrust = axes[4][col]

                # Cross-track error
                cls._plot_signal(
                    ax_error, episodes, colors,
                    signal_attr="cross_errors",
                    title=f"Cross-track error – {ship_id}",
                    ylabel="Cross-track error",
                    ylim=(0, 45),
                )

                # Rudder
                cls._plot_rudder(ax_rudder, episodes, colors, ship_id)

                # UKC
                cls._plot_signal(
                    ax_ukc, episodes, colors,
                    signal_attr="ukc_measurements",
                    title=f"UKC – {ship_id}",
                    ylabel="UKC",
                    ylim=(0, 45),
                )

                # Velocity
                cls._plot_signal(
                    ax_vel, episodes, colors,
                    signal_attr="velocities",
                    title=f"Surge velocity – {ship_id}",
                    ylabel="Velocity",
                    ylim=None,
                )

                # Thrust
                cls._plot_signal(
                    ax_thrust, episodes, colors,
                    signal_attr="thrust_actions",
                    title=f"Thrust – {ship_id}",
                    ylabel="Thrust",
                    ylim=None,
                )

            plt.tight_layout()
            metrics_path = f"{base}_metrics.{ext}"
            plt.savefig(metrics_path, bbox_inches="tight", dpi=300)
            logger.info("Metrics figure saved to '%s'", metrics_path)

            if show_plots:
                plt.show()
        finally:
            # With show_plots the windows stay open for viewing.
            if not show_plots:
                plt.close("all")

    @classmethod
    def _get_ship_env(cls, env, ship_id: str):
        """
        Return the environment of a specific ship.

        Args:
            env: Environment instance, possibly containing sub-environments.
            ship_id: Identifier of the ship.

        Returns:
            The sub-environment for the given ship in multi-ship mode,
            otherwise the main environment.
        """

        if hasattr(env, "sub_environments") and env.sub_environments:
            return env.sub_environments.get(ship_id)
        return env

    # ------------------------------------------------------------------
    # Sub-plot helpers
    # ------------------------------------------------------------------

    @classmethod
    def create_hashed_area(cls, obstacles, overall) -> list[np.ndarray]:
        """
        Return the area between the overall boundary and the navigable
        polygon (set difference), for hatching.

        Args:
            obstacles: Sequence of (x, y) coordinates defining the navigable boundary.
            overall: Sequence of (x, y) coordinates defining the outer boundary
                (e.g., Western Scheldt).

        Returns:
            Exterior rings of the difference area, each of shape (N, 2);
            empty if the difference is empty.
        """

        obstacle_poly = Polygon(obstacles)
        overall_poly = Polygon(overall)
        difference = overall_poly.difference(obstacle_poly)

        if difference.is_empty:
            return []

        if isinstance(difference, MultiPolygon):
            return [np.array(poly.exterior.coords) for poly in difference.geoms]
        return [np.array(difference.exterior.coords)]

    @classmethod
    def _plot_trajectory(cls, ax, episodes, colors, env, ship_id, metrics) -> None:
        """
        Plot the trajectory map of one ship: waterway geometry, checkpoints,
        episode paths and start/end markers.

        Args:
            ax: Matplotlib axis to draw on.
            episodes: List of `ShipEpisodeData` for the ship.
            colors: List of colors corresponding to episodes.
            env: Ship-specific environment instance. May be None.
            ship_id: Identifier of the ship.
            metrics: `ShipMetrics` instance for summary statistics.

        Episodes with fewer than two path points are skipped; with env None the
        geometry overlays are skipped.
        """

        if env is not None:
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

            if hasattr(env, "checkpoints"):
                cp_pos = np.array([cp["pos"] for cp in env.checkpoints])
                ax.plot(cp_pos[:, 0], cp_pos[:, 1], "g--", alpha=0.5, label="Ideal path")
                for cp in env.checkpoints:
                    ax.add_patch(plt.Circle(
                        cp["pos"], radius=10, color="gray", alpha=0.5, fill=True,
                    ))

        for i, ep in enumerate(episodes):
            if ep.path.shape[0] < 2:
                continue

            ax.plot(
                ep.path[:, 0], ep.path[:, 1],
                color=colors[i], alpha=0.5,
                label=f"Ep {i + 1} (R={ep.reward:.1f})",
            )

            # Start marker: blue dot.
            ax.scatter(*ep.path[0], c="blue", s=80, zorder=5)

            # End marker: X for env-reported collisions (magenta for ship),
            # a small gray dot for any other ending, so an ending near a bank
            # is not mistaken for a bank crash.
            if ep.collision:
                ctype = (ep.collision_type or "").lower()
                if ctype == "bank":
                    end_color, end_marker, end_size = "red", "X", 130
                elif ctype == "ship":
                    end_color, end_marker, end_size = "magenta", "X", 130
                else:
                    end_color, end_marker, end_size = "red", "X", 130
            else:
                end_color, end_marker, end_size = "dimgray", "o", 60
            ax.scatter(
                *ep.path[-1], c=end_color, marker=end_marker,
                s=end_size, zorder=6, edgecolors="black", linewidths=0.6,
            )

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
    def _plot_signal(
        cls,
        ax,
        episodes,
        colors,
        signal_attr: str,
        title: str,
        ylabel: str,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """
        Plot a time-series signal per episode, plus the average after
        resampling to a common time axis.

        Args:
            ax: Matplotlib axis to draw on.
            episodes: List of `ShipEpisodeData`.
            colors: List of colors corresponding to episodes.
            signal_attr: Name of the attribute to extract from each episode.
            title: Plot title.
            ylabel: Label for the y-axis.
            ylim: Optional tuple specifying y-axis limits.

        The x-axis is episode progress [0, 100] %; the average is drawn only if
        at least one episode has data.
        """

        all_signals = []

        for i, ep in enumerate(episodes):
            signal = getattr(ep, signal_attr)
            if signal.size == 0:
                continue
            t = normalized_time_axis(len(signal))
            ax.plot(t, signal, color=colors[i], alpha=0.2)
            all_signals.append(signal)

        if all_signals:
            # resample_to_common_time expects common_time in [0, 1]
            resampled = [resample_to_common_time(s, cls.COMMON_TIME) for s in all_signals]
            avg = np.mean(resampled, axis=0)
            # Scale COMMON_TIME to [0, 100] like normalized_time_axis
            t_common = cls.COMMON_TIME * 100.0
            ax.plot(t_common, avg, color="black", alpha=0.7, lw=1.5, label="Average")

        ax.set_title(title)
        ax.set_xlabel("Episode progress [%]")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.axhline(ylim[0], color="k", lw=0.8, ls="--", alpha=0.4)
            ax.axhline(ylim[1], color="k", lw=0.8, ls="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True)

    @classmethod
    def _plot_rudder(cls, ax, episodes, colors, ship_id: str) -> None:
        """
        Plot rudder actions per episode over normalized episode time.

        Args:
            ax: Matplotlib axis to draw on.
            episodes: List of `ShipEpisodeData`.
            colors: List of colors corresponding to episodes.
            ship_id: Identifier of the ship.

        Episodes without rudder data are skipped; no average is drawn.
        """

        for i, ep in enumerate(episodes):
            if ep.rudder_actions.size == 0:
                continue
            # resampled = resample_to_common_time(ep.rudder_actions, cls.COMMON_TIME)
            # ax.plot(cls.COMMON_TIME * 100.0, resampled, color=colors[i], alpha=0.2)
            t = normalized_time_axis(len(ep.rudder_actions))
            ax.plot(t, ep.rudder_actions, color=colors[i], alpha=0.2)

        ax.set_title(f"Rudder actions – {ship_id}")
        ax.set_xlabel("Episode progress [%]")
        ax.set_ylabel("Rudder action")
        ax.set_xlim(0.0, 100.0)
        ax.grid(True)
