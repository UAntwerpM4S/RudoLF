from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import patches

from fw.postprocessing.ship_episode_data import ShipData
from fw.postprocessing.output_helpers import SHIP_COLORS

logger = logging.getLogger(__name__)


class AnimationRenderer:
    """
    Render recorded episode trajectories to an animated GIF without using a GUI backend.
    """

    @staticmethod
    def _get_ship_env(env, ship_id: str):
        if hasattr(env, "sub_environments") and env.sub_environments:
            return env.sub_environments.get(ship_id)
        return env

    @classmethod
    def render_gif(
        cls,
        ship_data: ShipData,
        env,
        save_path: str,
        max_frames: int = 240,
        duration_ms: int = 60,
    ) -> None:
        """
        Save a lightweight trajectory animation as a GIF.
        """

        episodes = []
        for ship_id, ship_episodes in ship_data.items():
            if ship_episodes and len(ship_episodes[0].path) > 0:
                episodes.append((ship_id, ship_episodes[0]))

        if not episodes:
            logger.warning("No episode data to animate - skipping GIF export.")
            return

        max_len = max(len(ep.path) for _, ep in episodes)
        if max_len <= 1:
            logger.warning("Episode is too short to animate - skipping GIF export.")
            return

        frame_indices = np.linspace(0, max_len - 1, min(max_frames, max_len)).astype(int)
        frames: list[Image.Image] = []

        fig = Figure(figsize=(10, 7), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        for frame_idx in frame_indices:
            ax.clear()

            first_env = cls._get_ship_env(env, episodes[0][0])
            if first_env is not None:
                if hasattr(first_env, "overall"):
                    ax.add_patch(
                        patches.Polygon(
                            first_env.overall,
                            closed=True,
                            edgecolor="brown",
                            facecolor="none",
                            linewidth=1.5,
                        )
                    )
                if hasattr(first_env, "obstacles"):
                    ax.add_patch(
                        patches.Polygon(
                            first_env.obstacles,
                            closed=True,
                            edgecolor="red",
                            facecolor="none",
                            linewidth=1.5,
                        )
                    )
                if hasattr(first_env, "checkpoints"):
                    checkpoint_points = np.asarray([cp["pos"] for cp in first_env.checkpoints])
                    if checkpoint_points.size:
                        ax.plot(
                            checkpoint_points[:, 0],
                            checkpoint_points[:, 1],
                            "g--",
                            linewidth=1.0,
                            alpha=0.7,
                        )

            for color_idx, (ship_id, episode) in enumerate(episodes):
                path = episode.path
                last_idx = min(frame_idx, len(path) - 1)
                visible_path = path[: last_idx + 1]
                color = SHIP_COLORS[color_idx % len(SHIP_COLORS)]

                ax.plot(visible_path[:, 0], visible_path[:, 1], color=color, linewidth=2)
                ax.scatter(
                    [visible_path[-1, 0]],
                    [visible_path[-1, 1]],
                    color=color,
                    s=50,
                    label=ship_id,
                )

            ax.set_xlim(-11700, 14500)
            ax.set_ylim(-11700, 14500)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Ship Path Following")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            fig.tight_layout()

            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        logger.info("Animation GIF saved to '%s'", output_path)
