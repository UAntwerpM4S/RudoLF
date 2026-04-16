from __future__ import annotations

from typing import Optional
from fw.plotly_renderer import PlotlyRenderer
from fw.episode_collector import EpisodeCollector
from fw.matplotlib_renderer import MatplotlibRenderer
from fw.metrics_computer import ShipMetrics, MetricsComputer
from fw.ship_episode_data import RunData, CollisionAnnotator


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
) -> tuple[RunData, dict[str, ShipMetrics]]:
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
    run_data = collector.collect(num_episodes=num_episodes)

    # 2. Annotate collisions
    annotator = CollisionAnnotator(
        near_miss_threshold=near_miss_threshold,
        collision_threshold=collision_threshold,
    )
    annotator.annotate(run_data)

    # 3. Compute metrics
    metrics = MetricsComputer.compute(run_data)

    # 4. Print summary
    for sid, m in metrics.items():
        print(f"\n[{sid}]  avg_R={m.avg_reward:.2f}  mTEI={m.mTEI:.2f}  MTE={m.MTE:.2f}"
              f"  mRA={m.mRA:.3f}  mHE={m.mHE:.3f} rad  mu={m.mu:.2f}")

    # 5. Render
    traj_path = os.path.join(save_dir, "trajectory.png")
    dash_path = os.path.join(save_dir, "dashboard.html")

    MatplotlibRenderer.render(run_data.ships, metrics, env, save_path=traj_path)
    PlotlyRenderer.render(run_data.ships, metrics, save_path=dash_path)

    return run_data, metrics
