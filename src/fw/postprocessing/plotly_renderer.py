from __future__ import annotations

import numpy as np

from fw.postprocessing.ship_episode_data import ShipData
from fw.postprocessing.metrics_computer import ShipMetrics
from fw.postprocessing.output_helpers import SHIP_COLORS, normalized_time_axis, resample_to_common_time


# ---------------------------------------------------------------------------
# Plotly renderer (interactive HTML dashboard)
# ---------------------------------------------------------------------------

class PlotlyRenderer:
    """
    Renders an interactive multi-ship evaluation dashboard using Plotly.

    Produces an HTML file containing an interactive visualization of ship
    trajectories and associated time-series signals. The dashboard supports
    zooming, hovering, and toggling visibility of individual ships and episodes.

    Layout:
        Each ship is shown in a separate column with four stacked panels:
            1. Trajectory (with optional collision markers)
            2. Cross-track error
            3. Rudder actions
            4. Under-keel clearance (UKC)

    Requirements:
        plotly (pip install plotly)
    """

    @classmethod
    def render(
        cls,
        ship_data: ShipData,
        metrics: dict[str, ShipMetrics],
        save_path: str = "dashboard.html",
    ) -> None:
        """
        Generates an interactive HTML dashboard for multi-ship evaluation.

        Args:
            ship_data: Mapping of ship IDs to lists of `ShipEpisodeData`.
            metrics: Mapping of ship IDs to computed `ShipMetrics` (currently
                not directly visualized but reserved for future extensions).
            save_path: Output path for the generated HTML file.

        Returns:
            None.

        Notes:
            - Ships are assigned colors from a fixed palette (`SHIP_COLORS`).
            - Each episode is rendered with decreasing opacity for readability.
            - Collision and near-miss events are shown on the trajectory plot.
            - Legend grouping allows toggling full ship trajectories.
            - Requires Plotly; gracefully exits if not installed.
        """

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not installed. Run: pip install plotly")
            return

        ship_ids = list(ship_data.keys())
        n_ships = len(ship_ids)

        if n_ships == 0:
            print("[PlotlyRenderer] No ship data to render — skipping.")
            return

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
            if not episodes:
                continue

            ship_color_hex = SHIP_COLORS[(col - 1) % len(SHIP_COLORS)]

            for ep_idx, ep in enumerate(episodes):
                opacity = max(0.3, 1.0 - ep_idx * 0.12)
                name = f"{ship_id} ep{ep_idx + 1} R={ep.reward:.1f}"
                show_legend = ep_idx == 0

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
                    # resample_to_common_time expects common_time in [0, 1];
                    # divide ct by 100 so the x-axis stays in [0, 100] for display.
                    ct = np.linspace(0.0, 100.0, 200)
                    resampled = resample_to_common_time(ep.rudder_actions, ct / 100.0)
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
