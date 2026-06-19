import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional, Set, Tuple, Any


class RRTCollisionVisualizer:
    """
    Handles visualization of collision entities and RRT replanning overlays
    on trajectory plots.
    
    This class encapsulates the logic for drawing:
    - Static and dynamic obstacle spawn positions and tracks
    - RRT danger/search radius zones around obstacles
    - RRT replanned paths and checkpoints (when enabled)
    """
    
    def __init__(self, env: Any):
        """
        Initialize visualizer with environment data.
        
        Args:
            env: Environment instance (PySimColavPlannerEnv) containing:
                - collision_entities: list of collision entity objects
                - rrt_replanner or rrt_apf_replanner: RRT planner instance
                - checkpoints: checkpoint data for extracting positions
                - detection_radius: radius for obstacle detection
        """
        self.env = env
        
        # Extract collision entities from the environment
        self.collision_entities = getattr(env, 'collision_entities', [])
        
        # Get replanner (try both replanner types)
        self.replanner = getattr(env, 'replanner', None)
        
        # Control whether to draw RRT replanned paths
        self.draw_rrt_paths = getattr(env, 'draw_rrt_paths', False)
        
        # Extract checkpoint positions if available
        self.checkpoint_positions_2d = None
        if hasattr(env, 'checkpoints') and env.checkpoints:
            try:
                cp_pos = np.array([cp["pos"] for cp in env.checkpoints])
                self.checkpoint_positions_2d = cp_pos[:, :2]  # Take only x, y
            except (KeyError, IndexError, TypeError):
                pass
        
        # Initialize label tracking flags
        self.static_label_used = False
        self.dynamic_label_used = False
        self.dynamic_track_label_used = False
        self.danger_label_used = False
        self.search_inner_label_used = False
        self.search_outer_label_used = False
        self.rrt_path_label_used = False
        self.rrt_checkpoint_label_used = False
        self.rrt_overlay_drawn_keys: Set[Tuple[float, float, float]] = set()
    
    def plot_on_axis(self, ax: plt.Axes) -> None:
        """
        Plot collision entities and RRT overlays on the given axis.
        
        Args:
            ax: Matplotlib axis to draw on.
        """
        if not self.collision_entities:
            return
        
        for entity in self.collision_entities:
            # Get entity position and properties
            try:
                position = entity.get_position()
                if position is None or len(position) < 2:
                    continue
                
                spawn_pos = np.array(position[:2])
                radius = float(getattr(entity, 'radius', 0.0))
                is_dynamic = bool(getattr(entity, 'is_dynamic', False))
                
            except (AttributeError, TypeError, ValueError):
                continue
            
            edge_color = "tab:red"
            
            # Draw obstacle spawn circle
            circle_label = self._get_circle_label(is_dynamic)
            ax.add_patch(
                patches.Circle(
                    (spawn_pos[0], spawn_pos[1]),
                    radius,
                    fill=True,
                    color=edge_color,
                    linestyle="--" if is_dynamic else "-",
                    linewidth=1.2,
                    alpha=0.6,
                    label=circle_label,
                )
            )
            
            # Draw RRT planning overlays if planner is available
            if self.replanner is not None:
                self._plot_rrt_overlays(
                    ax=ax,
                    spawn_pos=spawn_pos,
                    radius=radius,
                )
            
            # Draw dynamic obstacle track if it has moved
            if is_dynamic and hasattr(entity, 'trajectory') and entity.trajectory:
                try:
                    positions = np.array(entity.trajectory)
                    if len(positions) > 1:
                        path_label = None
                        if not self.dynamic_track_label_used:
                            path_label = "Dynamic obstacle track"
                            self.dynamic_track_label_used = True
                        
                        ax.plot(
                            positions[:, 0],
                            positions[:, 1],
                            color=edge_color,
                            linewidth=1.0,
                            alpha=0.45,
                            linestyle=":",
                            label=path_label,
                        )
                except (AttributeError, TypeError, ValueError, IndexError):
                    pass
    
    def _get_circle_label(self, is_dynamic: bool) -> Optional[str]:
        """Get label for obstacle circle (only first occurrence)."""
        if is_dynamic and not self.dynamic_label_used:
            self.dynamic_label_used = True
            return "Dynamic obstacle (spawn)"
        elif not is_dynamic and not self.static_label_used:
            self.static_label_used = True
            return "Static obstacle"
        return None
    
    def _plot_rrt_overlays(
        self,
        ax: plt.Axes,
        spawn_pos: np.ndarray,
        radius: float,
    ) -> None:
        """
        Draw RRT danger/search radii and optionally replanned paths.
        
        Ensures each unique obstacle location is only drawn once by tracking
        a set of (x, y, radius) keys.
        """
        obstacle_center = np.asarray(spawn_pos[:2], dtype=float)
        effective_danger_radius = float(radius) + float(self.replanner.danger_radius)
        
        # For search radii, check if planner has margin attributes
        inner_margin = float(getattr(self.replanner, 'inner_margin', 50.0))
        outer_margin = float(getattr(self.replanner, 'outer_margin', 100.0))
        
        search_inner_radius = effective_danger_radius + inner_margin
        search_outer_radius = search_inner_radius + outer_margin
        
        # Create overlay key to avoid duplicate drawings
        overlay_key = (
            round(float(obstacle_center[0]), 2),
            round(float(obstacle_center[1]), 2),
            round(float(radius), 2),
        )
        
        if overlay_key in self.rrt_overlay_drawn_keys:
            return
        
        self.rrt_overlay_drawn_keys.add(overlay_key)
        
        # Draw danger radius
        danger_label = None
        if not self.danger_label_used:
            danger_label = "RRT danger radius"
            self.danger_label_used = True
        
        ax.add_patch(
            patches.Circle(
                (obstacle_center[0], obstacle_center[1]),
                effective_danger_radius,
                fill=False,
                edgecolor="tab:red",
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
                label=danger_label,
            )
        )
        
        # Draw search inner radius
        search_inner_label = None
        if not self.search_inner_label_used:
            search_inner_label = "RRT search inner"
            self.search_inner_label_used = True
        
        ax.add_patch(
            patches.Circle(
                (obstacle_center[0], obstacle_center[1]),
                search_inner_radius,
                fill=False,
                edgecolor="tab:orange",
                linestyle="--",
                linewidth=1.2,
                alpha=0.65,
                label=search_inner_label,
            )
        )
        
        # Draw search outer radius
        search_outer_label = None
        if not self.search_outer_label_used:
            search_outer_label = "RRT search outer"
            self.search_outer_label_used = True
        
        ax.add_patch(
            patches.Circle(
                (obstacle_center[0], obstacle_center[1]),
                search_outer_radius,
                fill=False,
                edgecolor="tab:orange",
                linestyle=":",
                linewidth=1.3,
                alpha=0.65,
                label=search_outer_label,
            )
        )
        
        # Draw RRT replanned paths (optional)
        if self.draw_rrt_paths and self.checkpoint_positions_2d is not None:
            self._plot_rrt_path(
                ax=ax,
                obstacle_center=obstacle_center,
                radius=radius,
            )
    
    def _plot_rrt_path(
        self,
        ax: plt.Axes,
        obstacle_center: np.ndarray,
        radius: float,
    ) -> None:
        """
        Attempt to compute and draw RRT replanned path around an obstacle.
        
        This method is included for completeness but was commented out in the
        original code. Enable by setting env.draw_rrt_paths = True.
        """
        try:
            replan_result = self.replanner.replan(
                obstacle_center=obstacle_center,
                obstacle_radius=radius,
                ship_trajectory=self.checkpoint_positions_2d,
            )
            
            if replan_result.success and len(replan_result.rrt_path) >= 2:
                rrt_path_xy = np.asarray(replan_result.rrt_path, dtype=float)
                
                # Draw replanned path
                rrt_label = None
                if not self.rrt_path_label_used:
                    rrt_label = "RRT replanned path"
                    self.rrt_path_label_used = True
                
                ax.plot(
                    rrt_path_xy[:, 0],
                    rrt_path_xy[:, 1],
                    color="magenta",
                    linewidth=2.0,
                    alpha=0.8,
                    linestyle="--",
                    label=rrt_label,
                )
                
                # Draw RRT checkpoints
                rrt_checkpoint_label = None
                if not self.rrt_checkpoint_label_used:
                    rrt_checkpoint_label = "RRT checkpoints"
                    self.rrt_checkpoint_label_used = True
                
                ax.scatter(
                    rrt_path_xy[:, 0],
                    rrt_path_xy[:, 1],
                    c="magenta",
                    s=28,
                    alpha=0.85,
                    marker="x",
                    label=rrt_checkpoint_label,
                )
        except Exception:
            # Silently fail if RRT replanning doesn't work for this obstacle
            pass