"""Live matplotlib minimap for Fossen-AirSim visualisation.

Mirrors the same map that ``AgentManager.render()`` draws but updates the
ship marker in real time, once per Fossen step, driven by the same clock as
the physics simulation.

The minimap runs in a dedicated child process so that GUI events and full
redraws never block the Fossen/AirSim main loop or the UE5 RPC cadence. On Windows,
the minimap also attempts to open on the same monitor as UE5, dock itself
near the top-right of that monitor's work area, and keep a local follow-view
around the ship instead of showing the entire map at all times.

Usage (embedded - called from run_visualise_fossen)::

    from airsim.inference.minimap_fossen_airsim import MinimapFossenAirsim

    minimap = MinimapFossenAirsim(env)  # once, after env.reset()
    minimap.update(env)                 # each step, after env.step()
    minimap.finish(total_reward)        # at episode end - saves PNG
    minimap.close()                     # on teardown
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _find_ue5_window_context():
    """Return UE5 window and monitor work-area geometry on Windows.

    Returns:
        ``None`` when UE5 is not found or the platform is not Windows.
        Otherwise returns a dictionary containing:
        - ``window_rect``: ``(x, y, w, h)`` of the UE5 window.
        - ``work_rect``: ``(x, y, w, h)`` of the monitor work area containing
          the UE5 window.
    """
    try:
        import platform
        if platform.system() != "Windows":
            return None
        import ctypes
        import ctypes.wintypes

        class _MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.wintypes.DWORD),
                ("rcMonitor", ctypes.wintypes.RECT),
                ("rcWork", ctypes.wintypes.RECT),
                ("dwFlags", ctypes.wintypes.DWORD),
            ]

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW("UnrealWindow", None)
        if not hwnd:
            return None

        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None

        window_x = int(rect.left)
        window_y = int(rect.top)
        window_w = int(rect.right - rect.left)
        window_h = int(rect.bottom - rect.top)
        if window_w <= 0 or window_h <= 0:
            return None

        monitor = user32.MonitorFromWindow(hwnd, 2)
        if not monitor:
            return {
                "window_rect": (window_x, window_y, window_w, window_h),
                "work_rect": (window_x, window_y, window_w, window_h),
            }

        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return {
                "window_rect": (window_x, window_y, window_w, window_h),
                "work_rect": (window_x, window_y, window_w, window_h),
            }

        work_x = int(info.rcWork.left)
        work_y = int(info.rcWork.top)
        work_w = int(info.rcWork.right - info.rcWork.left)
        work_h = int(info.rcWork.bottom - info.rcWork.top)
        if work_w <= 0 or work_h <= 0:
            work_x, work_y, work_w, work_h = window_x, window_y, window_w, window_h

        return {
            "window_rect": (window_x, window_y, window_w, window_h),
            "work_rect": (work_x, work_y, work_w, work_h),
        }
    except Exception:
        return None


def _find_ue5_rect():
    """Return (x, y, w, h) of the UE5 window on Windows, or None.

    Uses the Win32 class name ``UnrealWindow`` which all UE4/UE5 instances
    register automatically.  Pure-ctypes, no extra dependencies.  Returns
    ``None`` on non-Windows platforms or when no UE5 window is found.
    """
    ctx = _find_ue5_window_context()
    if ctx is None:
        return None
    return ctx["window_rect"]


def _resolve_minimap_pixels(ue5_context) -> int:
    """Choose a square minimap size from the UE5 window/monitor geometry."""
    if ue5_context is None:
        return 700

    _, _, window_w, window_h = ue5_context["window_rect"]
    _, _, work_w, work_h = ue5_context["work_rect"]
    size = int(min(window_h * 0.30, window_w * 0.24, work_h * 0.32, work_w * 0.24))
    return max(360, min(size, 760))


def _resolve_minimap_position(ue5_context, minimap_px: int, inset: int = 16) -> Optional[tuple[int, int]]:
    """Return the top-right dock position for the minimap on the UE5 monitor."""
    if ue5_context is None:
        return None

    work_x, work_y, work_w, _ = ue5_context["work_rect"]
    mm_x = work_x + work_w - minimap_px - inset
    mm_y = work_y + inset
    return max(work_x, mm_x), max(work_y, mm_y)


def _estimate_view_half_span(snap: dict) -> float:
    """Estimate a local follow-camera span from the waypoint geometry."""
    checkpoints = snap.get("checkpoints") or []
    if len(checkpoints) >= 2:
        cp_pos = np.array([cp["pos"] for cp in checkpoints], dtype=np.float64)
        deltas = np.linalg.norm(np.diff(cp_pos, axis=0), axis=1)
        deltas = deltas[deltas > 1.0]
        if deltas.size > 0:
            return float(np.clip(np.median(deltas) * 2.0, 850.0, 2600.0))

    ship_pos = np.array(snap["ship_pos_0"], dtype=np.float64)
    target_pos = snap.get("target_pos")
    if target_pos is not None:
        distance = float(np.linalg.norm(np.array(target_pos, dtype=np.float64) - ship_pos))
        if distance > 1.0:
            return float(np.clip(distance / 9.0, 850.0, 2600.0))

    return 1050.0


def _preferred_view_center(pos: np.ndarray, checkpoints, half_span: float) -> np.ndarray:
    """Center the viewport slightly ahead of the ship along the route."""
    pos = np.array(pos, dtype=np.float64)
    if not checkpoints:
        return pos

    cp_pos = np.array([cp["pos"] for cp in checkpoints], dtype=np.float64)
    distances = np.linalg.norm(cp_pos - pos, axis=1)
    nearest_idx = int(np.argmin(distances))
    lookahead_idx = min(nearest_idx + 2, len(cp_pos) - 1)
    lookahead = cp_pos[lookahead_idx]

    direction = lookahead - pos
    distance = float(np.linalg.norm(direction))
    if distance <= 1.0:
        return pos

    forward_weight = min(0.40, max(0.18, distance / max(half_span * 2.0, 1.0) * 0.35))
    return pos + direction * forward_weight


def _should_shift_view(pos: np.ndarray, center: np.ndarray, half_span: float, safe_ratio: float = 0.58) -> bool:
    """Return True when the ship is near the viewport edge or outside it."""
    safe_half = half_span * safe_ratio
    return bool(np.any(np.abs(np.array(pos, dtype=np.float64) - np.array(center, dtype=np.float64)) > safe_half))


def _apply_view_limits(ax, center: np.ndarray, half_span: float) -> None:
    """Apply a square local viewport around the current follow center."""
    ax.set_xlim(float(center[0] - half_span), float(center[0] + half_span))
    ax.set_ylim(float(center[1] - half_span), float(center[1] + half_span))


def _try_put_latest(message_queue, item) -> None:
    """Push an item into a latest-wins queue without blocking the caller."""
    try:
        message_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        message_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        message_queue.put_nowait(item)
    except queue.Full:
        pass


def _drain_latest_update(message_queue):
    """Return the newest queued update message, discarding older ones."""
    latest = None
    while True:
        try:
            latest = message_queue.get_nowait()
        except queue.Empty:
            return latest


def _snapshot_env(env) -> dict:
    """Extract static env data as plain Python/NumPy objects.

    Called from the main thread before the worker is spawned so the
    snapshot is safe to consume on the worker thread without sharing
    mutable env state across thread boundaries.

    Args:
        env: A fully reset Fossen environment.

    Returns:
        Dictionary of serialisable static map data.
    """
    return {
        "ship_pos_0": np.array(env.ship_pos[0], dtype=np.float64),
        "initial_heading": float(env.phys_sims[0].state.heading),
        "target_pos": (
            np.array(env.target_pos, dtype=np.float64)
            if getattr(env, "target_pos", None) is not None
            else None
        ),
        "checkpoints": [
            {
                "pos": [float(env.checkpoints[i]["pos"][0]), float(env.checkpoints[i]["pos"][1])],
                "radius": float(env.checkpoints[i].get("radius", 10)),
            }
            for i in range(len(env.checkpoints))
        ] if hasattr(env, "checkpoints") and env.checkpoints else [],
        "obstacles": (
            np.array(env.obstacles, dtype=np.float64)
            if hasattr(env, "obstacles") and env.obstacles is not None
            else None
        ),
        "overall": (
            np.array(env.overall, dtype=np.float64)
            if hasattr(env, "overall") and env.overall is not None
            else None
        ),
    }


def _worker_save_png(
    fig,
    ax,
    paths: List[np.ndarray],
    rewards: List[float],
    save_dir: "Path",
    episode: int,
) -> None:
    """Draw all completed episode trails and write a PNG.

    Runs on the minimap-worker thread.  All trail artists are removed after
    saving so the blit background stays clean for the next episode.

    Args:
        fig: The matplotlib Figure.
        ax: The Axes to draw on.
        paths: List of (N, 2) position arrays, one per completed episode.
        rewards: Corresponding per-episode total rewards (unused in labels).
        save_dir: Output directory.
        episode: 1-based episode index used in the filename.
    """
    import matplotlib.pyplot as _plt

    colors = _plt.cm.tab10.colors  # type: ignore[attr-defined]
    artists = []
    for i, (path, _r) in enumerate(zip(paths, rewards)):
        col = colors[i % len(colors)]
        (line,) = ax.plot(
            path[:, 0], path[:, 1],
            "-", color=col, linewidth=1.2, alpha=0.8,
            label=f"Ep {i + 1}",
        )
        artists.append(line)
    ax.legend(loc="upper right", fontsize=7)
    out = save_dir / f"minimap_ep{episode:03d}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"[minimap] saved -> {out}")
    for a in artists:
        a.remove()
    ax.legend(loc="upper right", fontsize=7)


def _minimap_process_main(
    snap: dict,
    save_dir: str,
    window_title: str,
    update_interval: float,
    cmd_q,
    update_q,
    ack_q,
    ready_evt,
) -> None:
    """Run the live minimap in a dedicated child process."""
    import gc as _gc
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.patches as _patches
    import matplotlib.pyplot as _plt

    lo, hi = MinimapFossenAirsim.MIN_GRID_POS, MinimapFossenAirsim.MAX_GRID_POS
    hlen = 300.0
    save_dir_path = Path(save_dir)

    ue5_ctx = _find_ue5_window_context()
    fig_dpi = 100
    minimap_px = _resolve_minimap_pixels(ue5_ctx)
    fig_in = minimap_px / fig_dpi
    sp = snap["ship_pos_0"]
    view_half_span = _estimate_view_half_span(snap)
    view_center = _preferred_view_center(sp, snap["checkpoints"], view_half_span)

    fig, ax = _plt.subplots(figsize=(fig_in, fig_in), dpi=fig_dpi)
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass
    ax.set_title(window_title, fontsize=10)
    ax.set_xlabel("X / East (m)")
    ax.set_ylabel("Y / North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    ax.plot(
        [lo, hi, hi, lo, lo],
        [lo, lo, hi, hi, lo],
        "k-", linewidth=0.6, label="Grid boundary",
    )
    if snap["obstacles"] is not None:
        ax.add_patch(_patches.Polygon(
            snap["obstacles"], closed=True,
            edgecolor="#cc3333", facecolor="none",
            linewidth=1.5, label="Waterway",
        ))
    if snap["overall"] is not None:
        ax.add_patch(_patches.Polygon(
            snap["overall"], closed=True,
            edgecolor="#8B4513", facecolor="none",
            linewidth=1.0, label="Western Scheldt",
        ))
    checkpoints = snap["checkpoints"]
    if checkpoints:
        cp_pos = np.array([cp["pos"] for cp in checkpoints])
        ax.plot(cp_pos[:, 0], cp_pos[:, 1], "g--", linewidth=0.8, alpha=0.6, label="Ideal path")
    ax.scatter([sp[0]], [sp[1]], c="blue", s=80, zorder=5, label="Start")
    if snap["target_pos"] is not None:
        tp = snap["target_pos"]
        ax.scatter([tp[0]], [tp[1]], c="red", s=100, marker="*", zorder=5, label="Target")
    ax.legend(loc="upper right", fontsize=7)
    _apply_view_limits(ax, view_center, view_half_span)

    current_path: List[np.ndarray] = [sp.copy()]
    (trail_line,) = ax.plot([], [], color="royalblue", linewidth=0.9, alpha=0.7)
    (ship_dot,) = ax.plot([sp[0]], [sp[1]], "o", color="royalblue", markersize=8, zorder=6)
    hdg0 = snap["initial_heading"]
    (heading_line,) = ax.plot(
        [sp[0], sp[0] + np.cos(hdg0) * hlen],
        [sp[1], sp[1] + np.sin(hdg0) * hlen],
        "-", color="royalblue", linewidth=1.5, zorder=6,
    )

    fig.canvas.draw()
    bg = [fig.canvas.copy_from_bbox(ax.bbox)]
    pending_reframe = [False]

    def on_draw(_event):
        bg[0] = fig.canvas.copy_from_bbox(ax.bbox)
        pending_reframe[0] = False

    fig.canvas.mpl_connect("draw_event", on_draw)
    _plt.show(block=False)
    try:
        fig.canvas.flush_events()
    except Exception:
        pass

    minimap_position = _resolve_minimap_position(ue5_ctx, minimap_px)
    if minimap_position is not None:
        try:
            mm_x, mm_y = minimap_position
            fig.canvas.manager.window.wm_geometry(f"+{mm_x}+{mm_y}")
            fig.canvas.flush_events()
        except Exception:
            pass

    ready_evt.set()
    paths: List[np.ndarray] = []
    rewards: List[float] = []
    last_update_time = 0.0

    while True:
        try:
            cmd = cmd_q.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd is not None:
            kind = cmd[0]
            if kind == "close":
                break
            if kind == "finish":
                _, reward, episode = cmd
                if current_path:
                    paths.append(np.array(current_path))
                    rewards.append(float(reward))
                current_path = [sp.copy()]
                _worker_save_png(fig, ax, paths, rewards, save_dir_path, episode)
                fig.canvas.draw_idle()
                ack_q.put(("finish", int(episode)))

        latest_update = _drain_latest_update(update_q)
        if latest_update is not None and latest_update[0] == "update":
            _, pos, hdg = latest_update
            now = time.monotonic()
            if now - last_update_time >= update_interval:
                last_update_time = now
                current_path.append(pos.copy())
                trail = np.array(current_path)
                trail_line.set_data(trail[:, 0], trail[:, 1])
                ship_dot.set_data([pos[0]], [pos[1]])
                hx = pos[0] + np.cos(hdg) * hlen
                hy = pos[1] + np.sin(hdg) * hlen
                heading_line.set_data([pos[0], hx], [pos[1], hy])
                if _should_shift_view(pos, view_center, view_half_span):
                    view_center = _preferred_view_center(pos, checkpoints, view_half_span)
                    _apply_view_limits(ax, view_center, view_half_span)
                    pending_reframe[0] = True
                    fig.canvas.draw_idle()
                elif bg[0] is not None and not pending_reframe[0]:
                    fig.canvas.restore_region(bg[0])
                    ax.draw_artist(trail_line)
                    ax.draw_artist(ship_dot)
                    ax.draw_artist(heading_line)
                    fig.canvas.blit(ax.bbox)

        try:
            fig.canvas.flush_events()
        except Exception:
            pass

        time.sleep(0.003)

    try:
        _plt.close(fig)
    except Exception:
        pass
    _gc.collect()


class MinimapFossenAirsim:
    """Live-updating ship minimap isolated in a child process.

    This keeps expensive viewport changes and Tk event handling out of the
    main inference process, which prevents small redraw stalls from also
    stalling UE5 updates.
    """

    MIN_GRID_POS = -11700
    MAX_GRID_POS = 14500

    def __init__(
        self,
        env,
        save_dir: str = ".",
        window_title: str = "RudoLF - Live Minimap",
        max_fps: float = 15.0,
    ) -> None:
        self._save_dir = Path(save_dir)
        self._window_title = window_title
        self._update_interval = 1.0 / max(max_fps, 1.0)
        self._snap = _snapshot_env(env)
        self._ctx = multiprocessing.get_context("spawn")
        self._cmd_q = self._ctx.Queue()
        self._update_q = self._ctx.Queue(maxsize=1)
        self._ack_q = self._ctx.Queue()
        self._ready_evt = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_minimap_process_main,
            name="minimap-process",
            daemon=True,
            args=(
                self._snap,
                str(self._save_dir),
                self._window_title,
                self._update_interval,
                self._cmd_q,
                self._update_q,
                self._ack_q,
                self._ready_evt,
            ),
        )
        self._proc.start()
        if not self._ready_evt.wait(timeout=15.0):
            logger.warning("Minimap process did not become ready within 15 s.")

    def update(self, env) -> None:
        """Queue the current ship position for the next minimap redraw."""
        pos = np.array(env.ship_pos[0], dtype=np.float64)
        hdg = float(env.phys_sims[0].state.heading)
        _try_put_latest(self._update_q, ("update", pos, hdg))

    def finish(self, total_reward: float, episode: int = 1) -> None:
        """Finalise an episode and wait briefly for the PNG save acknowledgement."""
        self._cmd_q.put(("finish", float(total_reward), int(episode)))
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                ack_kind, ack_episode = self._ack_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if ack_kind == "finish" and ack_episode == int(episode):
                return

    def close(self) -> None:
        """Close the minimap process and clean up IPC resources."""
        if self._proc is None:
            return

        if self._proc.is_alive():
            self._cmd_q.put(("close",))
            self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)

        self._proc = None
