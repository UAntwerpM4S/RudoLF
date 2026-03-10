"""Live matplotlib minimap for Fossen-AirSim visualisation.

Mirrors the same map that ``AgentManager.render()`` draws but updates the
ship marker in real time, once per Fossen step, driven by the same clock as
the physics simulation.

The minimap runs on a dedicated daemon thread so that GUI events (resize,
drag, modal dialogs on Windows) never block the Fossen/AirSim main loop.
Holding the window title bar on Windows freezes only the minimap thread;
the UE5 RPC channel and Fossen physics continue uninterrupted.

Usage (embedded - called from run_visualise_fossen)::

    from airsim.inference.minimap_fossen_airsim import MinimapFossenAirsim

    minimap = MinimapFossenAirsim(env)  # once, after env.reset()
    minimap.update(env)                 # each step, after env.step()
    minimap.finish(total_reward)        # at episode end - saves PNG
    minimap.close()                     # on teardown
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _find_ue5_rect():
    """Return (x, y, w, h) of the UE5 window on Windows, or None.

    Uses the Win32 class name ``UnrealWindow`` which all UE4/UE5 instances
    register automatically.  Pure-ctypes, no extra dependencies.  Returns
    ``None`` on non-Windows platforms or when no UE5 window is found.
    """
    try:
        import platform
        if platform.system() != "Windows":
            return None
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW("UnrealWindow", None)
        if not hwnd:
            return None
        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        x, y = int(rect.left), int(rect.top)
        w, h = int(rect.right - rect.left), int(rect.bottom - rect.top)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
    except Exception:
        return None


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


class MinimapFossenAirsim:
    """Live-updating ship position minimap, isolated on a daemon thread.

    All matplotlib/Tk operations run exclusively on a dedicated daemon
    thread.  Holding or dragging the window on Windows blocks only that
    thread; the main Fossen/AirSim inference loop and the UE5 RPC channel
    continue uninterrupted.

    The map draws a static background (waterway polygon, Scheldt outline,
    checkpoint circles, ideal path line) once at construction time and
    then blits only the ship marker on each update cycle.

    Args:
        env: A fully reset Fossen environment (``PySimEnv`` or ``FhSimEnv``).
             Must have ``checkpoints``, ``obstacles``, ``overall``,
             ``target_pos``, ``ship_pos``, and ``phys_sims`` attributes.
        save_dir: Directory for saved PNG files.  Defaults to current dir.
        window_title: Title shown in the Matplotlib window title bar.
        max_fps: Maximum screen refresh rate in frames per second.  Calls to
            ``update()`` that arrive faster than this rate are silently
            dropped, keeping the main inference loop unblocked.  Defaults
            to ``15.0`` (one redraw per 66 ms).
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
        self._update_interval: float = 1.0 / max(max_fps, 1.0)

        # Latest position buffer.  deque(maxlen=1) gives latest-wins semantics.
        # In CPython individual append() and popleft() are each protected by the
        # GIL, making cross-thread access safe without an explicit lock.
        self._pos_buf: collections.deque = collections.deque(maxlen=1)

        # Control messages processed in submission order (finish / close).
        self._cmd_q: queue.Queue = queue.Queue()

        # Signalled by the worker once its figure is ready to display.
        self._ready_evt = threading.Event()

        # Snapshot all static env data on the calling thread before spawning.
        self._snap = _snapshot_env(env)

        self._thread = threading.Thread(
            target=self._worker,
            name="minimap-worker",
            daemon=True,
        )
        self._thread.start()

        # Block the caller until the worker's figure is initialised.
        if not self._ready_evt.wait(timeout=15.0):
            logger.warning("Minimap worker thread did not start within 15 s.")

    # -- Public API (main thread) --

    def update(self, env) -> None:
        """Queue the current ship position for the next minimap redraw.

        Non-blocking and latest-wins: if the worker has not yet consumed
        the previous update it is silently overwritten.  The calling thread
        is never stalled, even when the matplotlib window is being dragged
        on Windows.

        Args:
            env: The active Fossen environment after a ``step()`` call.
        """
        pos = np.array(env.ship_pos[0], dtype=np.float64)
        hdg = float(env.phys_sims[0].state.heading)
        self._pos_buf.append((pos, hdg))

    def finish(self, total_reward: float, episode: int = 1) -> None:
        """Finalise episode: archive the trail and save the cumulative PNG.

        Blocks until the PNG has been written (or a 10-second timeout).

        Args:
            total_reward: Scalar episode reward (kept for internal bookkeeping).
            episode: 1-based episode index used in the output filename.
        """
        done_evt = threading.Event()
        self._cmd_q.put(("finish", float(total_reward), int(episode), done_evt))
        done_evt.wait(timeout=10.0)

    def close(self) -> None:
        """Close the matplotlib window and stop the worker thread."""
        self._cmd_q.put(("close",))
        self._thread.join(timeout=5.0)

    # -- Worker thread --

    def _worker(self) -> None:
        """Matplotlib event loop running entirely on the minimap-worker thread.

        All figure creation, drawing, and Tk event pumping is confined here.
        Any OS-level blocking caused by user window interaction (resize, drag,
        title-bar hold) cannot propagate to the calling thread.
        """
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.patches as _patches
        import matplotlib.pyplot as _plt

        snap = self._snap
        lo, hi = self.MIN_GRID_POS, self.MAX_GRID_POS
        hlen = 300.0

        # Detect the UE5 window before creating the figure so we can size and
        # position the minimap correctly from the start.
        _ue5_rect = _find_ue5_rect()
        _fig_dpi = 100
        if _ue5_rect is not None:
            _, _, _, _ue5_h = _ue5_rect
            # 28 % of UE5 height, kept within a sensible pixel range.
            _mm_px = max(350, min(int(_ue5_h * 0.28), 650))
        else:
            _mm_px = 700  # default when UE5 is not detected
        _fig_in = _mm_px / _fig_dpi

        # -- Build figure --
        fig, ax = _plt.subplots(figsize=(_fig_in, _fig_in), dpi=_fig_dpi)
        try:
            fig.canvas.manager.set_window_title(self._window_title)
        except Exception:
            pass
        ax.set_title(self._window_title, fontsize=10)
        ax.set_xlabel("X / East (m)")
        ax.set_ylabel("Y / North (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

        # -- Static layer --
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
        cps = snap["checkpoints"]
        if cps:
            cp_pos = np.array([cp["pos"] for cp in cps])
            ax.plot(cp_pos[:, 0], cp_pos[:, 1], "g--", linewidth=0.8, alpha=0.6, label="Ideal path")
        sp = snap["ship_pos_0"]
        ax.scatter([sp[0]], [sp[1]], c="blue", s=80, zorder=5, label="Start")
        if snap["target_pos"] is not None:
            tp = snap["target_pos"]
            ax.scatter([tp[0]], [tp[1]], c="red", s=100, marker="*", zorder=5, label="Target")
        ax.legend(loc="upper right", fontsize=7)

        # -- Dynamic artists --
        current_path: List[np.ndarray] = [sp.copy()]
        (trail_line,) = ax.plot([], [], color="royalblue", linewidth=0.9, alpha=0.7)
        (ship_dot,) = ax.plot([sp[0]], [sp[1]], "o", color="royalblue", markersize=8, zorder=6)
        hdg0 = snap["initial_heading"]
        (heading_line,) = ax.plot(
            [sp[0], sp[0] + np.cos(hdg0) * hlen],
            [sp[1], sp[1] + np.sin(hdg0) * hlen],
            "-", color="royalblue", linewidth=1.5, zorder=6,
        )

        # Bake static background for blit.
        fig.canvas.draw()
        # Use a list so the closure inside on_draw can rebind the cached value.
        bg = [fig.canvas.copy_from_bbox(ax.bbox)]

        def on_draw(_event):
            bg[0] = fig.canvas.copy_from_bbox(ax.bbox)

        fig.canvas.mpl_connect("draw_event", on_draw)
        _plt.show(block=False)
        try:
            fig.canvas.flush_events()
        except Exception:
            pass

        # Auto-position the minimap in the top-right corner of the UE5 window.
        # Works for borderless-fullscreen (the most common AirSim setup);
        # silently skipped if UE5 was not found or the Tk call fails.
        if _ue5_rect is not None:
            try:
                _ue5_x, _ue5_y, _ue5_w, _ = _ue5_rect
                _inset = 8
                _mm_x = _ue5_x + _ue5_w - _mm_px - _inset
                _mm_y = _ue5_y + _inset
                fig.canvas.manager.window.wm_geometry(f"+{_mm_x}+{_mm_y}")
                fig.canvas.flush_events()
            except Exception:
                pass

        self._ready_evt.set()

        # -- Per-episode persistent state --
        paths: List[np.ndarray] = []
        rewards: List[float] = []
        last_update_time = 0.0

        # -- Worker event loop --
        while True:
            # Drain one control command without blocking.
            try:
                cmd = self._cmd_q.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                kind = cmd[0]
                if kind == "close":
                    break
                elif kind == "finish":
                    _, reward, episode, done_evt = cmd
                    if current_path:
                        paths.append(np.array(current_path))
                        rewards.append(float(reward))
                    current_path = [sp.copy()]
                    _worker_save_png(fig, ax, paths, rewards, self._save_dir, episode)
                    fig.canvas.draw()
                    bg[0] = fig.canvas.copy_from_bbox(ax.bbox)
                    done_evt.set()

            # Consume the latest position update (latest-wins).
            if self._pos_buf:
                pos, hdg = self._pos_buf.popleft()
                now = time.monotonic()
                if now - last_update_time >= self._update_interval:
                    last_update_time = now
                    current_path.append(pos.copy())
                    if bg[0] is not None:
                        fig.canvas.restore_region(bg[0])
                    trail = np.array(current_path)
                    trail_line.set_data(trail[:, 0], trail[:, 1])
                    ship_dot.set_data([pos[0]], [pos[1]])
                    hx = pos[0] + np.cos(hdg) * hlen
                    hy = pos[1] + np.sin(hdg) * hlen
                    heading_line.set_data([pos[0], hx], [pos[1], hy])
                    ax.draw_artist(trail_line)
                    ax.draw_artist(ship_dot)
                    ax.draw_artist(heading_line)
                    fig.canvas.blit(ax.bbox)

            # Flush Tk events.  On Windows, a window-drag blocks here.
            # Blocking is intentional: only this thread pauses; the main
            # Fossen/AirSim loop continues unaffected.
            try:
                fig.canvas.flush_events()
            except Exception:
                pass

            # Brief yield to avoid monopolising the GIL while idle.
            time.sleep(0.003)

        try:
            _plt.close(fig)
        except Exception:
            pass
        # Force-collect remaining Tk objects while still on the worker thread.
        # Without this, Python's GC finalises them on the main thread at exit,
        # triggering "RuntimeError: main thread is not in main loop" noise.
        import gc as _gc
        _gc.collect()
