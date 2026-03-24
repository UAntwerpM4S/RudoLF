"""UE5 inference runner with legacy Fossen-visualisation compatibility.

Two control modes are available, selected automatically from the model filename
or overridden via CLI flags:

**Fossen-translation mode**  (default; auto-detected for PySimEnv / FhSimEnv models)
    The policy was trained against Fossen physics (PySimEnv or FhSimEnv).
    Observations are read from UE5 and converted to the same 10-dim
    Fossen-compatible space the policy was trained on.  Actions from the
    policy are forwarded through :class:`~airsim.environments.airsim_env.AirSimEnv`
    which applies the Fossen-equivalent affine throttle formula:

    .. code-block:: text

        throttle = min_thrust + 0.5 * (action[1] + 1) * (max_thrust - min_thrust)
        #  action = -1 → 0.10   (slow ahead)
        #  action =  0 → 0.55   (half ahead)
        #  action = +1 → 1.00   (full ahead)

    Before sending, controls pass through ``AirSimBridge.set_controls()`` –
    a pure RPC pass-through with **no PID control**.

**UE5-direct mode**  (``--ue5`` flag  OR  ``AirSimEnv`` in the model filename)
    The policy was trained directly in UE5 (AirSimEnv).
    Actions are sent straight to the ``CarControls`` interface via
    :class:`~airsim.communication.airsim_bridge.AirSimBridge` using a simple
    linear mapping that bypasses the Fossen affine formula:

    .. code-block:: text

        steering = action[0]               # direct [-1, 1]
        throttle = (action[1] + 1.0) / 2  # simple remap to [0, 1]

    **NOTE**: This path is structurally ready and tested.
    Full UE5-native inference (including reward-based checkpoint tracking)
    will be finalized once UE5 training is complete.

Communication layer
-------------------
:class:`~airsim.communication.airsim_bridge.AirSimBridge` is the only transport
layer.  **PID control is explicitly NOT used anywhere in this script or in
the bridge.**  The bridge provides a pure pass-through (``set_controls``,
``set_pose``, ``get_state``).

World load
----------
On startup :class:`~airsim.environments.airsim_env.AirSimEnv` calls
``simSetSaveSlot("simulator_1")`` + ``simRequestLoad()``, restoring the MYZAKO
ship to its correct starting position and heading before the first episode.
Simulation speed is set to ``1.0`` (real-time) for visualization.

Usage::

    # Fossen-trained model (auto-detected)
    python -m airsim.inference.run_inference \\
        --model src/fw/best_models/PySimEnv_PPO_policy

    # UE5-trained model (explicit flag or auto-detected from filename)
    python -m airsim.inference.run_inference \\
        --model src/fw/best_models/AirSimEnv_PPO_policy --ue5

    # Dedicated framework-to-UE5 visualisation runner
    python -m airsim.inference.run_visualise_models \
        --model src/fw/best_models/PySimEnv_PPO_policy

    # Override start/target position
    python -m airsim.inference.run_inference \\
        --model my_model --ship-pos 2245 -47 --target-pos 11530 13000

    python -m airsim.inference.run_inference --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Bootstrap the Python path when run directly (python run_inference.py) ──
_SRC_DIR = Path(__file__).resolve().parent.parent.parent


if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.environments.airsim_env import AirSimEnv          # noqa: E402
from airsim.communication.airsim_bridge import AirSimBridge   # noqa: E402
from airsim.communication.control_mapping import map_direct_policy_action  # noqa: E402
from fw.common.airsim.tools.airsim_export import (  # noqa: E402
    AirSimSessionExporter,
    build_airsim_env_snapshot,
)
from fw.common.airsim.tools.path_utils import resolve_model_path  # noqa: E402
from fw.policies.ppo_policy import PPOPolicy                   # noqa: E402

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Model origin detection
# ─────────────────────────────────────────────────────────

_UE5_KEYWORDS: Tuple[str, ...] = ("airsimenv", "airsim_env", "ue5")
_FOSSEN_KEYWORDS: Tuple[str, ...] = (
    "pysimenv", "fhsimenv", "fossen", "py_sim", "fh_sim",
)


def detect_model_origin(model_path: str) -> str:
    """Determine whether a model was trained in UE5 or a Fossen simulator.

    Checks the model filename stem against known keyword lists.
    If no keyword matches the function defaults to ``"fossen"`` – the safer
    fallback because applying the Fossen mapping to an unknown model is
    less likely to cause runaway behaviour than skipping it.

    Args:
        model_path: Filesystem path to the ``.zip`` model file.

    Returns:
        ``"ue5"`` when the model was trained in :class:`AirSimEnv`, or
        ``"fossen"`` for all Fossen-physics models (PySimEnv / FhSimEnv).
    """
    stem = Path(model_path).stem.lower()
    for kw in _UE5_KEYWORDS:
        if kw in stem:
            return "ue5"
    for kw in _FOSSEN_KEYWORDS:
        if kw in stem:
            return "fossen"
    logger.warning(
        "Cannot determine training origin from filename '%s'. "
        "Defaulting to Fossen-translation mode (safe fallback).",
        Path(model_path).name,
    )
    return "fossen"


# ─────────────────────────────────────────────────────────
# Policy loader
# ─────────────────────────────────────────────────────────

def load_policy(model_path: str, device: str = "cpu") -> PPOPolicy:
    """Load a :class:`PPOPolicy` from a saved ``.zip`` file.

    ``PPOPolicy.load`` appends ``.zip`` internally, so that extension is
    stripped here before passing the path.

    Args:
        model_path: Path to the model file, with or without ``.zip``.
        device: PyTorch device – ``"cpu"`` or ``"cuda"``.

    Returns:
        A :class:`PPOPolicy` ready for inference (``network.eval()``).

    Raises:
        RuntimeError: If the model file cannot be found or parsed.
    """
    path = resolve_model_path(model_path, _SRC_DIR)
    stem = str(path.with_suffix("")) if path.suffix == ".zip" else str(path)
    policy = PPOPolicy.load(stem, device=device)
    policy.network.eval()
    logger.info("Policy loaded from '%s' (device=%s).", path.name, device)
    return policy


# ─────────────────────────────────────────────────────────
# UE5-direct control step  (placeholder ready for future use)
# ─────────────────────────────────────────────────────────

def _ue5_direct_step(env: AirSimEnv, action: np.ndarray) -> np.ndarray:
    """Send an action directly to UE5 via :class:`AirSimBridge` – no Fossen formula.

    This is the UE5-native control path for models trained directly in
    :class:`AirSimEnv`.  Controls are mapped with a simple linear formula:

    .. code-block:: text

        steering = action[0]               # direct [-1, 1]
        throttle = (action[1] + 1.0) / 2  # remap to [0, 1]

    After sending the controls the function reads back vehicle state from
    UE5 via ``bridge.get_state()`` → ``env._read_airsim_state()``, updates
    the internal ``env`` state (position, heading, speed), and returns the
    new 10-dim observation.

    **No PID control is involved** – actions flow directly to the
    ``AirSimBridge.set_controls()`` RPC call.

    Args:
        env: An initialised :class:`AirSimEnv` whose bridge is connected.
             ``env.reset()`` must have been called before the first step.
        action: Normalised action ``[steering, throttle]`` each in [-1, 1].

    Returns:
        New 10-dim observation from :meth:`AirSimEnv._get_obs`.

    Note:
        Checkpoint-index tracking (used for reward shaping) is not updated
        in this path because rewards are not computed in UE5-direct mode.
        When UE5 training is complete and reward-based evaluation is needed,
        move checkpoint tracking here or refactor into a public
        ``AirSimEnv.update_checkpoints()`` helper.
    """
    bridge: AirSimBridge = env.bridge

    # ── Direct CarControls – NO Fossen affine formula, NO PID ──
    controls = map_direct_policy_action(env._airsim, action)
    env.last_controls = {
        "steering": float(controls.steering),
        "throttle": float(controls.throttle),
        "brake": float(controls.brake),
    }

    # Save state needed by _read_airsim_state for yaw-rate derivation
    env.previous_ship_pos = env.ship_pos.copy()
    env.previous_heading = env.ship_heading

    # ── Send to UE5 (pure pass-through, no PID) ──
    bridge.set_controls(controls)
    time.sleep(env.time_step)

    # ── Read back UE5 state → update env fields ──
    env._read_airsim_state()
    env.step_count += 1

    return env._get_obs()


# ─────────────────────────────────────────────────────────
# Episode runners
# ─────────────────────────────────────────────────────────

def _run_episode_fossen(
    env: AirSimEnv,
    policy: PPOPolicy,
    max_steps: int,
    episode_index: int = 1,
    exporter: Optional[AirSimSessionExporter] = None,
    model_origin: str = "fossen",
) -> Tuple[List[float], List[float], float]:
    """Run one inference episode in Fossen-translation mode.

    Uses :meth:`AirSimEnv.step` which internally applies the Fossen→UE5
    affine throttle formula and sends commands via :class:`AirSimBridge`.
    No PID control is used anywhere in this path.

    Args:
        env: An initialised :class:`AirSimEnv`.
        policy: Loaded :class:`PPOPolicy` in eval mode.
        max_steps: Step cap for this episode.

    Returns:
        A tuple ``(trajectory_x, trajectory_y, total_reward)`` where
        positions are in Fossen east/north coordinates.
    """
    obs, _ = env.reset()
    traj_x: List[float] = [float(env.ship_pos[0])]
    traj_y: List[float] = [float(env.ship_pos[1])]
    total_reward = 0.0
    step = 0

    while step < max_steps:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(policy.device)
        with torch.no_grad():
            action_t, _, _ = policy._predict(state_t, deterministic=True)
        action = action_t.detach().cpu().numpy().flatten()

        # AirSimEnv.step() applies:
        #   1. Actuator smoothing (rate-limited rudder)
        #   2. Fossen affine throttle formula  → CarControls
        #   3. bridge.set_controls()           → UE5 RPC (no PID)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1

        traj_x.append(float(env.ship_pos[0]))
        traj_y.append(float(env.ship_pos[1]))
        if exporter is not None:
            exporter.log_inference_step(
                _build_inference_row(
                    env=env,
                    episode_index=episode_index,
                    step_index=step,
                    reward=float(reward),
                    action=action,
                    model_origin=model_origin,
                    control_mode="fossen_translation",
                    outcome="terminated" if done else ("truncated" if truncated else "running"),
                )
            )

        if step % 200 == 0:
            logger.info(
                "  [Fossen] step=%d  pos=(%.0f, %.0f)  "
                "reward=%+.2f  cumulative=%+.1f  checkpoint=%d",
                step,
                env.ship_pos[0],
                env.ship_pos[1],
                reward,
                total_reward,
                env.checkpoint_index,
            )

        if done or truncated:
            break

    return traj_x, traj_y, total_reward


def _run_episode_ue5_direct(
    env: AirSimEnv,
    policy: PPOPolicy,
    max_steps: int,
    episode_index: int = 1,
    exporter: Optional[AirSimSessionExporter] = None,
    model_origin: str = "ue5",
) -> Tuple[List[float], List[float], float]:
    """Run one inference episode in UE5-direct control mode.

    Actions are sent directly to :class:`AirSimBridge` via
    :func:`_ue5_direct_step` without the Fossen affine throttle formula.
    Observations are still built by :class:`AirSimEnv` (same 10-dim format
    used when training in UE5) so the policy experiences the same interface
    it was trained against.

    **Placeholder note**: Checkpoint-index tracking and reward computation
    are not active in this path.  These will be wired up once UE5 training
    is complete and the full obs → action → reward loop has been validated.

    Args:
        env: An initialised :class:`AirSimEnv`.
        policy: Loaded :class:`PPOPolicy` in eval mode.
        max_steps: Step cap for this episode.

    Returns:
        A tuple ``(trajectory_x, trajectory_y, 0.0)``; reward is always
        ``0.0`` because the reward function is not called in this path.
    """
    obs, _ = env.reset()
    traj_x: List[float] = [float(env.ship_pos[0])]
    traj_y: List[float] = [float(env.ship_pos[1])]
    step = 0

    while step < max_steps:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(policy.device)
        with torch.no_grad():
            action_t, _, _ = policy._predict(state_t, deterministic=True)
        action = action_t.detach().cpu().numpy().flatten()

        # ── UE5-direct: bridge.set_controls() – no Fossen formula, no PID ──
        # TODO (when UE5 training completes):
        #   • Add checkpoint-index tracking via a public update helper
        #   • Wire in reward computation for performance monitoring
        #   • Validate obs normalisation matches the UE5 training config
        obs = _ue5_direct_step(env, action)
        step += 1

        traj_x.append(float(env.ship_pos[0]))
        traj_y.append(float(env.ship_pos[1]))
        if exporter is not None:
            exporter.log_inference_step(
                _build_inference_row(
                    env=env,
                    episode_index=episode_index,
                    step_index=step,
                    reward=0.0,
                    action=action,
                    model_origin=model_origin,
                    control_mode="ue5_direct",
                    outcome="running",
                )
            )

        if step % 200 == 0:
            logger.info(
                "  [UE5-direct] step=%d  pos=(%.0f, %.0f)  speed=%.2f m/s",
                step,
                env.ship_pos[0],
                env.ship_pos[1],
                env.ship_speed,
            )

    return traj_x, traj_y, 0.0


def _build_inference_row(
    env: AirSimEnv,
    episode_index: int,
    step_index: int,
    reward: float,
    action: np.ndarray,
    model_origin: str,
    control_mode: str,
    outcome: str,
) -> Dict[str, float | int | str]:
    """Build one inference telemetry row for CSV export."""
    snapshot = build_airsim_env_snapshot(env)
    return {
        "step": int(step_index),
        "sim_time": float(step_index * env.time_step),
        "episode": int(episode_index),
        "episode_step": int(step_index),
        "x": float(snapshot["ship_x"]),
        "y": float(snapshot["ship_y"]),
        "pos_x": float(snapshot["ship_x"]),
        "pos_y": float(snapshot["ship_y"]),
        "heading_deg": float(snapshot["heading_deg"]),
        "speed": float(snapshot["speed"]),
        "reward": float(reward),
        "steering": float(action[0]),
        "throttle": float(action[1]),
        "brake": float(env.last_controls.get("brake", 0.0)),
        "checkpoint_index": int(snapshot["checkpoint_index"]),
        "cross_track_error": float(snapshot["cross_track_error"]),
        "physics_wait_s": float(snapshot["physics_wait_s"]),
        "step_latency_s": float(snapshot["step_latency_s"]),
        "ue5_timestamp": int(snapshot["ue5_timestamp"]),
        "outcome": str(outcome),
        "model_origin": str(model_origin),
        "control_mode": str(control_mode),
    }


# ─────────────────────────────────────────────────────────
# Main inference entry-point
# ─────────────────────────────────────────────────────────

def run_inference(
    model_path: str,
    ship_pos: Optional[List[float]] = None,
    target_pos: Optional[List[float]] = None,
    ip: str = "",
    port: int = 41451,
    max_steps: int = 14000,
    num_episodes: int = 3,
    device: str = "cpu",
    force_mode: Optional[str] = None,
) -> None:
    """Run visualization inference: control the UE5 MYZAKO ship with a PPO policy.

    Connects to the running UE5 / CosysAirSim instance via
    :class:`AirSimBridge`, loads the world save ``"simulator_1"`` automatically
    inside :class:`AirSimEnv`, and runs ``num_episodes`` of live ship control.

    The control mode is detected from the model filename:

    * Filename contains ``pysimenv`` / ``fhsimenv`` / ``fossen``
      → Fossen-translation mode (Fossen→UE5 action formula applied)

    * Filename contains ``airsimenv`` / ``ue5``
      → UE5-direct mode (actions sent straight to bridge, no Fossen formula)

    This can be overridden with ``force_mode="fossen"`` or ``force_mode="ue5"``.

    Args:
        model_path: Path to the ``.zip`` PPO model file.
        ship_pos: Fossen start position ``[x_east, y_north]``.
                  Defaults to the MYZAKO default ``[2245, -47]``.
        target_pos: Fossen target position ``[x_east, y_north]``.
        ip: AirSim server IP (empty string = ``127.0.0.1``).
        port: AirSim RPC port.
        max_steps: Maximum steps per episode.
        num_episodes: Number of inference episodes to run.
        device: PyTorch device (``"cpu"`` or ``"cuda"``).
        force_mode: ``"fossen"`` or ``"ue5"`` to override auto-detection.
    """
    _BANNER = "=" * 62
    resolved_model_path = resolve_model_path(model_path, _SRC_DIR)

    # ── Detect / resolve control mode ──────────────────────────────────────
    mode = force_mode if force_mode in ("fossen", "ue5") else detect_model_origin(model_path)

    if mode == "ue5":
        print(_BANNER)
        print("  UE5-NATIVE DIRECT CONTROL MODE")
        print(f"  Model  : {Path(model_path).name}")
        print("  Actions: bridge.set_controls() – linear mapping  [ NO Fossen formula ]")
        print("  PID    : NOT USED")
        print(_BANNER)
    else:
        print(_BANNER)
        print("  FOSSEN-TRANSLATION MODE  [compatibility path]")
        print(f"  Model  : {Path(model_path).name}")
        print("  Actions: Fossen→UE5 affine formula → bridge.set_controls()")
        print("  Prefer : python -m airsim.inference.run_visualise_models")
        print("  PID    : NOT USED")
        print(_BANNER)

    # ── Load PPO policy ─────────────────────────────────────────────────────
    policy = load_policy(model_path, device=device)
    session_exporter = AirSimSessionExporter(
        session_name=f"inference_{resolved_model_path.stem}",
        metadata={
            "script": "airsim.inference.run_inference",
            "model_path": str(resolved_model_path),
            "model_name": resolved_model_path.name,
            "environment_name": "AirSimEnv",
            "model_origin": mode,
            "configuration": {
                "ship_pos": ship_pos,
                "target_pos": target_pos,
                "ip": ip or "127.0.0.1",
                "port": port,
                "max_steps": max_steps,
                "num_episodes": num_episodes,
                "device": device,
                "force_mode": force_mode,
                "simulation_speed": 1.0,
            },
        },
    )

    # ── Initialise AirSimEnv ────────────────────────────────────────────────
    # AirSimEnv.__init__ performs:
    #   1. AirSimBridge connection (no PID in the bridge)
    #   2. simSetSaveSlot("simulator_1") + simRequestLoad()
    #      → restores MYZAKO ship to its exact Fossen starting position
    #   3. simSetSimulationSpeed(1.0)  → real-time for visualization
    logger.info("Connecting to UE5 and loading world save 'simulator_1'…")
    env = AirSimEnv(
        ship_pos=ship_pos,
        target_pos=target_pos,
        ip=ip,
        port=port,
        max_steps=max_steps,
        training_mode=False,
        use_pause_stepping=False,
        simulation_speed=1.0,
        csv_log_path=str(session_exporter.bridge_log_path),
    )
    session_exporter.update_metadata(
        {
            "resolved_model_path": str(resolved_model_path),
            "export_directory": str(session_exporter.export_dir),
            "bridge_telemetry_path": str(session_exporter.bridge_log_path),
        }
    )

    bridge: AirSimBridge = env.bridge
    logger.info(
        "AirSimBridge connected=%s | mode=%s | episodes=%d",
        bridge.is_connected,
        mode.upper(),
        num_episodes,
    )

    # ── Optional matplotlib trajectory plot ────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except ImportError:
        _HAS_MPL = False
        logger.warning("matplotlib not installed – trajectory plotting disabled.")

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"RudoLF Inference  [{mode.upper()}]  –  {Path(model_path).name}", fontsize=11)
        ax.set_xlabel("Fossen X / East (m)")
        ax.set_ylabel("Fossen Y / North (m)")
        ax.set_aspect("equal", adjustable="datalim")
        _colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    last_traj_x: List[float] = []
    last_traj_y: List[float] = []

    # ── Episode loop ────────────────────────────────────────────────────────
    try:
        for ep in range(num_episodes):
            print(f"\n── Episode {ep + 1}/{num_episodes}  [{mode.upper()}] ──")

            if mode == "fossen":
                # Fossen-trained policy: AirSimEnv.step() applies the
                # Fossen→UE5 affine throttle formula before calling bridge.
                traj_x, traj_y, total_reward = _run_episode_fossen(
                    env,
                    policy,
                    max_steps,
                    episode_index=ep + 1,
                    exporter=session_exporter,
                    model_origin=mode,
                )
                outcome = f"reward={total_reward:+.1f}"
            else:
                # UE5-trained policy: actions sent directly to bridge via
                # _ue5_direct_step() – no Fossen mapping applied.
                traj_x, traj_y, total_reward = _run_episode_ue5_direct(
                    env,
                    policy,
                    max_steps,
                    episode_index=ep + 1,
                    exporter=session_exporter,
                    model_origin=mode,
                )
                outcome = "direct-control (no reward)"

            steps_run = len(traj_x) - 1
            print(f"   Finished: {steps_run} steps | {outcome}")
            last_traj_x, last_traj_y = traj_x, traj_y

            if _HAS_MPL and traj_x:
                col = _colors[ep % len(_colors)]
                label = f"Ep {ep + 1}  r={total_reward:.0f}" if mode == "fossen" else f"Ep {ep + 1}"
                ax.plot(traj_x, traj_y, color=col, linewidth=1.3, label=label)
                ax.plot(traj_x[0], traj_y[0], "o", color=col, markersize=7)
                ax.plot(traj_x[-1], traj_y[-1], "x", color=col,
                        markersize=9, markeredgewidth=2)
            session_exporter.update_metadata(
                {
                    "last_episode_index": ep + 1,
                    "last_outcome": outcome,
                    "last_episode_steps": steps_run,
                    "last_ship_position": {
                        "x": float(traj_x[-1]) if traj_x else None,
                        "y": float(traj_y[-1]) if traj_y else None,
                    },
                }
            )
    finally:
        # Always close cleanly: stops the vehicle, restores real-time speed
        env.close()
        session_exporter.save()
        session_exporter.close()
        print("\nInference complete – AirSim connection closed.")

    # ── Save & show trajectory plot ─────────────────────────────────────────
    if _HAS_MPL and last_traj_x:
        ax.legend(loc="best")
        plt.tight_layout()
        out_path = session_exporter.export_dir / "inference_trajectory.png"
        plt.savefig(out_path, dpi=150)
        print(f"Trajectory plot saved to '{out_path}'.")
        plt.show()


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for AirSim inference runs.

    Returns:
        A configured ``ArgumentParser`` for the backwards-compatible AirSim
        inference entry point.
    """
    p = argparse.ArgumentParser(
        description=(
            "Inference runner: load a trained PPO policy and control the "
            "UE5 MYZAKO ship in real-time. Use run_visualise_models for "
            "Fossen-trained policies; this entry point is kept for UE5-direct "
            "inference and backwards compatibility. PID control is not used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "Path to the .zip model file.  "
            "Example: models/PySimEnv_PPO_policy.zip"
        ),
    )

    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--ue5",
        action="store_true",
        default=False,
        help=(
            "Force UE5-direct control mode: actions sent directly to "
            "AirSimBridge.set_controls() – Fossen action formula skipped."
        ),
    )
    mode_grp.add_argument(
        "--fossen",
        action="store_true",
        default=False,
        help=(
            "Force Fossen-translation mode: AirSimEnv applies the Fossen "
            "affine throttle formula before forwarding to AirSimBridge."
        ),
    )

    p.add_argument(
        "--ip",
        default="",
        help="AirSim server IP address (empty string = 127.0.0.1).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=41451,
        help="AirSim RPC port.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of inference episodes to run.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=14000,
        dest="max_steps",
        help="Maximum steps allowed per episode.",
    )
    p.add_argument(
        "--ship-pos",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        dest="ship_pos",
        help="Start position in Fossen coords (east  north).",
    )
    p.add_argument(
        "--target-pos",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        dest="target_pos",
        help="Target position in Fossen coords (east  north).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device for policy inference.",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _build_parser().parse_args()
    force_mode = "ue5" if args.ue5 else ("fossen" if args.fossen else None)

    args.model = str(resolve_model_path(args.model, _SRC_DIR))

    run_inference(
        model_path=args.model,
        ship_pos=args.ship_pos,
        target_pos=args.target_pos,
        ip=args.ip,
        port=args.port,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        device=args.device,
        force_mode=force_mode,
    )
