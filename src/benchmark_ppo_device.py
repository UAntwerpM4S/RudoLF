"""
Time the torch-heavy parts of RudoLF training on CPU and CUDA:

1. Rollout inference: one PPOPolicy forward pass for num_envs observations, as
   run_parallel_episodes does every env step ('rollout_device' decides where).
2. One PPO update (SinglePPOAgent.train) on rollout_steps transitions ('device').
3. Optional (--surrogate-preset): one UAntwerp surrogate prediction for a single
   row, as UAntwerpSurrogate.step() does every SURROGATE_DT (now always CPU).

Environment stepping is NOT included: compare these numbers with the time per
iteration in your iteration log to see what share the policy takes at all.

--threads times the CPU parts with several intra-op thread counts (torch.set_num_threads).
Note that main_pipeline.py / main.py set OMP_NUM_THREADS=1, so training runs use 1
thread unless that is changed; this script itself starts with torch's default.

Run from src/:
    python benchmark_ppo_device.py --obs-dim 20 --act-dim 2 --num-envs 12 \
        --rollout-steps 8192 --batch-size 64 2048 --epochs 10 --surrogate-preset tripoli_full
    python benchmark_ppo_device.py --obs-dim 17 --num-envs 1 8 --rollout-steps 1700 13600 \
        --batch-size 64 --threads 1 2 4 8 16 --cpu-only
"""

import argparse
import time

import numpy as np
import torch

from fw.policies.ppo_policy import PPOPolicy
from fw.policies.single_ppo_agent import SinglePPOAgent


def _sync(device: torch.device) -> None:
    """
    Wait for queued CUDA work, so timings include it.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_rollout_inference(device: str, obs_dim: int, act_dim: int, num_envs: int, steps: int) -> float:
    """
    Mean time [ms] of one PPOPolicy.predict_with_raw call on num_envs NumPy rows.
    """

    policy = PPOPolicy(obs_dim, act_dim, device)
    states = np.random.randn(num_envs, obs_dim).astype(np.float32)
    for _ in range(50):  # warm-up (CUDA context, kernels)
        policy.predict_with_raw(states)
    _sync(policy.device)
    start = time.perf_counter()
    for _ in range(steps):
        policy.predict_with_raw(states)
    _sync(policy.device)
    return (time.perf_counter() - start) / steps * 1e3


def time_ppo_update(device: str, obs_dim: int, act_dim: int, rollout_steps: int, batch_size: int,
                    epochs: int, repeats: int = 3) -> float:
    """
    Median time [s] of `repeats` SinglePPOAgent.train() calls on rollout_steps
    random transitions (after one warm-up call).
    """

    agent = SinglePPOAgent.__new__(SinglePPOAgent)  # no env needed for train()
    agent.policy = PPOPolicy(obs_dim, act_dim, device)
    agent.device = agent.policy.device
    agent.optimizer = torch.optim.Adam(agent.policy.network.parameters(), lr=3e-4)
    agent.num_epochs, agent.batch_size = epochs, batch_size
    agent.clip_range, agent.value_loss_coef, agent.entropy_coef, agent.max_grad_norm = 0.2, 0.5, 0.01, 0.5

    rng = np.random.default_rng(0)
    episode = {
        "states": rng.standard_normal((rollout_steps, obs_dim)).astype(np.float32),
        "raw_actions": rng.standard_normal((rollout_steps, act_dim)).astype(np.float32),
        "advantages": rng.standard_normal(rollout_steps).astype(np.float32),
        "log_probs": rng.standard_normal(rollout_steps).astype(np.float32),
        "returns": rng.standard_normal(rollout_steps).astype(np.float32),
    }
    agent.train([episode])  # warm-up
    times = []
    for _ in range(max(1, repeats)):
        _sync(agent.device)
        start = time.perf_counter()
        agent.train([episode])
        _sync(agent.device)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def time_surrogate(device: str, preset_name: str, steps: int) -> float:
    """
    Mean time [ms] of one single-row SurrogatePredictor.predict_single call.
    """

    from fw.surrogate.inference_surrogate import SurrogatePredictor
    from fw.surrogate.presets import get_preset, resolve_checkpoint

    predictor = SurrogatePredictor(str(resolve_checkpoint(get_preset(preset_name).checkpoint)),
                                   device=torch.device(device))
    row = np.random.default_rng(0).standard_normal(predictor.raw_input_dim).astype(np.float32)
    for _ in range(50):
        predictor.predict_single(row)
    start = time.perf_counter()
    for _ in range(steps):
        predictor.predict_single(row)  # ends in .cpu(): includes the GPU wait
    return (time.perf_counter() - start) / steps * 1e3


def main() -> None:
    """
    Parse arguments, run both timings per device and print a table.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--act-dim", type=int, default=2)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[12])
    parser.add_argument("--inference-steps", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, nargs="+", default=[8192], help="transitions per PPO update")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[64, 2048])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3, help="PPO update timings per case (median reported)")
    parser.add_argument("--threads", type=int, nargs="+", default=None,
                        help="CPU intra-op thread counts to time (default: torch's default only)")
    parser.add_argument("--cpu-only", action="store_true", help="skip CUDA")
    parser.add_argument("--surrogate-preset", default=None,
                        help="UAntwerp surrogate preset to time, e.g. tripoli_full (needs its checkpoint)")
    args = parser.parse_args()

    default_threads = torch.get_num_threads()
    use_cuda = torch.cuda.is_available() and not args.cpu_only
    if not torch.cuda.is_available():
        print("CUDA not available: CPU only.")
    print(f"torch {torch.__version__}, default CPU threads {default_threads}"
          + (f", GPU {torch.cuda.get_device_name(0)}" if use_cuda else ""))
    # (label, device, CPU thread count or None)
    runs = [(f"cpu x{t}", "cpu", t) for t in (args.threads or [default_threads])]
    if use_cuda:
        runs.append(("cuda", "cuda", None))
    width = max(len(r[0]) for r in runs)

    def select(threads):
        """
        Apply a run's CPU thread count (CUDA runs: torch's default).
        """

        torch.set_num_threads(threads or default_threads)

    for num_envs in args.num_envs:
        for label, device, threads in runs:
            select(threads)
            ms = time_rollout_inference(device, args.obs_dim, args.act_dim, num_envs, args.inference_steps)
            print(f"[{label:{width}}] rollout inference, {num_envs} envs: {ms:.3f} ms per env step")
    for rollout_steps in args.rollout_steps:
        for batch_size in args.batch_size:
            for label, device, threads in runs:
                select(threads)
                sec = time_ppo_update(device, args.obs_dim, args.act_dim, rollout_steps, batch_size, args.epochs,
                                      args.repeats)
                print(f"[{label:{width}}] PPO update, {rollout_steps} transitions, batch {batch_size}, "
                      f"{args.epochs} epochs: {sec:.3f} s (median of {max(1, args.repeats)})")
    select(None)
    devices = ["cpu"] + (["cuda"] if use_cuda else [])
    if args.surrogate_preset:
        for device in devices:
            try:
                ms = time_surrogate(device, args.surrogate_preset, args.inference_steps)
            except FileNotFoundError as exc:
                print(f"UA surrogate skipped: checkpoint not found ({exc.filename}).")
                break
            print(f"[{device:4}] UA surrogate '{args.surrogate_preset}', 1 row: {ms:.3f} ms per SURROGATE_DT")


if __name__ == "__main__":
    main()
