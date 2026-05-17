"""
Transfer learning from 2D PPO policies to AirSimEnv (3D).

Only used by AirSimEnv training — has no effect on 2D environment training.

Tasks implemented:
    T-054: Network topology comparison (2D vs 3D input dims, layers, neurons, activations).
    T-055: Incompatibility reporting with clear error messages.
    T-056: 2D weight loading with layer matching and selective backbone freezing.
    T-057: Implicit mapping layer for extra 3D inputs via the untrained first linear
           layer, controlled by the ``transfer_learning`` config block.

Mapping layer strategy
----------------------
The ``ActorCriticNetwork`` first layer ``shared_network[0]: Linear(input_dim → 256)``
differs in shape between 2D (e.g. input_dim=10) and 3D (input_dim=18), so it is
**not** transferred.  Instead, it is left randomly initialized and kept trainable —
acting as the implicit mapping layer that learns to project the extra AirSim
observation dimensions into the feature space that the pretrained deeper layers
already understand.

Freeze strategy
---------------
When ``freeze_shared=True``, only the backbone layer ``shared_network[2]: 256→128``
is frozen so the pre-trained feature representation is preserved.  The actor and
critic heads (``actor_mean``, ``critic``) remain **trainable** so they can
fine-tune to the new environment's reward structure and 3D physics. Freezing
the heads would prevent the policy distribution from adapting, causing
convergence failure.

Exploration strategy
--------------------
``actor_log_std`` is **not** transferred. A converged 2D policy can collapse
the log_std for some action dimensions (e.g. rudder) to highly negative values,
producing near-deterministic actions. Loading that into AirSim freezes
exploration on those dimensions and PPO's small clip_range cannot recover.
The target policy keeps its freshly-initialized ``log_std = -0.5`` so it
re-explores in the new environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from fw.policies.ppo_policy import PPOPolicy, get_device


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TransferIncompatibilityError(ValueError):
    """ Raised when a 2D policy cannot be transferred to the target 3D dimensions. """


# ---------------------------------------------------------------------------
# T-054 — topology comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyReport:
    """
    Comparison of source (2D) and target (3D) network topology.

    Attributes:
        source_input_dim: Observation dimension of the pretrained 2D policy.
        target_input_dim: Observation dimension of the 3D AirSimEnv policy.
        source_output_dim: Action dimension of the pretrained 2D policy.
        target_output_dim: Action dimension of the 3D AirSimEnv policy.
        extra_input_dims: Additional observation features in the 3D environment.
        shared_layer_shapes: ``(in_features, out_features)`` for each
            transferable hidden layer (excludes the first input layer).
        output_compatible: ``True`` when action dimensions match.
    """

    source_input_dim: int
    target_input_dim: int
    source_output_dim: int
    target_output_dim: int
    extra_input_dims: int
    shared_layer_shapes: List[Tuple[int, int]]
    output_compatible: bool

    def format(self) -> str:
        """
        Return a human-readable topology comparison string.
        """

        lines = [
            "=" * 58,
            "  Transfer Learning — Network Topology Comparison",
            "=" * 58,
            f"  Source input dim   : {self.source_input_dim}  (2D policy)",
            f"  Target input dim   : {self.target_input_dim}  (AirSimEnv)",
            f"  Extra input dims   : {self.extra_input_dims}  (handled by mapping layer)",
            f"  Source output dim  : {self.source_output_dim}",
            f"  Target output dim  : {self.target_output_dim}",
            f"  Output compatible  : {self.output_compatible}",
            "  Transferable hidden layers (shared_network[2], actor, critic):",
        ]
        for i, (in_f, out_f) in enumerate(self.shared_layer_shapes):
            lines.append(f"    [{i}] Linear({in_f} → {out_f})")
        lines += [
            "  Mapping layer      : shared_network[0] — random-init, trainable",
            f"                       Linear({self.target_input_dim} → 256)",
            "=" * 58,
        ]
        return "\n".join(lines)


def compare_network_topology(
    source_policy: PPOPolicy,
    target_input_dim: int,
    target_output_dim: int,
) -> TopologyReport:
    """
    Compare a 2D source policy topology against the 3D AirSimEnv dimensions.

    Args:
        source_policy: Pretrained 2D ``PPOPolicy`` instance.
        target_input_dim: Observation dimension of the 3D AirSimEnv.
        target_output_dim: Action dimension of the 3D AirSimEnv.

    Returns:
        ``TopologyReport`` describing transferable and incompatible elements.
    """

    sd = source_policy.network.state_dict()

    shared_layers: List[Tuple[int, int]] = [
        (
            int(sd["shared_network.2.weight"].shape[1]),
            int(sd["shared_network.2.weight"].shape[0]),
        ),
    ]

    return TopologyReport(
        source_input_dim=int(source_policy.input_dim),
        target_input_dim=int(target_input_dim),
        source_output_dim=int(source_policy.output_dim),
        target_output_dim=int(target_output_dim),
        extra_input_dims=int(target_input_dim) - int(source_policy.input_dim),
        shared_layer_shapes=shared_layers,
        output_compatible=(int(source_policy.output_dim) == int(target_output_dim)),
    )


# ---------------------------------------------------------------------------
# T-055 — incompatibility reporting
# ---------------------------------------------------------------------------


def validate_transfer_compatibility(report: TopologyReport) -> None:
    """
    Raise `TransferIncompatibilityError` with a clear message if transfer is blocked.

    Args:
        report: Topology comparison from :func:`compare_network_topology`.

    Raises:
        TransferIncompatibilityError: When action dimensions differ or the target
            observation space is smaller than the source.
    """

    errors: List[str] = []

    if not report.output_compatible:
        errors.append(
            f"Action dimension mismatch: source has {report.source_output_dim} output(s), "
            f"target expects {report.target_output_dim}. "
            "Transfer requires identical action spaces."
        )

    if report.extra_input_dims < 0:
        errors.append(
            f"Target input dimension ({report.target_input_dim}) is smaller than the "
            f"source ({report.source_input_dim}). "
            "Transfer requires target_input_dim >= source_input_dim."
        )

    if errors:
        raise TransferIncompatibilityError(
            "Cannot transfer 2D policy to 3D AirSimEnv:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# T-056 — weight loading with layer matching and freezing
# ---------------------------------------------------------------------------

# NOTE: ``actor_log_std`` is intentionally NOT transferred. A converged 2D
# policy can collapse its log_std for some action dimensions to very negative
# values (near-deterministic actions). Copying that into AirSim freezes
# exploration on those dimensions, the agent then dies at the same pose every
# episode, advantage estimates flatten to ~0, and PPO (with its small
# clip_range) cannot recover. Keeping the target's freshly-initialized
# ``log_std = -0.5`` restores exploration in the new environment.
_TRANSFERABLE_KEYS = (
    "shared_network.0.weight",
    "shared_network.0.bias",
    "shared_network.2.weight",
    "shared_network.2.bias",
    "actor_mean.weight",
    "actor_mean.bias",
    "critic.weight",
    "critic.bias",
)

# Only the backbone hidden layer is frozen when freeze_shared=True.
# The actor and critic heads must stay trainable so the policy distribution
# can adapt to AirSim's reward structure and 3D physics.
_BACKBONE_KEYS = frozenset({"shared_network.2.weight", "shared_network.2.bias"})


def apply_transfer_weights(
    source_policy: PPOPolicy,
    target_policy: PPOPolicy,
    freeze_shared: bool = True,
) -> List[str]:
    """
    Copy matching weights from *source_policy* to *target_policy*.

    Only layers whose parameter tensors have identical shapes in source and
    target are transferred. The first layer ``shared_network[0]`` (the mapping
    layer) is intentionally skipped because it differs in input dimension —
    it remains randomly initialized and trainable to adapt the extra 3D inputs.

    Args:
        source_policy: Pretrained 2D policy.
        target_policy: Freshly initialized 3D AirSimEnv policy.
        freeze_shared: If ``True``, freeze only ``shared_network[2]`` (the
            backbone hidden layer) so the pretrained feature representation is
            preserved. The actor and critic heads remain trainable so the policy
            distribution can adapt to AirSim's reward structure and 3D physics.
            If ``False``, all layers remain trainable for full fine-tuning.

    Returns:
        List of state-dict keys that were successfully transferred.
    """

    src_sd = source_policy.network.state_dict()
    tgt_sd = target_policy.network.state_dict()

    transferred: List[str] = []
    for key in _TRANSFERABLE_KEYS:
        if key not in src_sd or key not in tgt_sd:
            continue
        if src_sd[key].shape != tgt_sd[key].shape:
            continue
        # Move to target device before copying so CPU→CUDA transfers work correctly.
        tgt_sd[key] = src_sd[key].to(tgt_sd[key].device).clone()
        transferred.append(key)

    target_policy.network.load_state_dict(tgt_sd)

    if freeze_shared:
        for name, param in target_policy.network.named_parameters():
            if name in _BACKBONE_KEYS:
                param.requires_grad_(False)

    return transferred


# ---------------------------------------------------------------------------
# T-057 — full pipeline: mapping layer + config integration
# ---------------------------------------------------------------------------


def build_transfer_policy(
    source_zip_path: str | Path,
    target_input_dim: int,
    target_output_dim: int,
    target_device: torch.device | None = None,
    freeze_shared: bool = True,
) -> PPOPolicy:
    """
    Build an AirSimEnv policy initialized from a pretrained 2D checkpoint.

    Pipeline:
        1. Load the 2D source policy on CPU (weight extraction only — no GPU needed).
        2. Compare topology and validate compatibility (raises on mismatch).
        3. Initialize a fresh target ``PPOPolicy`` on *target_device*.
        4. Transfer all matching hidden layers, moving tensors to *target_device* automatically.
        5. The first linear layer ``shared_network[0]: Linear(target_input_dim → 256)``
           remains randomly initialized and trainable — this is the mapping layer
           that learns to project the extra 3D observation dimensions into the feature
           space that the pretrained deeper layers understand.

    The source checkpoint is always loaded on CPU regardless of *target_device*, so any
    checkpoint (CPU or CUDA) can be transferred to any training device without manual
    conversion.  When *target_device* is ``None`` the device is auto-detected (CUDA if
    available, otherwise CPU).

    Args:
        source_zip_path: Path to the 2D ``.zip`` policy checkpoint.
        target_input_dim: Observation dimension of AirSimEnv (3D).
        target_output_dim: Action dimension of AirSimEnv.
        target_device: PyTorch device for the target policy.  ``None`` triggers
            auto-detection via :func:`~fw.policies.ppo_policy.get_device`.
        freeze_shared: Freeze transferred layers so only the mapping layer trains.

    Returns:
        ``PPOPolicy`` on *target_device*, configured for AirSimEnv fine-tuning.

    Raises:
        TransferIncompatibilityError: If action dims differ or target < source input dim.
        RuntimeError: If the source checkpoint cannot be loaded.
    """

    resolved_device = target_device if target_device is not None else get_device("auto")
    source_policy = PPOPolicy.load(str(source_zip_path), device="cpu")

    report = compare_network_topology(source_policy, target_input_dim, target_output_dim)
    print(report.format())
    validate_transfer_compatibility(report)

    target_policy = PPOPolicy(target_input_dim, target_output_dim, device=resolved_device.type)
    transferred = apply_transfer_weights(source_policy, target_policy, freeze_shared=freeze_shared)

    n_trainable = sum(p.numel() for p in target_policy.network.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in target_policy.network.parameters())
    frozen_note = "backbone frozen, mapping+heads trainable" if freeze_shared else "full network"
    print(
        f"[TransferLearning] Source device : cpu (weight extraction)"
        f"\n[TransferLearning] Target device : {resolved_device}"
        f"\n[TransferLearning] Transferred {len(transferred)} layer(s): {transferred}"
        f"\n[TransferLearning] Trainable: {n_trainable:,} / {n_total:,} params ({frozen_note})"
    )

    return target_policy
