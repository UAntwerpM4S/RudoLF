"""Transfer learning from 2D PPO policies to AirSimEnv (3D).

Only used by AirSimEnv training — has no effect on 2D environment training.

Tasks implemented:
    T-054: Network topology comparison (2D vs 3D input dims, layers, neurons, activations).
    T-055: Incompatibility reporting with clear error messages.
    T-056: 2D weight loading with layer matching and shared-layer freezing.
    T-057: Implicit mapping layer for extra 3D inputs via the untrained first linear
           layer, controlled by the ``transfer_learning`` config block.

Mapping layer strategy
----------------------
The ``ActorCriticNetwork`` first layer ``shared_network[0]: Linear(input_dim → 256)``
differs in shape between 2D (e.g. input_dim=10) and 3D (input_dim=18), so it is
**not** transferred.  Instead it is left randomly initialised and kept trainable —
acting as the implicit mapping layer that learns to project the extra AirSim
observation dimensions into the feature space that the pretrained deeper layers
already understand.

All deeper shared layers (``shared_network[2]: 256→128``), the actor head, and the
critic head have identical shapes in 2D and 3D networks and are transferred + frozen
so that only the mapping-layer first layer adapts during fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from fw.policies.ppo_policy import PPOPolicy, get_device


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TransferIncompatibilityError(ValueError):
    """Raised when a 2D policy cannot be transferred to the target 3D dimensions."""


# ---------------------------------------------------------------------------
# T-054 — topology comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyReport:
    """Comparison of source (2D) and target (3D) network topology.

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
        """Return a human-readable topology comparison string."""
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
        if self.extra_input_dims > 0:
            mapping_desc = "shared_network[0] kept as random-init,"
            mapping_detail = f"Linear({self.target_input_dim} → 256) — trainable"
        else:
            mapping_desc = "shared_network[0] transferred from source,"
            mapping_detail = f"Linear({self.target_input_dim} → 256) — trainable"
        lines += [
            f"  Mapping layer      : {mapping_desc}",
            f"                       {mapping_detail}",
            "=" * 58,
        ]
        return "\n".join(lines)


def compare_network_topology(
    source_policy: PPOPolicy,
    target_input_dim: int,
    target_output_dim: int,
) -> TopologyReport:
    """Compare a 2D source policy topology against the 3D AirSimEnv dimensions.

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
    """Raise ``TransferIncompatibilityError`` with a clear message if transfer is blocked.

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

_TRANSFERABLE_KEYS = (
    "shared_network.0.weight",
    "shared_network.0.bias",
    "shared_network.2.weight",
    "shared_network.2.bias",
    "actor_mean.weight",
    "actor_mean.bias",
    "actor_log_std",
    "critic.weight",
    "critic.bias",
)

# Mapping-layer keys are transferred when dims match but must never be frozen —
# they are always the trainable adapter regardless of freeze_shared.
_MAPPING_LAYER_KEYS = frozenset({"shared_network.0.weight", "shared_network.0.bias"})


def apply_transfer_weights(
    source_policy: PPOPolicy,
    target_policy: PPOPolicy,
    freeze_shared: bool = True,
) -> List[str]:
    """Copy matching weights from *source_policy* to *target_policy*.

    Only layers whose parameter tensors have identical shapes in source and
    target are transferred (``shared_network[2]``, actor head, critic head).
    The first layer ``shared_network[0]`` is always skipped — it is the
    mapping layer and must remain trainable regardless of dimension match.

    Args:
        source_policy: Pretrained 2D policy.
        target_policy: Freshly initialised 3D AirSimEnv policy.
        freeze_shared: If ``True``, freeze all transferred parameters so only
            the mapping-layer first linear trains during fine-tuning.

    Returns:
        List of state-dict keys that were successfully transferred.
    """
    src_sd = source_policy.network.state_dict()
    tgt_sd = target_policy.network.state_dict()

    transferred: List[str] = []
    for key in _TRANSFERABLE_KEYS:
        if key not in src_sd or key not in tgt_sd:
            continue
        # Treat mapping-layer weight+bias as a unit: skip bias if weight was not
        # transferred (e.g. different input dims), so the layer is never partially applied.
        if key == "shared_network.0.bias" and "shared_network.0.weight" not in transferred:
            continue
        if src_sd[key].shape != tgt_sd[key].shape:
            continue
        # Move to target device before copying so CPU→CUDA transfers work correctly.
        tgt_sd[key] = src_sd[key].to(tgt_sd[key].device).clone()
        transferred.append(key)

    target_policy.network.load_state_dict(tgt_sd)

    if freeze_shared:
        transferred_set = set(transferred)
        for name, param in target_policy.network.named_parameters():
            if name in transferred_set and name not in _MAPPING_LAYER_KEYS:
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
    mapping_layer_init_gain: float = 0.1,
) -> PPOPolicy:
    """Build an AirSimEnv policy initialised from a pretrained 2D checkpoint.

    Pipeline:
        1. Load the 2D source policy on CPU (weight extraction only — no GPU needed).
        2. Compare topology and validate compatibility (raises on mismatch).
        3. Initialise a fresh target ``PPOPolicy`` on *target_device*.
        4. Transfer all matching hidden layers, moving tensors to *target_device* automatically.
        5. The first linear layer ``shared_network[0]: Linear(target_input_dim → 256)``
           is the mapping layer. It is reinitialised with a reduced orthogonal gain
           (default 0.1) so its initial outputs are smaller, dampening the activation
           distribution mismatch with the pre-trained deep layers without killing
           the gradient signal the way gain=0.01 does.

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
        mapping_layer_init_gain: Orthogonal init gain for the mapping layer.
            0.1 (default) reduces activation mismatch with pre-trained deep layers.
            Use smaller values to further suppress initial gradient explosion at the
            cost of slower early learning.

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

    mapping_layer_transferred = "shared_network.0.weight" in transferred
    if not mapping_layer_transferred:
        # Dims differ (e.g. 12→18): mapping layer was not transferred, apply
        # controlled orthogonal init to dampen activation mismatch on first forward pass.
        mapping_layer = target_policy.network.shared_network[0]
        nn.init.orthogonal_(mapping_layer.weight, gain=mapping_layer_init_gain)
        nn.init.constant_(mapping_layer.bias, 0.0)

    n_trainable = sum(p.numel() for p in target_policy.network.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in target_policy.network.parameters())
    frozen_note = "mapping layer only" if freeze_shared else "full network"
    mapping_note = "transferred from source" if mapping_layer_transferred else f"random-init (gain={mapping_layer_init_gain})"
    print(
        f"[TransferLearning] Source device : cpu (weight extraction)"
        f"\n[TransferLearning] Target device : {resolved_device}"
        f"\n[TransferLearning] Transferred {len(transferred)} layer(s): {transferred}"
        f"\n[TransferLearning] Trainable: {n_trainable:,} / {n_total:,} params ({frozen_note})"
        f"\n[TransferLearning] Mapping layer: {mapping_note}"
    )

    return target_policy
