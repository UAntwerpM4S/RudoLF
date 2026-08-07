import torch

from pathlib import Path


def data_path(*relative):
    """
    Return an absolute path inside the local data/ folder (project-local).
    """

    base = Path(__file__).resolve().parent
    return base / "data" / Path(*relative)

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Select the torch device to use for surrogate training/inference.

    Args:
        prefer_gpu: If True, use CUDA when available; otherwise always use CPU.

    Returns:
        torch.device: "cuda" when `prefer_gpu` is True and CUDA is
            available, otherwise "cpu".
    """

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(state: dict, path: Path):
    """
    Save a training checkpoint to disk, creating parent directories as needed.

    Args:
        state: Checkpoint payload (e.g. model/optimizer state dicts,
            normalization stats, architecture parameters) to serialize.
        path: Destination file path for the checkpoint.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path: Path, device: torch.device):
    """
    Load a previously saved training checkpoint from disk.

    Args:
        path: Path to the checkpoint file.
        device: Torch device to map the loaded tensors onto.

    Returns:
        The deserialized checkpoint payload (same structure passed to
        `save_checkpoint`).

    Raises:
        FileNotFoundError: If `path` does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
