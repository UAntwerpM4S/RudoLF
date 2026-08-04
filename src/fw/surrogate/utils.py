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
    Return the torch device to use.

    Args:
        prefer_gpu: If True (default) and a CUDA device is available,
            return it; otherwise fall back to CPU.

    Returns:
        `torch.device("cuda")` or `torch.device("cpu")`.
    """

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(state: dict, path: Path):
    """
    Save a checkpoint dict to disk, creating parent directories as needed.

    Args:
        state: Checkpoint payload to save (e.g. model/optimizer state).
        path: Destination file path; its parent directories are created
            if they don't already exist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path: Path, device: torch.device):
    """
    Load a checkpoint dict previously saved by `save_checkpoint`.

    Args:
        path: Path to the checkpoint file.
        device: Device to map the loaded tensors onto.

    Returns:
        The deserialized checkpoint object (as passed to `torch.save`).

    Raises:
        FileNotFoundError: If `path` does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
