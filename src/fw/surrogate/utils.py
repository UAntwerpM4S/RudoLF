import torch
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def data_path(*relative):
    """
    Absolute path inside fw/surrogate/data/.
    """

    base = Path(__file__).resolve().parent
    return base / "data" / Path(*relative)

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Return cuda if prefer_gpu and CUDA is available, else cpu.
    """

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(state: dict, path: Path):
    """
    Save a checkpoint dict with torch.save, creating parent directories.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)

def load_checkpoint(path: Path, device: torch.device):
    """
    Load a checkpoint saved by save_checkpoint, mapped onto device.

    Raises:
        FileNotFoundError: If path does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
