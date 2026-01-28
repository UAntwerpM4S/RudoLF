import torch

from pathlib import Path


def data_path(*relative):
    """Return an absolute path inside the local data/ folder (project-local)."""
    base = Path(__file__).resolve().parent
    return base / "data" / Path(*relative)

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
