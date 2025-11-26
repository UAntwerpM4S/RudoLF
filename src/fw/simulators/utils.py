import os
import torch

from pathlib import Path

# Directory where THIS file lives
BASE_DIR = Path(__file__).resolve().parent

def data_path(*relative):
    """Return an absolute path inside the local data/ folder."""
    return BASE_DIR / "data" / Path(*relative)

def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded checkpoint: {path}")
