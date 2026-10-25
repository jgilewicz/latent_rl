import torch


def resolve_device(name: str, config_key: str) -> torch.device:
    if name not in ("cpu", "cuda"):
        raise ValueError(f"{config_key} must be 'cpu' or 'cuda', got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{config_key} 'cuda' requested but CUDA is unavailable; "
            f"set {config_key}: cpu in the config"
        )
    return torch.device(name)
