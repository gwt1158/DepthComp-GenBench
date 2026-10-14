"""Checkpoint I/O for the current GeoLayerNet module layout only."""

import torch

def load_checkpoint(model, path, strict=True, map_location="cpu"):
    """Load only current-layout weights; historical prefix remapping is forbidden."""
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=bool(strict))
    return model


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
