"""Small construction helpers for HIFTool loaders and standalone use."""

from __future__ import annotations

from pathlib import Path

import torch

from .blind import BlindCaFormer, UHNTCConfig
from .net import CaFormer


def build_caformer(
    scale: int,
    hsi_channels: int = 31,
    msi_channels: int = 3,
    total_stages: int = 3,
    n_feat: int = 64,
    n_depth: int = 3,
    device: str | torch.device = "cuda",
) -> CaFormer:
    if total_stages < 1:
        raise ValueError("total_stages must be at least 1")
    return CaFormer(
        sf=scale,
        in_c=hsi_channels,
        out_c=msi_channels,
        n_feat=n_feat,
        nums_stages=total_stages - 1,
        n_depth=n_depth,
    ).to(device)


def load_checkpoint(
    model: CaFormer,
    checkpoint: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CaFormer:
    state = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state_dict = state["net"] if isinstance(state, dict) and "net" in state else state
    model.load_state_dict(state_dict, strict=strict)
    return model


def build_blind_caformer(
    checkpoint: str | Path,
    config: UHNTCConfig,
    total_stages: int = 3,
    n_feat: int = 64,
    n_depth: int = 3,
    device: str | torch.device = "cuda",
) -> BlindCaFormer:
    model = build_caformer(
        scale=config.scale,
        hsi_channels=config.hsi_channels,
        msi_channels=config.msi_channels,
        total_stages=total_stages,
        n_feat=n_feat,
        n_depth=n_depth,
        device=device,
    )
    load_checkpoint(model, checkpoint, map_location=device)
    return BlindCaFormer(model, config)
