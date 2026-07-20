"""HIFTool wrapper around the official PUT-PDN Physics-Constrained PUT model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def _load_put():
    root = Path(__file__).resolve().parents[2]
    source_root = root / "external" / "comparison_official" / "PUT-PDN" / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(
            "Official PUT-PDN source was not found at " + str(source_root) +
            ". Clone https://github.com/XWangBin/PUT-PDN into external/comparison_official/PUT-PDN."
        )
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from put_pdn.models import PUT
    return PUT


class PUTPDN(nn.Module):
    """PUT with a fixed, explicit HR-MSI-to-PAN adapter for HIFTool data."""

    def __init__(self, options):
        super().__init__()
        if options.put_pan_mode != "mean":
            raise ValueError(f"Unsupported PAN proxy: {options.put_pan_mode}")
        PUT = _load_put()
        self.options = options
        self.model = PUT(
            ratio=options.sf,
            in_channels=1,
            out_channels=options.hsi_channel,
            stage=options.put_stages,
            channels=options.hsi_channel,
            neumann_terms=options.put_neumann_terms,
        )

    def pan_proxy(self, hr_msi: torch.Tensor) -> torch.Tensor:
        if hr_msi.shape[1] != self.options.msi_channel:
            raise ValueError(f"Expected {self.options.msi_channel} MSI channels, got {hr_msi.shape[1]}")
        return hr_msi.mean(dim=1, keepdim=True)

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor):
        prediction, reconstructed_pan, reconstructed_lr_hsi = self.model(
            lr_hsi, self.pan_proxy(hr_msi)
        )
        return prediction, reconstructed_pan, reconstructed_lr_hsi
