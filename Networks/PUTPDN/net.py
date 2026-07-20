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
    """PUT-PDN-MSI: the PUT architecture with a native three-channel MSI guide."""

    def __init__(self, options):
        super().__init__()
        PUT = _load_put()
        self.options = options
        self.model = PUT(
            ratio=options.sf,
            in_channels=options.msi_channel,
            out_channels=options.hsi_channel,
            stage=options.put_stages,
            channels=options.hsi_channel,
            neumann_terms=options.put_neumann_terms,
        )

    def msi_guide(self, hr_msi: torch.Tensor) -> torch.Tensor:
        if hr_msi.shape[1] != self.options.msi_channel:
            raise ValueError(f"Expected {self.options.msi_channel} MSI channels, got {hr_msi.shape[1]}")
        return hr_msi

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor):
        prediction, reconstructed_msi, reconstructed_lr_hsi = self.model(
            lr_hsi, self.msi_guide(hr_msi)
        )
        return prediction, reconstructed_msi, reconstructed_lr_hsi
