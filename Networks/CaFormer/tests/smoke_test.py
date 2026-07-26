"""CPU smoke tests; no dataset or checkpoint is required."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from CaFormer_HIFTool_replacement import build_caformer, forward_tiled
from CaFormer_HIFTool_replacement.blind import default_spectral_spans


def main() -> None:
    torch.manual_seed(1)
    for scale in (2, 4, 8, 16, 32):
        model = build_caformer(
            scale=scale,
            hsi_channels=31,
            msi_channels=3,
            total_stages=1,
            n_feat=8,
            n_depth=2,
            device="cpu",
        ).eval()
        lr_hsi = torch.rand(1, 31, 2, 3)
        hr_msi = torch.rand(1, 3, 2 * scale + 1, 3 * scale + 3)
        with torch.no_grad():
            prediction = model(lr_hsi, hr_msi)
        assert prediction.shape == (1, 31, *hr_msi.shape[-2:])
        print({"scale": scale, "shape": tuple(prediction.shape)})

    model = build_caformer(
        scale=4,
        total_stages=1,
        n_feat=8,
        n_depth=2,
        device="cpu",
    ).eval()
    lr_hsi = torch.rand(1, 31, 16, 24)
    hr_msi = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        prediction = forward_tiled(model, lr_hsi, hr_msi, 4, 32, 8)
    assert prediction.shape == (1, 31, 64, 96)
    assert default_spectral_spans(31, 3) == [
        list(range(18, 31)),
        list(range(10, 23)),
        list(range(0, 12)),
    ]
    print("CaFormer replacement smoke tests passed.")


if __name__ == "__main__":
    main()
