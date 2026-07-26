"""One-step uHNTC blind-path smoke test inside a complete HIFTool checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
HIFTOOL_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PACKAGE_PARENT), str(HIFTOOL_ROOT)]

from CaFormer_HIFTool_replacement import BlindCaFormer, UHNTCConfig, build_caformer


def main() -> None:
    torch.manual_seed(7)
    model = build_caformer(
        scale=2,
        total_stages=1,
        n_feat=8,
        n_depth=2,
        device="cpu",
    )
    config = UHNTCConfig(
        scale=2,
        spe_init_steps=1,
        spa_init_steps=1,
        couple_init_steps=1,
        adaptation_steps=1,
        tile_size=4,
    )
    lr_hsi = torch.rand(1, 31, 4, 4)
    hr_msi = torch.rand(1, 3, 8, 8)
    result = BlindCaFormer(model, config)(lr_hsi, hr_msi)
    assert result.reconstruction.shape == (1, 31, 8, 8)
    assert result.diagnostics["uses_ground_truth"] == 0
    assert result.diagnostics["uses_true_degradation"] == 0
    assert result.diagnostics["adaptation_tiles"] == 4
    print(result.diagnostics)
    print("uHNTC blind CaFormer smoke test passed.")


if __name__ == "__main__":
    main()
