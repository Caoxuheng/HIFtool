"""Compare the replacement with the installed CaFormer on a standard input."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
HIFTOOL_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PACKAGE_PARENT), str(HIFTOOL_ROOT)]

from CaFormer_HIFTool_replacement.net import CaFormer as Replacement
from Networks.CaFormer.net import CaFormer as Installed


def main() -> None:
    torch.manual_seed(11)
    installed = Installed(
        sf=32, in_c=31, out_c=3, n_feat=8, nums_stages=1, n_depth=2
    ).eval()
    replacement = Replacement(
        sf=32, in_c=31, out_c=3, n_feat=8, nums_stages=1, n_depth=2
    ).eval()
    replacement.load_state_dict(installed.state_dict(), strict=True)
    lr_hsi = torch.rand(1, 31, 2, 2)
    hr_msi = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = installed(lr_hsi, hr_msi)
        actual = replacement(lr_hsi, hr_msi)
    maximum_error = float((expected - actual).abs().max())
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    print({"standard_input_max_abs_error": maximum_error, "compatible": True})


if __name__ == "__main__":
    main()
