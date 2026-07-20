"""Configuration for the HIFTool tensor-input ZSL integration."""

from __future__ import annotations

import argparse
from pathlib import Path


def args_parser():
    parser = argparse.ArgumentParser(description="HIFTool ZSL zero-shot sharpening")
    parser.add_argument("--sf", type=int, default=32, help="Requested HSI sharpening scale.")
    parser.add_argument("--hsi_channel", type=int, default=31)
    parser.add_argument("--msi_channel", type=int, default=3)
    parser.add_argument("--rank", type=int, default=10, help="HSI subspace rank used by released ZSL.")
    parser.add_argument("--zsl-steps", type=int, default=1000,
                        help="Total optimisation updates per image. Cascade stages split this evenly.")
    parser.add_argument("--zsl-lr", type=float, default=1e-3)
    parser.add_argument("--zsl-training-patch", type=int, default=32)
    parser.add_argument("--zsl-batch-size", type=int, default=64)
    parser.add_argument("--srfpath", type=str,
                        default=str(Path("Dataloader_tool") / "srflib" / "NikonD700.npy"))
    return parser.parse_known_args()[0]


# Backward compatibility with modules that import ``Networks.ZSL.config.opt``.
opt = args_parser()
