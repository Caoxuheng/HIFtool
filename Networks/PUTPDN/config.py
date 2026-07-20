"""HIFTool options for the official PUT-PDN fusion backbone."""

from __future__ import annotations

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="PUT-PDN HIFTool adapter")
    parser.add_argument("--sf", type=int, default=32)
    parser.add_argument("--hsi_channel", type=int, default=31)
    parser.add_argument("--msi_channel", type=int, default=3)
    parser.add_argument("--put-stages", type=int, default=3)
    parser.add_argument("--put-neumann-terms", type=int, default=3)
    parser.add_argument("--put-pan-mode", choices=("mean",), default="mean",
                        help="Fixed HR-MSI to single-PAN proxy; PUT itself is a PAN method.")
    return parser.parse_known_args()[0]
