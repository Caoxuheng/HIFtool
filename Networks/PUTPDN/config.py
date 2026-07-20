"""HIFTool options for the PUT-PDN-MSI fusion baseline."""

from __future__ import annotations

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="PUT-PDN-MSI HIFTool adapter")
    parser.add_argument("--sf", type=int, default=32)
    parser.add_argument("--hsi_channel", type=int, default=31)
    parser.add_argument("--msi_channel", type=int, default=3)
    parser.add_argument("--put-stages", type=int, default=3)
    parser.add_argument("--put-neumann-terms", type=int, default=3)
    return parser.parse_known_args()[0]
