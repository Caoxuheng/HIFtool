"""Strictly load an existing HIFTool CaFormer checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from CaFormer_HIFTool_replacement.factory import build_caformer, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scale", type=int, default=32)
    parser.add_argument("--total-stages", type=int, default=3)
    args = parser.parse_args()
    model = build_caformer(
        scale=args.scale, total_stages=args.total_stages, device="cpu"
    )
    load_checkpoint(model, args.checkpoint, strict=True)
    print(
        {
            "checkpoint": str(args.checkpoint),
            "parameters": len(model.state_dict()),
            "strict_load": True,
        }
    )


if __name__ == "__main__":
    main()
