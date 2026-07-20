"""Evaluate integrated ZSL on HIFTool CAVE data without any HR-HSI optimisation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataloader_tool import Large_dataset
from comparison_benchmarks.common import metrics
from Networks import model_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HIFTool's per-scene ZSL / high-scale cascade.")
    parser.add_argument("--dataset-root", default="D:/CaoXuheng/dataset")
    parser.add_argument("--cave-split", default="sequential_60_40", choices=("official", "sequential_60_40"))
    parser.add_argument("--output-root", type=Path, default=Path("Benchmarks/CAVE_sequential_60_40/ZSL"))
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--device", default="cuda")
    # ZSL-specific arguments are intentionally parsed by Networks.ZSL.config
    # via parse_known_args, including --sf and --zsl-steps.
    args, _ = parser.parse_known_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical per-scene ZSL adaptation.")
    device = torch.device(args.device)
    model, opt = model_generator("ZSL", str(device))
    opt.dataset_root = args.dataset_root
    opt.cave_split = args.cave_split
    dataset = Large_dataset(opt, 512, "CAVE", type="test", crop=True, lazy=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, (target, lrhsi, hrmsi) in enumerate(loader):
        if args.max_scenes is not None and position >= args.max_scenes:
            break
        sample_id = dataset.test_name[position]
        prediction = model(lrhsi.to(device), hrmsi.to(device))
        score = {"scene": sample_id, **metrics(prediction, target.to(device), opt.sf)}
        scene_dir = args.output_root / f"scene_{sample_id}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(scene_dir / "prediction.npy", prediction[0].cpu().numpy())
        (scene_dir / "metrics.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
        rows.append(score)
        print(f"scene {sample_id}: PSNR={score['psnr']:.3f}, SAM={score['sam']:.3f}", flush=True)
    if rows:
        summary = {name: float(np.mean([row[name] for row in rows])) for name in ("psnr", "sam", "ergas", "ssim")}
        summary.update({"method": "ZSL", "scale": opt.sf, "scenes": [row["scene"] for row in rows]})
        (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
