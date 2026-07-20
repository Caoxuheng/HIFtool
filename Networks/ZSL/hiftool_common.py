"""Shared CAVE x32 protocol, metrics, tiling, and checkpoint helpers."""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import Tensor
from torch.utils.data import DataLoader


# This module lives in ``Networks/UTAL`` when delivered.  Two parents up is
# the HIFTool workspace, not the ``Networks`` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = PROJECT_ROOT / "external" / "comparison_official"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATASET_ROOT = Path("D:/CaoXuheng/dataset")
TRAIN_IDS = tuple(range(18))
TEST_IDS = tuple(range(18, 31))


@dataclass
class RunSettings:
    method: str
    regime: str
    scale: int
    max_steps: int
    patch_size: int
    seed: int
    dataset_root: str
    output_dir: str
    source_commit: str
    source_url: str


SOURCE_MANIFEST = {
    "MoGDCN": ("HIFTool bundled Networks/MoGDCN", "workspace-snapshot"),
    "PSTUN": ("https://github.com/NIM-NMDC/PSTUN", "925caa6254d83296bd57f1c85286f1fb84ae3dce"),
    "SHOTUN": ("https://github.com/Shawn-H-Wang/SHOTUN", "d0bec639d1ed77150c0e4c2aaac3da876c9886bf"),
    "IR-ArF": ("https://github.com/Jiahuiqu/IR-ArF", "68d0924208c72ba1b129970b7ef3980dddc6a76c"),
    "EMR-Diff": ("https://github.com/luocz55/EMR-Diff", "cc851ec7b597bbe259c1cdc413a13e19a386d228"),
    "UAFL": ("https://github.com/yingkai-zhang/UAFL", "24c69ea7fdfc2d8e7f83d4aefa266ac86471c856"),
    "ZSL": ("https://github.com/renweidian/ZSL", "0fc2d8a2a6686077b75692e2d15241d0e6511d76"),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hif_options(dataset_root: Path, scale: int = 32) -> SimpleNamespace:
    return SimpleNamespace(
        sf=scale,
        hsi_channel=31,
        msi_channel=3,
        dataset_root=str(dataset_root),
        cave_split="sequential_60_40",
    )


def cave_loader(
    dataset_root: Path,
    split: str,
    patch_size: int,
    scale: int = 32,
    shuffle: bool = False,
    lazy: bool = True,
    batch_size: int = 1,
    drop_last: bool = False,
) -> DataLoader:
    from Dataloader_tool import Large_dataset

    dataset = Large_dataset(
        hif_options(dataset_root, scale), patch_size, "CAVE", type=split,
        crop=True, lazy=lazy,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def endless(loader: Iterable[Tuple[Tensor, Tensor, Tensor]]) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
    while True:
        yield from loader


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def candidate_patch_sizes(requested: int, scale: int = 32) -> List[int]:
    values = [requested, 256, 128, 64]
    return sorted({v for v in values if v <= requested and v >= 2 * scale and v % scale == 0}, reverse=True)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    settings: RunSettings,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "completed_steps": completed_steps,
        "settings": asdict(settings),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "extra": extra or {},
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    restore_rng: bool = True,
) -> Dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return state


def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


@torch.no_grad()
def metrics(prediction: Tensor, target: Tensor, scale: int) -> Dict[str, float]:
    prediction = prediction.detach().float().clamp(0.0, 1.0)
    target = target.detach().float()
    error = prediction - target
    mse = error.square().mean()
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    dot = (prediction * target).sum(dim=1)
    norms = prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    cosine = (dot / norms.clamp_min(1e-8)).clamp(-1 + 1e-7, 1 - 1e-7)
    sam = torch.rad2deg(torch.acos(cosine)).mean()
    band_rmse = error.square().mean(dim=(-2, -1)).sqrt()
    band_mean = target.mean(dim=(-2, -1)).abs().clamp_min(1e-6)
    ergas = 100.0 / scale * torch.sqrt(((band_rmse / band_mean).square()).mean())

    pred_np = prediction[0].cpu().numpy()
    target_np = target[0].cpu().numpy()
    ssim = float(np.mean([
        structural_similarity(target_np[idx], pred_np[idx], data_range=1.0)
        for idx in range(target_np.shape[0])
    ]))
    return {
        "psnr": float(psnr.item()),
        "sam": float(sam.item()),
        "ergas": float(ergas.item()),
        "ssim": ssim,
    }


@torch.no_grad()
def tiled_predict(
    predict: Callable[[Tensor, Tensor], Tensor],
    lrhsi: Tensor,
    hrmsi: Tensor,
    scale: int,
    tile_size: int,
) -> Tensor:
    height, width = hrmsi.shape[-2:]
    if tile_size >= height and tile_size >= width:
        return predict(lrhsi, hrmsi)
    if tile_size % scale or height % tile_size or width % tile_size:
        raise ValueError("tile_size must be scale-aligned and divide the CAVE image size")
    lr_tile = tile_size // scale
    output = torch.empty(
        (lrhsi.shape[0], lrhsi.shape[1], height, width),
        device=lrhsi.device,
        dtype=lrhsi.dtype,
    )
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            lr_patch = lrhsi[
                :, :, top // scale:top // scale + lr_tile,
                left // scale:left // scale + lr_tile,
            ]
            ms_patch = hrmsi[:, :, top:top + tile_size, left:left + tile_size]
            output[:, :, top:top + tile_size, left:left + tile_size] = predict(lr_patch, ms_patch)
    return output


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, float]:
    metric_names = ("psnr", "sam", "ergas", "ssim")
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names}


class Stopwatch:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.started
