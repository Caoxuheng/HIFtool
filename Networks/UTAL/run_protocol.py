"""Leak-free UTAL protocol for the CAVE sequential-60/40 x32 benchmark.

This runner follows UTAL's three stages: supervised fusion-prior training on
the training split, synthetic-task meta-training of its adaptor on that same
split, then independent test-time self-supervision for every held-out scene.
Ground truth from the test split is read only after adaptation to calculate
metrics; it is never an input to an optimisation step.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import torch
from torch import nn

from hiftool_common import (
    DEFAULT_DATASET_ROOT,
    TEST_IDS,
    TRAIN_IDS,
    cave_loader,
    metrics,
    seed_everything,
    write_json,
)

# ``run_protocol.py`` is installed as ``Networks/UTAL/run_protocol.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://github.com/JiangtaoNie/UAL"
SOURCE_REVISION = "master (repository has 7 commits; accessed 2026-07-20)"


def utal_options(scale: int, patch_size: int, dataset_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        sf=scale,
        msi_channel=3,
        hsi_channel=31,
        h_size=[patch_size, patch_size],
        Depth=3,
        KS_1=3,
        KS_2=3,
        KS_3=3,
        dataset_root=str(dataset_root),
        cave_split="sequential_60_40",
    )


def build_fusion(opt: SimpleNamespace, device: torch.device) -> nn.Module:
    from Networks.UTAL.net import ThreeBranch_Net

    return ThreeBranch_Net(opt, str(device)).to(device)


def build_adaptor(device: torch.device) -> nn.Module:
    from Networks.UTAL.net import FineNet_SelfAtt_InputK_P_V2

    return FineNet_SelfAtt_InputK_P_V2().to(device)


def canonical_srf(device: torch.device) -> torch.Tensor:
    path = PROJECT_ROOT / "Networks" / "UTAL" / "knowledge" / "P_N_V2.mat"
    matrix = sio.loadmat(path.as_posix())["P"].astype(np.float32)
    return torch.from_numpy(matrix).to(device)


def make_spatial_estimator(scale: int, device: torch.device) -> tuple[nn.Module, nn.Conv2d]:
    """The learnable spatial degradation module used by the released adaptor."""
    from Networks.UTAL.Model.function import Apply
    from Networks.UTAL.Model.Spa_downs import get_kernel

    kernel_size = 32
    kernel = torch.from_numpy(
        get_kernel(scale, "gauss", 0, kernel_size, sigma=3)
    ).float()
    conv = nn.Conv2d(1, 1, kernel_size, scale)
    with torch.no_grad():
        conv.weight.copy_(kernel[None, None])
    module = Apply(nn.Sequential(nn.ReplicationPad2d((kernel_size - scale) // 2), conv), 1)
    return module.to(device), conv


def random_synthetic_pair(
    ground_truth: torch.Tensor, scale: int, srf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, nn.Module]:
    """Released UTAL-style random spatial and spectral synthetic task."""
    from Networks.UTAL.Model.Spa_downs import Spa_Downs

    choices = ((7, 0.5), (8, 3.0), (9, 2.0), (13, 4.0), (15, 1.5))
    width, sigma = choices[np.random.randint(len(choices))]
    down = Spa_Downs(
        31, scale, kernel_type=np.random.randint(2, 9), kernel_width=width,
        sigma=sigma, preserve_size=True,
    ).to(ground_truth.device)
    lr_hsi = down(ground_truth)
    if torch.rand((), device=ground_truth.device) > 0.3:
        perturbation = 1e-4 * torch.randint(50, 80, (), device=ground_truth.device)
        task_srf = (srf + perturbation) / (srf.sum(1, keepdim=True) + perturbation * 31)
    else:
        task_srf = srf
    hr_msi = torch.matmul(
        task_srf.unsqueeze(0), ground_truth.flatten(2)
    ).reshape(ground_truth.shape[0], 3, ground_truth.shape[-2], ground_truth.shape[-1])

    # FineNet receives the physical operators as compact conditioning inputs.
    spatial_kernel = down.kernel
    pad_left = (32 - spatial_kernel.shape[0]) // 2
    pad_right = 32 - spatial_kernel.shape[0] - pad_left
    spatial_kernel = torch.from_numpy(spatial_kernel).to(ground_truth.device).float()
    spatial_kernel = nn.functional.pad(spatial_kernel, (pad_left, pad_right, pad_left, pad_right))
    kernel_input = spatial_kernel[None, None]
    # FineNet's released linear layer consumes the MSI dimension last, hence
    # it receives P^T (31 x 3), while the degradation loss uses P (3 x 31).
    spectral_input = task_srf.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    down.requires_grad_(False)
    return lr_hsi, hr_msi, kernel_input, spectral_input, down


def save_state(path: Path, **state: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_state(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_fusion(args: argparse.Namespace, run_dir: Path, device: torch.device) -> Path:
    checkpoint = run_dir / "fusion_last.pth"
    options = utal_options(args.scale, args.train_patch_size, args.dataset_root)
    model = build_fusion(options, device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.fusion_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.7)
    start_epoch = 0
    if args.resume and checkpoint.exists():
        state = load_state(checkpoint, device)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        print(f"Resuming fusion prior at epoch {start_epoch}.")

    loader = cave_loader(
        args.dataset_root, "train", args.train_patch_size, args.scale,
        shuffle=True, lazy=True, batch_size=args.batch_size,
    )
    l1 = nn.L1Loss()
    for epoch in range(start_epoch + 1, args.pretrain_epochs + 1):
        model.train()
        losses = []
        for ground_truth, lr_hsi, hr_msi in loader:
            ground_truth = ground_truth.to(device, non_blocking=True)
            lr_hsi = lr_hsi.to(device, non_blocking=True)
            hr_msi = hr_msi.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            output = model(lr_hsi, hr_msi)
            loss = l1(output, ground_truth)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record = {"epoch": epoch, "l1": float(np.mean(losses)), "lr": optimiser.param_groups[0]["lr"]}
        append_history(run_dir / "fusion_history.csv", record)
        print(f"fusion epoch {epoch}/{args.pretrain_epochs}: L1={record['l1']:.6f}")
        if epoch % args.checkpoint_every == 0 or epoch == args.pretrain_epochs:
            save_state(
                checkpoint, model=model.state_dict(), optimizer=optimiser.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, options=vars(args),
            )
    return checkpoint


def meta_train_adaptor(args: argparse.Namespace, run_dir: Path, device: torch.device, fusion_path: Path) -> Path:
    checkpoint = run_dir / "adaptor_meta_last.pth"
    options = utal_options(args.scale, args.train_patch_size, args.dataset_root)
    fusion = build_fusion(options, device)
    fusion.load_state_dict(load_state(fusion_path, device)["model"])
    fusion.eval()
    adaptor = build_adaptor(device)
    optimiser = torch.optim.Adam(adaptor.parameters(), lr=args.meta_lr, weight_decay=1e-4)
    start_step = 0
    if args.resume and checkpoint.exists():
        state = load_state(checkpoint, device)
        adaptor.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"Resuming adaptor meta-training at task {start_step}.")

    loader = cave_loader(
        args.dataset_root, "train", args.train_patch_size, args.scale,
        shuffle=True, lazy=True, batch_size=1,
    )
    iterator = iter(loader)
    l1 = nn.L1Loss()
    srf = canonical_srf(device)
    for step in range(start_step + 1, args.meta_steps + 1):
        try:
            ground_truth, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            ground_truth, _, _ = next(iterator)
        ground_truth = ground_truth.to(device, non_blocking=True)
        lr_hsi, hr_msi, kernel, spectral, synthetic_spatial = random_synthetic_pair(
            ground_truth, args.scale, srf
        )
        with torch.no_grad():
            base = fusion(lr_hsi, hr_msi)

        # First-order MAML episode: self-supervised inner adaptation, followed
        # by a synthetic-GT outer objective.  All such tasks use training scenes.
        task_adaptor = copy.deepcopy(adaptor).to(device)
        inner = torch.optim.SGD(task_adaptor.parameters(), lr=args.inner_lr)
        task_output = task_adaptor(base, kernel, spectral)
        reconstruction = l1(synthetic_spatial(task_output), lr_hsi) + l1(
            torch.matmul(spectral[:, 0].transpose(1, 2), task_output.flatten(2)).reshape_as(hr_msi),
            hr_msi,
        )
        inner.zero_grad(set_to_none=True)
        reconstruction.backward()
        inner.step()
        outer = l1(task_adaptor(base, kernel, spectral), ground_truth)
        gradients = torch.autograd.grad(outer, task_adaptor.parameters())
        optimiser.zero_grad(set_to_none=True)
        for parameter, gradient in zip(adaptor.parameters(), gradients):
            parameter.grad = gradient.detach().clone()
        optimiser.step()
        if step % args.log_every == 0 or step == args.meta_steps:
            record = {"step": step, "inner_l1": float(reconstruction.detach().cpu()), "outer_l1": float(outer.detach().cpu())}
            append_history(run_dir / "meta_history.csv", record)
            print(f"meta task {step}/{args.meta_steps}: inner={record['inner_l1']:.6f} outer={record['outer_l1']:.6f}")
        if step % args.checkpoint_every == 0 or step == args.meta_steps:
            save_state(checkpoint, model=adaptor.state_dict(), optimizer=optimiser.state_dict(), step=step, options=vars(args))
    return checkpoint


def fusion_initialisation(
    fusion: nn.Module, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, options: SimpleNamespace
) -> torch.Tensor:
    options.h_size = list(hr_msi.shape[-2:])
    fusion.Re_generate_Mask(str(lr_hsi.device))
    with torch.no_grad():
        return fusion(lr_hsi, hr_msi)


def adapt_scene(
    adaptor: nn.Module, initial: torch.Tensor, lr_hsi: torch.Tensor, hr_msi: torch.Tensor,
    scale: int, srf: torch.Tensor, stages: int, inner_steps: int,
) -> tuple[torch.Tensor, dict]:
    """Released alternating test-time update order, using observations only."""
    spatial, spatial_conv = make_spatial_estimator(scale, initial.device)
    spectral = nn.Parameter(srf.clone())
    opt_spatial = torch.optim.Adam(spatial.parameters(), lr=1e-4, weight_decay=1e-4)
    opt_spectral = torch.optim.Adam([spectral], lr=1e-4, weight_decay=1e-5)
    opt_adaptor = torch.optim.Adam(adaptor.parameters(), lr=1e-3, weight_decay=1e-4)
    l1 = nn.L1Loss()
    output = initial.detach()
    latest_loss = float("nan")
    for _ in range(stages):
        for _ in range(inner_steps):
            opt_spatial.zero_grad(set_to_none=True)
            loss = l1(spatial(output.detach()), lr_hsi)
            loss.backward()
            opt_spatial.step()
        for _ in range(inner_steps):
            opt_spectral.zero_grad(set_to_none=True)
            loss = l1(torch.matmul(spectral, output.detach().flatten(2)).reshape_as(hr_msi), hr_msi)
            loss.backward()
            opt_spectral.step()
        kernel = spatial_conv.weight.detach()
        spectral_input = spectral.detach().transpose(0, 1).unsqueeze(0).unsqueeze(0)
        for _ in range(inner_steps):
            output = adaptor(initial, kernel, spectral_input)
            loss = l1(spatial(output), lr_hsi) + l1(
                torch.matmul(spectral, output.flatten(2)).reshape_as(hr_msi), hr_msi
            )
            opt_adaptor.zero_grad(set_to_none=True)
            opt_spatial.zero_grad(set_to_none=True)
            opt_spectral.zero_grad(set_to_none=True)
            loss.backward()
            opt_adaptor.step()
            opt_spatial.step()
            opt_spectral.step()
            latest_loss = float(loss.detach().cpu())
    return output.detach(), {"self_supervised_l1": latest_loss, "updates": stages * inner_steps}


def test_time_adaptation(
    args: argparse.Namespace, run_dir: Path, device: torch.device, fusion_path: Path, adaptor_path: Path
) -> None:
    options = utal_options(args.scale, args.train_patch_size, args.dataset_root)
    fusion = build_fusion(options, device)
    fusion.load_state_dict(load_state(fusion_path, device)["model"])
    fusion.eval()
    meta_state = load_state(adaptor_path, device)["model"]
    srf = canonical_srf(device)
    loader = cave_loader(args.dataset_root, "test", 512, args.scale, shuffle=False, lazy=True, batch_size=1)
    rows = []
    for sequence, (ground_truth, lr_hsi, hr_msi) in enumerate(loader):
        if args.max_test_scenes is not None and sequence >= args.max_test_scenes:
            break
        scene_id = TEST_IDS[sequence]
        scene_dir = run_dir / "test" / str(scene_id)
        result_path = scene_dir / "result.json"
        if args.resume and result_path.exists():
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            print(f"scene {scene_id}: already complete")
            continue
        scene_dir.mkdir(parents=True, exist_ok=True)
        lr_hsi = lr_hsi.to(device, non_blocking=True)
        hr_msi = hr_msi.to(device, non_blocking=True)
        adaptor = build_adaptor(device)
        adaptor.load_state_dict(meta_state)
        adaptor.train()
        started = time.perf_counter()
        initial = fusion_initialisation(fusion, lr_hsi, hr_msi, options)
        prediction, adaptation = adapt_scene(
            adaptor, initial, lr_hsi, hr_msi, args.scale, srf,
            args.adapt_stages, args.adapt_inner_steps,
        )
        # This is deliberately the first use of held-out HR-HSI in the scene.
        score = metrics(prediction.clamp(0, 1), ground_truth.to(device), args.scale)
        row = {
            "scene": scene_id,
            **score,
            **adaptation,
            "seconds": time.perf_counter() - started,
        }
        np.save(scene_dir / "prediction.npy", prediction[0].cpu().permute(1, 2, 0).numpy())
        write_json(result_path, row)
        rows.append(row)
        print(
            f"scene {scene_id}: PSNR={row['psnr']:.3f}, SAM={row['sam']:.3f}, "
            f"ERGAS={row['ergas']:.3f}, SSIM={row['ssim']:.4f}, {row['seconds']:.1f}s"
        )
    if rows:
        summary = {key: float(np.mean([row[key] for row in rows])) for key in ("psnr", "sam", "ergas", "ssim", "seconds")}
        summary.update({"method": "UTAL", "test_scenes": len(rows), "test_ids": [row["scene"] for row in rows]})
        write_json(run_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UTAL train/meta-train/test-time-adaptation protocol.")
    parser.add_argument("--output-root", type=Path, default=Path("Benchmarks/CAVE_sequential_60_40/comparisons/UTAL"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--phase", choices=("all", "train", "meta", "test"), default="all")
    parser.add_argument("--scale", type=int, default=32)
    parser.add_argument("--train-patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=150)
    parser.add_argument("--meta-steps", type=int, default=1800)
    parser.add_argument("--adapt-stages", type=int, default=40)
    parser.add_argument("--adapt-inner-steps", type=int, default=10)
    parser.add_argument("--fusion-lr", type=float, default=1e-4)
    parser.add_argument("--meta-lr", type=float, default=5e-5)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-test-scenes", type=int)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.scale != 32 or 512 % args.train_patch_size or args.train_patch_size % args.scale:
        raise ValueError("The CAVE x32 protocol needs a scale-aligned train patch size that divides 512.")
    if not torch.cuda.is_available():
        raise RuntimeError("UTAL's released implementation requires CUDA for test-time adaptation.")
    args.dataset_root = args.dataset_root.resolve()
    run_dir = args.output_root.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    write_json(run_dir / "run_config.json", {
        **vars(args), "output_root": str(run_dir), "dataset_root": str(args.dataset_root),
        "train_ids": list(TRAIN_IDS), "test_ids": list(TEST_IDS), "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "protocol": "train-only fusion/meta training; per-test-scene observation-only adaptation",
    })
    device = torch.device("cuda")
    fusion_path = run_dir / "fusion_last.pth"
    adaptor_path = run_dir / "adaptor_meta_last.pth"
    if args.phase in ("all", "train"):
        fusion_path = train_fusion(args, run_dir, device)
    if args.phase in ("all", "meta"):
        if not fusion_path.exists():
            raise FileNotFoundError("fusion_last.pth is required before meta-training")
        adaptor_path = meta_train_adaptor(args, run_dir, device, fusion_path)
    if args.phase in ("all", "test"):
        if not fusion_path.exists() or not adaptor_path.exists():
            raise FileNotFoundError("fusion_last.pth and adaptor_meta_last.pth are required before test adaptation")
        test_time_adaptation(args, run_dir, device, fusion_path, adaptor_path)


if __name__ == "__main__":
    main()
