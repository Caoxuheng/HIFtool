"""uHNTC-guided blind fusion for a supervised CaFormer prior.

The degradation networks are estimated from the LR-HSI/HR-MSI observation
pair only. They then provide spatial and spectral observation losses for
per-image adaptation of a pretrained CaFormer. Ground truth is never consumed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .inference import forward_tiled


def default_spectral_spans(hsi_channels: int, msi_channels: int) -> list[list[int]]:
    if (hsi_channels, msi_channels) == (31, 3):
        # Nikon-D700 support used by the current HIFTool CAVE/Harvard runs.
        return [list(range(18, 31)), list(range(10, 23)), list(range(0, 12))]
    width = max(2, int(round(0.4 * hsi_channels)))
    centers = torch.linspace(0, hsi_channels - 1, msi_channels)
    spans = []
    for center in centers.tolist():
        start = max(0, min(hsi_channels - width, int(round(center - width / 2))))
        spans.append(list(range(start, start + width)))
    return spans


@dataclass(frozen=True)
class UHNTCConfig:
    scale: int
    hsi_channels: int = 31
    msi_channels: int = 3
    spectral_spans: tuple[tuple[int, ...], ...] | None = None
    spe_init_steps: int = 500
    spa_init_steps: int = 500
    couple_init_steps: int = 500
    adaptation_steps: int = 1000
    adaptation_lr: float = 3e-5
    tile_size: int | None = None
    tile_overlap: int = 0

    def spans(self) -> list[list[int]]:
        if self.spectral_spans is None:
            return default_spectral_spans(self.hsi_channels, self.msi_channels)
        return [list(span) for span in self.spectral_spans]


@dataclass
class BlindResult:
    reconstruction: torch.Tensor
    baseline: torch.Tensor
    spatial_degrader: torch.nn.Module
    spectral_degrader: torch.nn.Module
    diagnostics: dict[str, float | int | str]


def _require_uhntc():
    try:
        from Networks.FeafusFormer.Model.spa_down import SpaDown
        from Networks.FeafusFormer.Model.spe_down import SpeDown
    except ImportError as exc:
        raise ImportError(
            "Blind CaFormer requires HIFTool's Networks/FeafusFormer "
            "(uHNTC SpaDNet and SpeDNet)."
        ) from exc
    return SpaDown, SpeDown


def _psnr_loss(reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    mse = (reference - prediction).square().mean(dim=(-1, -2), keepdim=True)
    return -(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean()


def _initialize_spectral(spe, hr_msi, lr_hsi, scale: int, steps: int) -> None:
    blurred_msi = F.avg_pool2d(
        F.pad(hr_msi, [scale] * 4, mode="reflect"),
        kernel_size=2 * scale + 1,
        stride=scale,
    )
    blurred_hsi = F.avg_pool2d(
        F.pad(lr_hsi, [1] * 4, mode="reflect"), kernel_size=3, stride=1
    )
    optimizer = torch.optim.Adam(spe.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.9)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.l1_loss(spe(blurred_hsi), blurred_msi)
        loss.backward()
        optimizer.step()
        scheduler.step()


def _initialize_spatial(spa, hr_msi, lr_msi, steps: int) -> None:
    optimizer = torch.optim.Adam(spa.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
    target = lr_msi.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _psnr_loss(target, spa(hr_msi))
        loss.backward()
        optimizer.step()
        scheduler.step()


def _low_rank_factors(lr_hsi: torch.Tensor, rank: int = 3):
    _, channels, height, width = lr_hsi.shape
    u, singular, vh = torch.linalg.svd(
        lr_hsi.flatten(2)[0], full_matrices=False
    )
    phi = u[:, :rank] @ torch.diag(singular[:rank])
    coefficients = vh[:rank]
    return (
        phi.view(1, channels, rank, 1),
        coefficients.view(1, rank, height, width),
    )


def _couple_initialize(spa, spe, hr_msi, lr_hsi, steps: int) -> None:
    rank = min(3, lr_hsi.shape[1], lr_hsi.shape[-2] * lr_hsi.shape[-1])
    phi, coefficients = _low_rank_factors(lr_hsi, rank)
    spa_optimizer = torch.optim.AdamW(spa.parameters(), lr=5e-3, weight_decay=1e-4)
    spe_optimizer = torch.optim.AdamW(spe.parameters(), lr=5e-3, weight_decay=1e-4)
    spa_scheduler = torch.optim.lr_scheduler.StepLR(spa_optimizer, 50, 0.8)
    spe_scheduler = torch.optim.lr_scheduler.StepLR(spe_optimizer, 50, 0.8)
    height, width = lr_hsi.shape[-2:]
    for _ in range(steps):
        spa_optimizer.zero_grad(set_to_none=True)
        spe_optimizer.zero_grad(set_to_none=True)
        low_msi = spa(hr_msi)
        projected = torch.bmm(
            spe(phi)[:, :, :, 0], coefficients.view(1, rank, -1)
        ).view(1, hr_msi.shape[1], height, width)
        loss = F.l1_loss(low_msi, projected) + F.l1_loss(low_msi, spe(lr_hsi))
        loss.backward()
        spa_optimizer.step()
        spe_optimizer.step()
        spa_scheduler.step()
        spe_scheduler.step()


def _adaptation_tiles(lr_hsi, hr_msi, scale: int, tile_size: int | None):
    if tile_size is None or tile_size <= 0:
        return [(lr_hsi, hr_msi)]
    height, width = hr_msi.shape[-2:]
    if max(height, width) <= tile_size:
        return [(lr_hsi, hr_msi)]
    if tile_size % scale:
        raise ValueError("blind tile_size must be divisible by scale")
    expected = (lr_hsi.shape[-2] * scale, lr_hsi.shape[-1] * scale)
    if (height, width) != expected:
        raise ValueError(
            f"blind tiled adaptation requires HR size {expected}, got {(height, width)}"
        )
    tile_h, tile_w = min(tile_size, height), min(tile_size, width)
    y_starts = list(range(0, height, tile_h))
    x_starts = list(range(0, width, tile_w))
    if y_starts[-1] + tile_h > height:
        y_starts[-1] = height - tile_h
    if x_starts[-1] + tile_w > width:
        x_starts[-1] = width - tile_w
    tiles = []
    for top in y_starts:
        for left in x_starts:
            bottom, right = top + tile_h, left + tile_w
            tiles.append(
                (
                    lr_hsi[
                        :,
                        :,
                        top // scale : bottom // scale,
                        left // scale : right // scale,
                    ],
                    hr_msi[:, :, top:bottom, left:right],
                )
            )
    return tiles


class BlindCaFormer:
    """Per-observation blind adapter preserving the supplied base checkpoint."""

    def __init__(self, model: torch.nn.Module, config: UHNTCConfig):
        self.base_model = model
        self.config = config

    def estimate_degradation(self, lr_hsi, hr_msi):
        if lr_hsi.shape[0] != 1 or hr_msi.shape[0] != 1:
            raise ValueError("uHNTC blind degradation estimation requires batch size 1")
        SpaDown, SpeDown = _require_uhntc()
        spa = SpaDown(sf=self.config.scale, predefine=None, iscal=True).to(lr_hsi.device)
        spe = SpeDown(
            span=self.config.spans(), predefine=None, iscal=True
        ).to(lr_hsi.device)
        _initialize_spectral(
            spe, hr_msi, lr_hsi, self.config.scale, self.config.spe_init_steps
        )
        _initialize_spatial(
            spa, hr_msi, spe(lr_hsi), self.config.spa_init_steps
        )
        _couple_initialize(
            spa, spe, hr_msi, lr_hsi, self.config.couple_init_steps
        )
        for module in (spa, spe):
            module.eval()
            module.requires_grad_(False)
        return spa, spe

    def __call__(self, lr_hsi, hr_msi) -> BlindResult:
        if lr_hsi.shape[1] != self.config.hsi_channels:
            raise ValueError("LR-HSI channel count does not match UHNTCConfig")
        if hr_msi.shape[1] != self.config.msi_channels:
            raise ValueError("HR-MSI channel count does not match UHNTCConfig")
        spa, spe = self.estimate_degradation(lr_hsi, hr_msi)
        model = copy.deepcopy(self.base_model).to(lr_hsi.device)
        model.eval()
        with torch.no_grad():
            baseline = forward_tiled(
                model,
                lr_hsi,
                hr_msi,
                self.config.scale,
                self.config.tile_size,
                self.config.tile_overlap,
            ).detach()
            cross_l1 = float(F.l1_loss(spa(hr_msi), spe(lr_hsi)))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.adaptation_lr)
        final_loss = final_spatial = final_spectral = float("nan")
        tiles = _adaptation_tiles(
            lr_hsi, hr_msi, self.config.scale, self.config.tile_size
        )
        for step in range(self.config.adaptation_steps):
            lr_patch, ms_patch = tiles[step % len(tiles)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lr_patch, ms_patch)
            spatial = F.l1_loss(spa(prediction), lr_patch)
            spectral = F.l1_loss(spe(prediction), ms_patch)
            loss = spatial + spectral
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite blind CaFormer loss")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            final_spatial = float(spatial.detach())
            final_spectral = float(spectral.detach())

        model.eval()
        with torch.no_grad():
            reconstruction = forward_tiled(
                model,
                lr_hsi,
                hr_msi,
                self.config.scale,
                self.config.tile_size,
                self.config.tile_overlap,
            )
        diagnostics = {
            "mode": "blind_uHNTC",
            "scale": self.config.scale,
            "cross_l1": cross_l1,
            "adaptation_steps": self.config.adaptation_steps,
            "adaptation_lr": self.config.adaptation_lr,
            "adaptation_tiles": len(tiles),
            "self_loss": final_loss,
            "spatial_loss": final_spatial,
            "spectral_loss": final_spectral,
            "uses_ground_truth": 0,
            "uses_true_degradation": 0,
        }
        return BlindResult(reconstruction, baseline, spa, spe, diagnostics)
