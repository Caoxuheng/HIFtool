"""Executable, scale-parameterized PSRF-DiffNet implementation for HIFTool.

The public release hard-codes x4, a 160-pixel patch, four MSI channels, and
contains two neighbourhood tensor shape mismatches.  This module keeps the
paper's three method components (coarse registration, attention matching fine
registration, and spatial-spectral coupled fusion) and its progressive
diffusion refinement, while making only the tensor/scale corrections required
for different HSI/MSI sensor dimensions and fusion ratios.

No HIFTool Sylvester, DNC, operator estimator, or physical solver is used.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.silu(value + self.body(value), inplace=True)


class CoarseRegistration(nn.Module):
    """Global LR cross-attention followed by the released affine warp."""

    def __init__(self, hsi_bands: int, msi_bands: int, width: int, lr_grid: int):
        super().__init__()
        self.lr_grid = int(lr_grid)
        self.hsi_embed = nn.Sequential(
            nn.Conv2d(hsi_bands, width, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.msi_embed = nn.Sequential(
            nn.Conv2d(msi_bands, width, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.hsi_norm = nn.LayerNorm(width)
        self.msi_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, num_heads=4, batch_first=True, bias=False,
        )
        tokens = self.lr_grid * self.lr_grid
        self.localization = nn.Sequential(
            nn.Linear(tokens * width, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )
        nn.init.zeros_(self.localization[-1].weight)
        nn.init.zeros_(self.localization[-1].bias)

    def forward(self, upsampled_hsi: Tensor, lrhsi: Tensor, hrmsi: Tensor) -> Tuple[Tensor, Tensor]:
        hsi_lr = F.interpolate(
            lrhsi, size=(self.lr_grid, self.lr_grid), mode="bicubic", align_corners=False,
        )
        msi_lr = F.interpolate(
            hrmsi, size=(self.lr_grid, self.lr_grid), mode="bicubic", align_corners=False,
        )
        hsi_tokens = self.hsi_embed(hsi_lr).flatten(2).transpose(1, 2)
        msi_tokens = self.msi_embed(msi_lr).flatten(2).transpose(1, 2)
        attended, _ = self.cross_attention(
            self.hsi_norm(hsi_tokens), self.msi_norm(msi_tokens), self.msi_norm(msi_tokens),
            need_weights=False,
        )
        delta = 0.10 * torch.tanh(self.localization(attended.flatten(1))).view(-1, 2, 3)
        identity = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=delta.device,
            dtype=delta.dtype,
        ).unsqueeze(0)
        theta = identity + delta
        grid = F.affine_grid(theta, upsampled_hsi.shape, align_corners=False)
        registered = F.grid_sample(
            upsampled_hsi, grid, mode="bilinear", padding_mode="border", align_corners=False,
        )
        return registered, theta


class AttentionFineRegistration(nn.Module):
    """Differentiable form of the released 3x3 attention-enhanced matching."""

    def __init__(self, hsi_bands: int, msi_bands: int, width: int):
        super().__init__()
        self.hsi_embed = nn.Sequential(
            nn.Conv2d(hsi_bands, width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        self.msi_embed = nn.Sequential(
            nn.Conv2d(msi_bands, width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
        )
        offsets = []
        for dy in (-1.0, 0.0, 1.0):
            for dx in (-1.0, 0.0, 1.0):
                offsets.append((dx, dy))
        self.register_buffer("offsets", torch.tensor(offsets).view(1, 9, 2, 1, 1), persistent=False)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(8.0)))

    def forward(self, hsi: Tensor, msi: Tensor) -> Tuple[Tensor, Tensor]:
        query = F.normalize(self.hsi_embed(hsi), dim=1, eps=1e-6)
        key = F.normalize(self.msi_embed(msi), dim=1, eps=1e-6)
        batch, channels, height, width = key.shape
        neighbours = F.unfold(key, 3, padding=1).view(batch, channels, 9, height, width)
        scores = (query.unsqueeze(2) * neighbours).sum(dim=1)
        temperature = self.log_temperature.exp().clamp(1.0, 32.0)
        weights = torch.softmax(scores * temperature, dim=1)
        flow_pixels = (weights.unsqueeze(2) * self.offsets.to(weights)).sum(dim=1)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=hsi.device, dtype=hsi.dtype),
            torch.linspace(-1.0, 1.0, width, device=hsi.device, dtype=hsi.dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        flow = flow_pixels.permute(0, 2, 3, 1)
        flow_x = 2.0 * flow[..., 0] / max(width - 1, 1)
        flow_y = 2.0 * flow[..., 1] / max(height - 1, 1)
        grid = base + torch.stack((flow_x, flow_y), dim=-1)
        registered = F.grid_sample(
            hsi, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return registered, flow_pixels


class SpatialStructure(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        ) / 8.0
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def forward(self, msi: Tensor) -> Tensor:
        intensity = msi.mean(dim=1, keepdim=True)
        gx = F.conv2d(intensity, self.sobel_x.to(intensity), padding=1)
        gy = F.conv2d(intensity, self.sobel_y.to(intensity), padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        normalizer = magnitude.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return torch.sigmoid(magnitude / normalizer)


class SpatialSpectralCoupledFusion(nn.Module):
    """S2CF: structure-guided MSI features plus channel-gated HSI features."""

    def __init__(self, hsi_bands: int, msi_bands: int, width: int, blocks: int):
        super().__init__()
        self.hsi_embed = nn.Conv2d(hsi_bands, width, 3, padding=1)
        self.state_embed = nn.Conv2d(hsi_bands, width, 3, padding=1)
        self.msi_embed = nn.Conv2d(msi_bands, width, 3, padding=1)
        hidden = max(width // 4, 8)
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, width, 1),
            nn.Sigmoid(),
        )
        self.structure = SpatialStructure()
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.SiLU(inplace=True),
            *[ResidualBlock(width) for _ in range(blocks)],
            nn.Conv2d(width, hsi_bands, 3, padding=1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, registered_hsi: Tensor, state: Tensor, msi: Tensor) -> Tensor:
        hsi_features = self.hsi_embed(registered_hsi)
        hsi_features = hsi_features * self.spectral_gate(hsi_features)
        state_features = self.state_embed(state)
        msi_features = self.msi_embed(msi) * (1.0 + self.structure(msi))
        residual = self.fuse(torch.cat((hsi_features, state_features, msi_features), dim=1))
        return registered_hsi + residual


class FineRegistrationFusionBridge(nn.Module):
    def __init__(self, hsi_bands: int, msi_bands: int, width: int, blocks: int):
        super().__init__()
        self.fine_registration = AttentionFineRegistration(hsi_bands, msi_bands, width)
        self.fusion = SpatialSpectralCoupledFusion(hsi_bands, msi_bands, width, blocks)

    def forward(self, registered_hsi: Tensor, state: Tensor, msi: Tensor) -> Tuple[Tensor, Tensor]:
        fine_hsi, flow = self.fine_registration(registered_hsi, msi)
        return self.fusion(fine_hsi, state, msi), flow


class PSRFDiffNet(nn.Module):
    """Progressive synergistic registration/fusion diffusion network."""

    def __init__(
        self,
        scale: int = 32,
        patch_size: int = 128,
        hsi_bands: int = 31,
        msi_bands: int = 3,
        width: int = 64,
        blocks: int = 6,
        diffusion_steps: int = 8,
        linear_start: float = 1e-4,
        linear_end: float = 2e-3,
    ):
        super().__init__()
        if patch_size % scale:
            raise ValueError("patch_size must be divisible by scale")
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        self.scale = int(scale)
        self.patch_size = int(patch_size)
        self.hsi_bands = int(hsi_bands)
        self.msi_bands = int(msi_bands)
        self.diffusion_steps = int(diffusion_steps)
        self.coarse_registration = CoarseRegistration(
            hsi_bands, msi_bands, width, patch_size // scale,
        )
        self.frfb = FineRegistrationFusionBridge(hsi_bands, msi_bands, width, blocks)

        betas = torch.linspace(linear_start, linear_end, diffusion_steps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("alpha_bar_prev", alpha_bar_prev.float())
        self.register_buffer("posterior_variance", variance.clamp_min(1e-20).float())
        self.register_buffer(
            "posterior_coef1",
            (betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)).float(),
        )
        self.register_buffer(
            "posterior_coef2",
            ((1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar)).float(),
        )

    def observation_prior(self, lrhsi: Tensor, hrmsi: Tensor) -> Tuple[Tensor, Tensor]:
        upsampled = F.interpolate(
            lrhsi, size=hrmsi.shape[-2:], mode="bicubic", align_corners=False,
        )
        return self.coarse_registration(upsampled, lrhsi, hrmsi)

    def q_sample(self, clean: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        alpha = self.alpha_bar[timestep].view(-1, 1, 1, 1).to(clean)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    def training_loss(self, target: Tensor, lrhsi: Tensor, hrmsi: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        prior, theta = self.observation_prior(lrhsi, hrmsi)
        timestep = torch.randint(
            1, self.diffusion_steps, (target.shape[0],), device=target.device,
        )
        state = self.q_sample(target, timestep, torch.randn_like(target))
        prediction, flow = self.frfb(prior, state, hrmsi)
        reconstruction = F.l1_loss(prediction, target)
        return reconstruction, {
            "reconstruction": float(reconstruction.detach().item()),
            "mean_abs_fine_flow": float(flow.detach().abs().mean().item()),
            "mean_abs_affine_delta": float(
                (theta.detach() - theta.new_tensor([[1, 0, 0], [0, 1, 0]])).abs().mean().item()
            ),
        }

    @torch.no_grad()
    def sample(self, lrhsi: Tensor, hrmsi: Tensor, stochastic: bool = False) -> Tensor:
        prior, _ = self.observation_prior(lrhsi, hrmsi)
        state = torch.randn_like(prior)
        for step in range(self.diffusion_steps - 1, -1, -1):
            clean, _ = self.frfb(prior, state, hrmsi)
            if step == 0:
                state = clean
                continue
            state = self.posterior_coef1[step].to(clean) * clean + self.posterior_coef2[step].to(clean) * state
            if stochastic:
                state = state + self.posterior_variance[step].to(clean).sqrt() * torch.randn_like(state)
        return state.clamp(0.0, 1.0)

    def forward(self, lrhsi: Tensor, hrmsi: Tensor) -> Tensor:
        return self.sample(lrhsi, hrmsi)


def random_unregistered_lrhsi(lrhsi: Tensor, probability: float = 0.75) -> Tensor:
    """Author-style random affine/nonregistration augmentation in LR space."""
    if probability <= 0.0 or torch.rand((), device=lrhsi.device) > probability:
        return lrhsi
    batch = lrhsi.shape[0]
    angle = (torch.rand(batch, device=lrhsi.device) - 0.5) * (4.0 * math.pi / 180.0)
    scale = 1.0 + (torch.rand(batch, device=lrhsi.device) - 0.5) * 0.04
    tx = (torch.rand(batch, device=lrhsi.device) - 0.5) * 0.30
    ty = (torch.rand(batch, device=lrhsi.device) - 0.5) * 0.30
    cosine = torch.cos(angle) * scale
    sine = torch.sin(angle) * scale
    theta = torch.zeros(batch, 2, 3, device=lrhsi.device, dtype=lrhsi.dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, lrhsi.shape, align_corners=False)
    return F.grid_sample(
        lrhsi, grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
