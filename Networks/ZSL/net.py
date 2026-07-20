"""HIFTool integration of ZSL with automatic high-scale cascading.

ZSL itself has a fixed x4 transposed-convolution stage.  For scale factors
above 16, HIFTool first performs the fixed bicubic bridge ``sf / 16`` and then
runs two unchanged, independently-trained x4 ZSL stages.  At x32 this is the
validated ``bicubic x2 -> ZSL x4 -> ZSL x4`` protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .Model.cnn import ZSLStageCNN


def _gaussian_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinate = torch.arange(7, device=device, dtype=dtype) - 3
    vector = torch.exp(-0.5 * (coordinate / 3.0).square())
    vector = vector / vector.sum()
    return vector[:, None] * vector[None, :]


def _spatial_downsample(value: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = value.shape[1]
    weight = kernel[None, None].repeat(channels, 1, 1, 1)
    padded = F.pad(value, (3, 3, 3, 3), mode="reflect")
    output = F.conv2d(padded, weight, stride=4, groups=channels)
    return output[:, :, : value.shape[-2] // 4, : value.shape[-1] // 4]


class _ZSLStage(nn.Module):
    scale = 4

    def __init__(self, observed_hsi: torch.Tensor, srf: torch.Tensor, rank: int, batch_size: int, patch_size: int):
        super().__init__()
        if min(observed_hsi.shape[-2:]) < patch_size:
            raise ValueError(
                f"ZSL requires an observed HSI of at least {patch_size}x{patch_size}; got "
                f"{tuple(observed_hsi.shape[-2:])}. Add a bicubic bridge before this x4 stage."
            )
        rank = min(rank, observed_hsi.shape[1], observed_hsi.shape[-2] * observed_hsi.shape[-1])
        spectra = observed_hsi[0].reshape(observed_hsi.shape[1], -1)
        self.register_buffer("basis", torch.linalg.svd(spectra, full_matrices=False).U[:, :rank])
        self.register_buffer("srf", srf)
        self.register_buffer("kernel", _gaussian_kernel(observed_hsi.device, observed_hsi.dtype))
        self.net = ZSLStageCNN(rank, srf.shape[0])
        self.batch_size = batch_size
        self.patch_size = patch_size

    def abundance(self, hsi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("cr,bchw->brhw", self.basis, hsi)

    def reconstruct(self, abundance: torch.Tensor) -> torch.Tensor:
        return torch.einsum("cr,brhw->bchw", self.basis, abundance)

    def spectral(self, hsi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mc,bchw->bmhw", self.srf, hsi)

    def loss(self, observed_hsi: torch.Tensor) -> torch.Tensor:
        pseudo_msi = self.spectral(observed_hsi)
        abundance_low = _spatial_downsample(self.abundance(observed_hsi), self.kernel)
        height, width = observed_hsi.shape[-2:]
        low_size = self.patch_size // self.scale
        targets, msis, lows = [], [], []
        for _ in range(self.batch_size):
            top = int(torch.randint(0, height - self.patch_size + 1, (), device=observed_hsi.device))
            left = int(torch.randint(0, width - self.patch_size + 1, (), device=observed_hsi.device))
            targets.append(observed_hsi[:, :, top:top + self.patch_size, left:left + self.patch_size])
            msis.append(pseudo_msi[:, :, top:top + self.patch_size, left:left + self.patch_size])
            lows.append(abundance_low[:, :, top // 4:top // 4 + low_size, left // 4:left // 4 + low_size])
        target = torch.cat(targets, 0)
        msi = torch.cat(msis, 0)
        prediction = self.reconstruct(self.net(torch.cat(lows, 0), msi))
        image = F.l1_loss(prediction, target)
        spectral = F.l1_loss(self.spectral(prediction), msi)
        down_prediction = _spatial_downsample(prediction, self.kernel)
        down_target = _spatial_downsample(target, self.kernel)
        spatial = F.l1_loss(down_prediction[:, :, 1:-1, 1:-1], down_target[:, :, 1:-1, 1:-1])
        return image + spectral + 0.1 * spatial

    @torch.no_grad()
    def predict(self, hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(self.net(self.abundance(hsi), msi)).clamp(0.0, 1.0)


class ZSL(nn.Module):
    """Per-image zero-shot HSI sharpening callable through ``model_generator``."""

    def __init__(self, args, device: str = "cuda"):
        super().__init__()
        self.args = args
        self.device = torch.device(device)

    def _srf(self, dtype: torch.dtype) -> torch.Tensor:
        array = np.load(Path(self.args.srfpath))
        if array.shape == (self.args.hsi_channel, self.args.msi_channel):
            array = array.T
        if array.shape != (self.args.msi_channel, self.args.hsi_channel):
            raise ValueError(f"SRF at {self.args.srfpath} has shape {array.shape}, expected 3x31 or 31x3.")
        return torch.as_tensor(array, device=self.device, dtype=dtype)

    def _train_stage(self, observed_hsi: torch.Tensor, condition_msi: torch.Tensor, steps: int, srf: torch.Tensor) -> torch.Tensor:
        stage = _ZSLStage(observed_hsi, srf, self.args.rank, self.args.zsl_batch_size, self.args.zsl_training_patch).to(self.device)
        optimizer = torch.optim.Adam(stage.parameters(), lr=self.args.zsl_lr, betas=(0.9, 0.999))
        stage.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = stage.loss(observed_hsi)
            if not torch.isfinite(loss):
                raise FloatingPointError("ZSL produced a non-finite self-supervised loss")
            loss.backward()
            optimizer.step()
        stage.eval()
        return stage.predict(observed_hsi, condition_msi)

    def forward(self, lrhsi: torch.Tensor, hrmsi: torch.Tensor) -> torch.Tensor:
        if lrhsi.shape[0] != 1 or hrmsi.shape[0] != 1:
            raise ValueError("ZSL performs independent per-image adaptation and currently expects batch size 1.")
        if self.args.sf <= 0:
            raise ValueError("--sf must be positive")
        lrhsi, hrmsi = lrhsi.to(self.device), hrmsi.to(self.device)
        if tuple(value * self.args.sf for value in lrhsi.shape[-2:]) != tuple(hrmsi.shape[-2:]):
            raise ValueError("LR-HSI and HR-MSI dimensions do not match --sf.")
        srf = self._srf(lrhsi.dtype)
        with torch.enable_grad():
            if self.args.sf == 4:
                return self._train_stage(lrhsi, hrmsi, self.args.zsl_steps, srf)
            if self.args.sf <= 16:
                raise ValueError("The integrated ZSL path is direct x4 only up to x16; use x4 or a scale >16 cascade.")
            if self.args.zsl_steps < 2:
                raise ValueError("The ZSL cascade needs at least two total updates.")
            # For x32: 16x16 -> fixed bicubic x2 -> 32x32 -> x4 -> x4.
            bridge = self.args.sf / 16.0
            bridged = F.interpolate(lrhsi, scale_factor=bridge, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
            first_size: Tuple[int, int] = tuple(value * 4 for value in bridged.shape[-2:])
            first_msi = F.interpolate(hrmsi, size=first_size, mode="bicubic", align_corners=False, antialias=True)
            first_steps = self.args.zsl_steps // 2
            first_output = self._train_stage(bridged, first_msi, first_steps, srf)
            return self._train_stage(first_output, hrmsi, self.args.zsl_steps - first_steps, srf)
