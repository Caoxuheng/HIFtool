"""HIFTool adapter for the official EMR-Diff network."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .arch.BAFUnet import BAFUNet
from .config import make_options
from .official_diffusion import EMRDIFF, Edge


class EMRDiffNet(nn.Module):
    """EMR-Diff with the standard ``model(lr_hsi, hr_msi)`` HIFTool API.

    The architecture released by the authors is fixed to 31 HSI bands and
    three MSI channels. Training must call :meth:`training_loss`; ``forward``
    is the stochastic five-step reverse process used for inference.
    """

    learning_rate = 1e-4
    batch_size = 1
    requires_specialized_loss = True
    source_url = "https://github.com/luocz55/EMR-Diff"
    source_commit = "cc851ec7b597bbe259c1cdc413a13e19a386d228"

    def __init__(self, options=None, **overrides) -> None:
        super().__init__()
        opt = options or make_options()
        values = vars(opt).copy()
        values.update(overrides)
        self.scale = int(values.get("sf", values.get("scale", 32)))
        self.hsi_bands = int(values.get("hsi_channel", values.get("hsi_bands", 31)))
        self.msi_bands = int(values.get("msi_channel", values.get("msi_bands", 3)))
        self.patch_size = int(values.get("patch_size", 256))
        self.diffusion_steps = int(values.get("diffusion_steps", 5))
        if (self.hsi_bands, self.msi_bands) != (31, 3):
            raise ValueError("The released EMR-Diff architecture requires 31 HSI and 3 MSI channels")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")

        total_channels = self.hsi_bands + self.msi_bands
        self.net = BAFUNet(
            image_size=self.patch_size,
            in_channels=total_channels,
            model_channels=total_channels,
            out_channels=total_channels,
            channel_mult=[1, 1, 1, 1, 1],
            num_res_blocks=[1, 1, 1, 1, 1],
            dims=2,
            lqrgb_channels=total_channels,
        )
        self.diffusion = EMRDIFF({
            "params": {
                "sf": self.scale,
                "schedule_name": "exponential",
                "schedule_kwargs": {"power": 0.3},
                "etas_end": 0.999,
                "steps": self.diffusion_steps,
                "min_noise_level": 0.001,
                "kappa": 2.0,
                "band_dim": self.hsi_bands,
                "normalize_input": False,
                "latent_flag": None,
            }
        })
        self.edge = Edge()

    def _validate(self, lrhsi: Tensor, hrmsi: Tensor) -> None:
        if lrhsi.ndim != 4 or hrmsi.ndim != 4:
            raise ValueError("EMR-Diff expects BCHW LR-HSI and HR-MSI tensors")
        if lrhsi.shape[0] != hrmsi.shape[0]:
            raise ValueError("LR-HSI and HR-MSI batch sizes must match")
        if lrhsi.shape[1] != self.hsi_bands or hrmsi.shape[1] != self.msi_bands:
            raise ValueError(
                f"Channel mismatch: got {lrhsi.shape[1]}/{hrmsi.shape[1]}, "
                f"expected {self.hsi_bands}/{self.msi_bands}"
            )
        expected = (lrhsi.shape[-2] * self.scale, lrhsi.shape[-1] * self.scale)
        if hrmsi.shape[-2:] != expected:
            raise ValueError(
                f"Scale mismatch: LR={tuple(lrhsi.shape[-2:])}, "
                f"HR={tuple(hrmsi.shape[-2:])}, scale={self.scale}"
            )
        if hrmsi.shape[-2] % 16 or hrmsi.shape[-1] % 16:
            raise ValueError("EMR-Diff HR tile dimensions must be divisible by 16")

    def _condition(self, lrhsi: Tensor, hrmsi: Tensor) -> Tuple[Tensor, Tensor]:
        self._validate(lrhsi, hrmsi)
        upsampled = F.interpolate(
            lrhsi, size=hrmsi.shape[-2:], mode="bicubic", align_corners=False,
        )
        return upsampled, torch.cat((upsampled, hrmsi), dim=1)

    def training_loss(
        self, target: Tensor, lrhsi: Tensor, hrmsi: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        upsampled, condition = self._condition(lrhsi, hrmsi)
        if target.shape != upsampled.shape:
            raise ValueError(
                f"Target shape {tuple(target.shape)} must equal HR-HSI shape {tuple(upsampled.shape)}"
            )
        # The official 34-channel clean target is [GT-HSI, GT bands 0:3].
        # Observed MSI belongs only to the condition because HIFTool SRFs need
        # not be literal selection of the first three hyperspectral bands.
        clean = torch.cat((target, target[:, : self.msi_bands]), dim=1)
        timestep = torch.randint(
            0, self.diffusion_steps, (target.shape[0],), device=target.device,
        )
        residual_calculator = self.diffusion.residual_calculator.to(
            device=target.device, dtype=target.dtype,
        )
        self.diffusion.residual_calculator = residual_calculator
        noisy = self.diffusion.forward_addnoise(
            x_start=clean,
            y=condition,
            t=timestep,
            noise=torch.randn_like(condition),
            rgb_hr=hrmsi,
        )
        residual, pyramid = self.net(noisy, hrmsi, upsampled, timestep)
        reconstruction = F.l1_loss(residual + condition, clean)
        multiscale = clean.new_zeros(())
        used = 0
        for index in (2, 4, 6):
            if index >= len(pyramid):
                continue
            output = pyramid[index]
            if output.shape[1] != clean.shape[1]:
                continue
            factor_h = clean.shape[-2] // output.shape[-2]
            factor_w = clean.shape[-1] // output.shape[-1]
            if factor_h != factor_w or factor_h < 1:
                raise ValueError(f"Unexpected EMR-Diff pyramid shape: {tuple(output.shape)}")
            target_scale = clean[:, :, ::factor_h, ::factor_h]
            target_scale = target_scale[:, :, : output.shape[-2], : output.shape[-1]]
            rgb_scale = hrmsi[:, :, ::factor_h, ::factor_h]
            rgb_scale = rgb_scale[:, :, : output.shape[-2], : output.shape[-1]]
            hsi_scale = F.interpolate(
                lrhsi, size=output.shape[-2:], mode="bicubic", align_corners=False,
            )
            condition_scale = torch.cat((hsi_scale, rgb_scale), dim=1)
            multiscale = multiscale + F.l1_loss(output + condition_scale, target_scale)
            used += 1
        loss = reconstruction + multiscale
        return loss, {
            "reconstruction": float(reconstruction.detach()),
            "multiscale": float(multiscale.detach()),
            "multiscale_terms": used,
        }

    @torch.no_grad()
    def predict(self, lrhsi: Tensor, hrmsi: Tensor) -> Tensor:
        upsampled, condition = self._condition(lrhsi, hrmsi)
        edge_map = self.edge(hrmsi)
        latent = self.diffusion.prior_sample(
            condition, torch.randn_like(condition), edge_map=edge_map,
        )
        for value in range(self.diffusion_steps - 1, -1, -1):
            timestep = torch.full(
                (latent.shape[0],), value, device=latent.device, dtype=torch.long,
            )
            residual, _ = self.net(latent, hrmsi, upsampled, timestep)
            clean = residual + condition
            latent = self.diffusion.inverse_denoise(
                x_start=clean,
                x_t=latent,
                t=timestep,
                noise=torch.randn_like(clean),
                edge_map=edge_map,
            )
        return latent[:, : self.hsi_bands].clamp(0.0, 1.0)

    def forward(self, lrhsi: Tensor, hrmsi: Tensor) -> Tensor:
        return self.predict(lrhsi, hrmsi)


EMRDiff = EMRDiffNet

