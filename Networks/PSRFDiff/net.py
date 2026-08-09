"""HIFTool wrapper around the scale-corrected PSRF-DiffNet core."""

from __future__ import annotations

from torch import Tensor

from .config import make_options
from .core import PSRFDiffNet as PSRFDiffCore
from .core import random_unregistered_lrhsi


class PSRFDiffNet(PSRFDiffCore):
    """Scale-corrected PSRF-DiffNet with the HIFTool registry interface."""

    learning_rate = 1e-4
    batch_size = 8
    clip_grad_norm = 1.0
    requires_specialized_loss = True
    source_url = "https://github.com/Jiahuiqu/PSRF-DiffNet"
    source_commit = "d4e88fa3fe2f13eb5a67e53ead99f6370713ab9b"

    def __init__(self, options=None, **overrides) -> None:
        opt = options or make_options()
        values = vars(opt).copy()
        values.update(overrides)
        self.scale = int(values.get("sf", values.get("scale", 32)))
        self.patch_size = int(values.get("patch_size", 128))
        self.hsi_bands = int(values.get("hsi_channel", values.get("hsi_bands", 31)))
        self.msi_bands = int(values.get("msi_channel", values.get("msi_bands", 3)))
        self.nonregistration_probability = float(
            values.get("nonregistration_probability", 0.75)
        )
        super().__init__(
            scale=self.scale,
            patch_size=self.patch_size,
            hsi_bands=self.hsi_bands,
            msi_bands=self.msi_bands,
            width=int(values.get("width", 64)),
            blocks=int(values.get("blocks", 6)),
            diffusion_steps=int(values.get("diffusion_steps", 8)),
        )

    def training_loss(self, target: Tensor, lrhsi: Tensor, hrmsi: Tensor):
        lrhsi = random_unregistered_lrhsi(
            lrhsi, probability=self.nonregistration_probability,
        )
        return super().training_loss(target, lrhsi, hrmsi)

    def predict(self, lrhsi: Tensor, hrmsi: Tensor) -> Tensor:
        return self.sample(lrhsi, hrmsi)

    def forward(self, lrhsi: Tensor, hrmsi: Tensor) -> Tensor:
        return self.predict(lrhsi, hrmsi)


PSRFDiff = PSRFDiffNet
