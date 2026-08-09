"""HIFTool adapter for the official BHSR-Net implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import make_options
from .official.loss import Losses
from .official.models import BHSR as _OfficialBHSR


class BHSRNet(nn.Module):
    """Released ten-stage BHSR architecture with HIFTool tensor ordering."""

    learning_rate = 1e-3
    batch_size = 8
    requires_specialized_loss = True
    source_url = "https://github.com/Dou0405/BHSR-Net"
    source_commit = "d5fa43ae36b2ac2831856cbace0de1c9116a8749"

    def __init__(self, options=None, **overrides) -> None:
        super().__init__()
        opt = options or make_options()
        values = vars(opt).copy()
        values.update(overrides)
        self.scale = int(values.get("sf", values.get("scale", 32)))
        self.hsi_bands = int(values.get("hsi_channel", values.get("hsi_bands", 31)))
        self.msi_bands = int(values.get("msi_channel", values.get("msi_bands", 3)))
        self.stages = int(values.get("stages", 10))
        self.net = _OfficialBHSR(
            stage_nBHSR=self.stages,
            C=self.hsi_bands,
            c=self.msi_bands,
            sigma1=float(values.get("sigma1", 1.0)),
            sigma2=float(values.get("sigma2", 2.0)),
        )
        # Official 1.0 assumes raw radiometry. HIFTool tensors are [0,1]; this
        # is the numeric x32 adaptation used by the retained checkpoints.
        update_init = float(values.get("update_init", 0.003))
        for parameter in (
            self.net.gx_update_param,
            self.net.l1_update_param,
            self.net.y_update_param,
            self.net.l2_update_param,
        ):
            parameter.data.fill_(update_init)
        self.criterion = Losses(
            scale=self.scale,
            model_name="BHSR",
            blur=float(values.get("blur_level", 0.0)),
        )

    def forward_all(self, lrhsi: Tensor, hrmsi: Tensor):
        if lrhsi.ndim != 4 or hrmsi.ndim != 4:
            raise ValueError("BHSR-Net expects BCHW LR-HSI and HR-MSI tensors")
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
        return self.net(lrhsi, hrmsi)

    def forward(self, lrhsi: Tensor, hrmsi: Tensor) -> Tensor:
        stages, _ = self.forward_all(lrhsi, hrmsi)
        return stages[-1]

    def training_loss(
        self, target: Tensor, lrhsi: Tensor, hrmsi: Tensor, epoch: int = 0,
    ):
        stages, restored_msi = self.forward_all(lrhsi, hrmsi)
        loss = self.criterion(stages, target, restored_msi, hrmsi, int(epoch))
        with torch.no_grad():
            final_l1 = torch.nn.functional.l1_loss(stages[-1], target)
        return loss, {
            "official_composite_loss": float(loss.detach()),
            "final_l1": float(final_l1.detach()),
        }


BHSR = BHSRNet
