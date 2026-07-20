"""Released ZSL x4 abundance CNN, expressed as a device-agnostic PyTorch module."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def fixed_x4_kernel(rank: int) -> torch.Tensor:
    base = torch.tensor([[
        -4.63495665e-03, -3.63442646e-03, 3.84904063e-18, 5.76678319e-03,
        1.08358664e-02, 1.01980790e-02, -9.31747402e-18, -1.75033181e-02,
        -3.17660068e-02, -2.84531643e-02, 1.85181518e-17, 4.42450253e-02,
        7.71733386e-02, 6.70554910e-02, -2.85299239e-17, -1.01548683e-01,
        -1.78708388e-01, -1.60004642e-01, 3.61741232e-17, 2.87940558e-01,
        6.25431459e-01, 8.97067600e-01, 1.00107877e+00, 8.97067600e-01,
        6.25431459e-01, 2.87940558e-01, 3.61741232e-17, -1.60004642e-01,
        -1.78708388e-01, -1.01548683e-01, -2.85299239e-17, 6.70554910e-02,
        7.71733386e-02, 4.42450253e-02, 1.85181518e-17, -2.84531643e-02,
        -3.17660068e-02, -1.75033181e-02, -9.31747402e-18, 1.01980790e-02,
        1.08358664e-02, 5.76678319e-03, 3.84904063e-18, -3.63442646e-03,
        -4.63495665e-03,
    ]], dtype=torch.float32)
    kernel = base.transpose(0, 1).matmul(base)[None, None]
    return kernel.repeat(rank, 1, 1, 1)


class ZSLStageCNN(nn.Module):
    """One unmodified x4 ZSL stage from the released implementation."""

    def __init__(self, rank: int, msi_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(rank + msi_channels, 128 - msi_channels, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128 - msi_channels, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128 - msi_channels, 3, 1, 1), nn.LeakyReLU(0.2))
        self.conv4 = nn.Conv2d(128, rank, 3, 1, 1)
        self.register_buffer("coefficient", fixed_x4_kernel(rank))

    def forward(self, abundance_lr: torch.Tensor, msi_hr: torch.Tensor) -> torch.Tensor:
        base = F.conv_transpose2d(
            abundance_lr, self.coefficient, stride=4, padding=21, output_padding=1,
            groups=abundance_lr.shape[1],
        )
        features = torch.cat((base, msi_hr), 1)
        features = torch.cat((self.conv1(features), msi_hr), 1)
        features = torch.cat((self.conv2(features), msi_hr), 1)
        features = torch.cat((self.conv3(features), msi_hr), 1)
        return self.conv4(features) + base


# Historical name retained for callers that used the bundled snapshot.
ZSL_cnn = ZSLStageCNN
