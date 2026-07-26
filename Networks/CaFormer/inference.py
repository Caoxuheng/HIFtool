"""Resolution-safe inference helpers for CaFormer."""

from __future__ import annotations

import torch


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def forward_tiled(
    model,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    scale: int,
    tile_size: int | None = None,
    overlap: int = 0,
) -> torch.Tensor:
    """Run CaFormer on a complete image or aligned overlapping tiles.

    The HR size must equal the LR size multiplied by ``scale`` when tiling is
    requested. Full-image inference also supports non-power-of-two and
    non-multiple-of-16 sizes because ``CaFormer.forward`` pads internally.
    """
    if scale < 1:
        raise ValueError("scale must be a positive integer")
    if tile_size is None or tile_size <= 0:
        return model(lr_hsi, hr_msi)
    height, width = hr_msi.shape[-2:]
    if height <= tile_size and width <= tile_size:
        return model(lr_hsi, hr_msi)
    if tile_size % scale or overlap % scale:
        raise ValueError("tile_size and overlap must be divisible by scale")
    expected = (lr_hsi.shape[-2] * scale, lr_hsi.shape[-1] * scale)
    if (height, width) != expected:
        raise ValueError(
            f"tiled inference requires HR size {expected}, got {(height, width)}"
        )

    y_starts = _starts(height, min(tile_size, height), overlap)
    x_starts = _starts(width, min(tile_size, width), overlap)
    output = torch.zeros(
        (lr_hsi.shape[0], lr_hsi.shape[1], height, width),
        dtype=lr_hsi.dtype,
        device=lr_hsi.device,
    )
    weight = torch.zeros_like(output[:, :1])
    for top in y_starts:
        for left in x_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            top = bottom - min(tile_size, height)
            left = right - min(tile_size, width)
            lr_top, lr_left = top // scale, left // scale
            lr_bottom, lr_right = bottom // scale, right // scale
            prediction = model(
                lr_hsi[:, :, lr_top:lr_bottom, lr_left:lr_right],
                hr_msi[:, :, top:bottom, left:right],
            )
            output[:, :, top:bottom, left:right] += prediction
            weight[:, :, top:bottom, left:right] += 1
    return output / weight.clamp_min_(1)
