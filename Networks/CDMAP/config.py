"""Configuration for the training-free CDMAP fusion method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CDMAPConfig:
    """Algorithm and execution parameters used by both CDMAP backends.

    The nominal values reproduce the controlled CAVE/Harvard experiments.
    ``backend='auto'`` prefers Numba CUDA and falls back to the strictly
    serial Numba CPU implementation when CUDA cannot be initialized.
    """

    backend: str = "auto"
    sf: int = 32
    hsi_channel: int = 31
    msi_channel: int = 3
    research_scale: int = 2
    top_k: int = 3
    tau: float = 0.1
    ridge_lambda: float = 1e-6
    wavelength_start_nm: float = 400.0
    wavelength_end_nm: float = 700.0
    block_size: int = 16
    cpu_threads: int = 1
    clip_output: bool = False
    srfpath: str = "Dataloader_tool/srflib/NikonD700.npy"
    dataset_root: str = "D:/CaoXuheng/dataset"
    cave_split: str = "official"

    @property
    def support_width(self) -> int:
        return 2 * self.research_scale

    @property
    def candidate_count(self) -> int:
        return self.support_width * self.support_width

    def validate(self) -> None:
        if self.backend.lower() not in {"auto", "cuda", "cpu"}:
            raise ValueError("backend must be one of: auto, cuda, cpu")
        if self.sf <= 0:
            raise ValueError("sf must be positive")
        if self.hsi_channel != 31 or self.msi_channel != 3:
            raise ValueError("This released CDMAP kernel supports 31-band HSI and 3-channel MSI.")
        if self.research_scale != 2:
            raise ValueError("This released CDMAP kernel uses the nominal 4x4 candidate support.")
        if self.top_k != 3:
            raise ValueError("This released CDMAP kernel uses nominal top_k=3.")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.block_size <= 0 or self.block_size * self.block_size > 1024:
            raise ValueError("block_size must be positive and block_size**2 must not exceed 1024")
        if self.cpu_threads != 1:
            raise ValueError("The published CDMAP CPU backend is strictly single-threaded.")

    def resolved_srf_path(self, repository_root: str | Path | None = None) -> Path:
        path = Path(self.srfpath)
        if path.is_absolute() or repository_root is None:
            return path
        return Path(repository_root) / path


args = CDMAPConfig()
