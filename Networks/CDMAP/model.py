"""Unified public model selecting the CDMAP CUDA or CPU backend."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .common import prepare_inputs
from .config import CDMAPConfig


class CDMAP:
    """Training-free CDMAP fusion with transparent backend selection.

    Parameters
    ----------
    config:
        Algorithm/runtime configuration. The default backend is ``auto``.
    srf:
        Optional 31x3 spectral response matrix. When omitted, the Nikon D700
        matrix already distributed with HIFTool is loaded.
    """

    training_free = True

    def __init__(self, config: CDMAPConfig | None = None, srf: np.ndarray | None = None):
        self.config = config or CDMAPConfig()
        self.config.validate()
        self._repository_root = Path(__file__).resolve().parents[2]
        self._srf = self._load_srf() if srf is None else np.asarray(srf, dtype=np.float32)
        self.requested_backend = self.config.backend.lower()
        self.backend_name, self.backend_detail = self._select_backend(self.requested_backend)
        print(f"CDMAP backend: {self.backend_name.upper()} ({self.backend_detail})")

    def _load_srf(self) -> np.ndarray:
        path = self.config.resolved_srf_path(self._repository_root)
        if not path.is_file():
            raise FileNotFoundError(
                f"CDMAP SRF file was not found: {path}. "
                "Keep Dataloader_tool/srflib/NikonD700.npy in the HIFTool repository."
            )
        return np.asarray(np.load(path), dtype=np.float32)

    @staticmethod
    def _probe_cuda() -> tuple[bool, str]:
        try:
            from .cuda_backend import probe_cuda

            return probe_cuda()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    @classmethod
    def _select_backend(cls, requested: str) -> tuple[str, str]:
        if requested == "cpu":
            return "cpu", "forced strictly single-threaded Numba CPU"

        available, detail = cls._probe_cuda()
        if available:
            return "cuda", detail
        if requested == "cuda":
            raise RuntimeError(
                "CDMAP-CUDA was explicitly requested, but Numba CUDA could not be initialized: "
                f"{detail}"
            )
        warnings.warn(
            "CDMAP could not initialize Numba CUDA and will use the strictly single-threaded "
            f"CPU backend. Reason: {detail}",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu", f"automatic fallback; {detail}"

    def __call__(
        self,
        lr_hsi: np.ndarray,
        hr_msi: np.ndarray,
        srf: np.ndarray | None = None,
    ) -> np.ndarray:
        response = self._srf if srf is None else np.asarray(srf, dtype=np.float32)
        prepared = prepare_inputs(lr_hsi, hr_msi, response, self.config)

        if self.backend_name == "cuda":
            from .cuda_backend import fuse_cuda

            output = fuse_cuda(
                *prepared,
                self.config.sf,
                self.config.tau,
                self.config.ridge_lambda,
                self.config.block_size,
            )
        else:
            from .cpu_backend import fuse_cpu

            output = fuse_cpu(
                *prepared,
                self.config.sf,
                self.config.tau,
                self.config.ridge_lambda,
            )

        if not np.isfinite(output).all():
            raise FloatingPointError(f"CDMAP {self.backend_name} backend produced NaN or Inf.")
        if self.config.clip_output:
            output = np.clip(output, 0.0, 1.0)
        return np.ascontiguousarray(output, dtype=np.float32)

    def fuse(self, lr_hsi: np.ndarray, hr_msi: np.ndarray, srf: np.ndarray | None = None) -> np.ndarray:
        return self(lr_hsi, hr_msi, srf=srf)
