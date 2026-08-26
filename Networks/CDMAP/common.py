"""Shared input validation and numerical preparation for CDMAP."""

from __future__ import annotations

import numpy as np

from .config import CDMAPConfig


def normalize_srf(srf: np.ndarray, bands: int, channels: int) -> np.ndarray:
    response = np.asarray(srf, dtype=np.float32)
    if response.shape == (channels, bands):
        response = response.T
    if response.shape != (bands, channels):
        raise ValueError(
            f"SRF must have shape ({bands}, {channels}) or ({channels}, {bands}); "
            f"got {response.shape}."
        )
    if not np.isfinite(response).all():
        raise ValueError("SRF contains NaN or Inf.")
    return np.ascontiguousarray(response)


def triangular_bases(wavelengths: np.ndarray, srf: np.ndarray) -> np.ndarray:
    centers = np.sort((wavelengths[:, None] * srf).sum(axis=0)).astype(np.float32)
    width = float(wavelengths.max() - wavelengths.min()) / 2.0 + 1e-6
    phi = np.stack(
        [
            np.clip(1.0 - np.abs(wavelengths - center) / width, 0.0, 1.0)
            for center in centers
        ],
        axis=1,
    ).astype(np.float32)
    phi /= phi.sum(axis=1, keepdims=True) + np.float32(1e-8)
    return np.ascontiguousarray(phi, dtype=np.float32)


def prepare_inputs(
    lr_hsi: np.ndarray,
    hr_msi: np.ndarray,
    srf: np.ndarray,
    config: CDMAPConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config.validate()
    lr = np.asarray(lr_hsi, dtype=np.float32)
    hr = np.asarray(hr_msi, dtype=np.float32)
    if lr.ndim != 3 or hr.ndim != 3:
        raise ValueError("CDMAP expects HWC arrays: LR-HSI [h,w,31] and HR-MSI [H,W,3].")
    if lr.shape[2] != config.hsi_channel:
        raise ValueError(f"LR-HSI must contain {config.hsi_channel} bands; got {lr.shape}.")
    if hr.shape[2] != config.msi_channel:
        raise ValueError(f"HR-MSI must contain {config.msi_channel} channels; got {hr.shape}.")
    expected = (lr.shape[0] * config.sf, lr.shape[1] * config.sf)
    if hr.shape[:2] != expected:
        raise ValueError(
            f"HR-MSI spatial shape must be LR-HSI shape multiplied by sf={config.sf}; "
            f"expected {expected}, got {hr.shape[:2]}."
        )
    if not np.isfinite(lr).all() or not np.isfinite(hr).all():
        raise ValueError("Input images contain NaN or Inf.")

    response = normalize_srf(srf, config.hsi_channel, config.msi_channel)
    wavelengths = np.linspace(
        config.wavelength_start_nm,
        config.wavelength_end_nm,
        config.hsi_channel,
        dtype=np.float32,
    )
    phi = triangular_bases(wavelengths, response)

    pad = config.research_scale
    # NumPy ``symmetric`` includes the edge pixel and matches cv2.BORDER_REFLECT.
    padded = np.pad(lr, ((pad, pad), (pad, pad), (0, 0)), mode="symmetric")
    padded = np.ascontiguousarray(padded, dtype=np.float32)
    projected = np.matmul(padded, response).astype(np.float32)
    normalized = projected / (projected.max(axis=2, keepdims=True) + np.float32(1e-12))
    return (
        padded,
        np.ascontiguousarray(hr, dtype=np.float32),
        np.ascontiguousarray(normalized, dtype=np.float32),
        response,
        phi,
    )
