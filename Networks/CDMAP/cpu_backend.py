"""Strictly serial Numba CPU backend for CDMAP."""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import njit


@njit(cache=True, parallel=False, fastmath=False)
def _fuse_serial_kernel(
    padded_hsi: np.ndarray,
    hr_msi: np.ndarray,
    lr_normalized: np.ndarray,
    srf: np.ndarray,
    phi: np.ndarray,
    scale_factor: int,
    tau: np.float32,
    ridge_lambda: np.float32,
) -> np.ndarray:
    """Complete one HR pixel before moving to the next; no prange is used."""

    height, width, _ = hr_msi.shape
    bands = padded_hsi.shape[2]
    output = np.empty((height, width, bands), np.float32)
    eps12 = np.float32(1e-12)
    eps20 = np.float32(1e-20)
    eps8 = np.float32(1e-8)

    angles = np.empty(16, np.float32)
    selected = np.empty(3, np.int32)
    selected_angle = np.empty(3, np.float32)
    used = np.empty(16, np.uint8)
    weights = np.empty(3, np.float32)
    prior = np.empty(31, np.float32)
    projection = np.empty((3, 3), np.float32)

    for row in range(height):
        base_i = row // scale_factor
        for col in range(width):
            base_j = col // scale_factor
            hr0 = hr_msi[row, col, 0]
            hr1 = hr_msi[row, col, 1]
            hr2 = hr_msi[row, col, 2]
            hr_max = max(hr0, hr1, hr2)
            hn0 = np.float32(hr0 / (hr_max + eps12))
            hn1 = np.float32(hr1 / (hr_max + eps12))
            hn2 = np.float32(hr2 / (hr_max + eps12))
            hr_norm2 = np.float32(hn0 * hn0 + hn1 * hn1 + hn2 * hn2)

            candidate = 0
            for di in range(4):
                for dj in range(4):
                    ln0 = lr_normalized[base_i + di, base_j + dj, 0]
                    ln1 = lr_normalized[base_i + di, base_j + dj, 1]
                    ln2 = lr_normalized[base_i + di, base_j + dj, 2]
                    dot = np.float32(hn0 * ln0 + hn1 * ln1 + hn2 * ln2)
                    lr_norm2 = np.float32(ln0 * ln0 + ln1 * ln1 + ln2 * ln2)
                    denominator = max(
                        np.float32(math.sqrt(hr_norm2)) * np.float32(math.sqrt(lr_norm2)),
                        eps20,
                    )
                    cosine = min(np.float32(1.0), max(np.float32(-1.0), np.float32(dot / denominator)))
                    angles[candidate] = np.float32(math.acos(cosine))
                    used[candidate] = 0
                    candidate += 1

            for rank in range(3):
                best_index = -1
                best_angle = np.float32(math.inf)
                for candidate in range(16):
                    if used[candidate] == 0 and angles[candidate] < best_angle:
                        best_angle = angles[candidate]
                        best_index = candidate
                selected[rank] = best_index
                selected_angle[rank] = best_angle
                used[best_index] = 1

            minimum_angle = selected_angle[0]
            weight_sum = np.float32(0.0)
            for rank in range(3):
                weights[rank] = np.float32(
                    math.exp(np.float32(-(selected_angle[rank] - minimum_angle) / tau))
                )
                weight_sum = np.float32(weight_sum + weights[rank])
            for rank in range(3):
                weights[rank] = np.float32(weights[rank] / (weight_sum + eps12))

            for band in range(31):
                value = np.float32(0.0)
                for rank in range(3):
                    index = selected[rank]
                    value = np.float32(
                        value
                        + weights[rank]
                        * padded_hsi[base_i + index // 4, base_j + index % 4, band]
                    )
                prior[band] = value

            for basis in range(3):
                for channel in range(3):
                    value = np.float32(0.0)
                    for band in range(31):
                        value = np.float32(
                            value + prior[band] * phi[band, basis] * srf[band, channel]
                        )
                    projection[basis, channel] = value

            # Explicit float32 3x3 Gauss-Jordan solve. Operation order and
            # pivot tie-breaking match the released CUDA kernel/reference.
            p00, p01, p02 = projection[0, 0], projection[0, 1], projection[0, 2]
            p10, p11, p12 = projection[1, 0], projection[1, 1], projection[1, 2]
            p20, p21, p22 = projection[2, 0], projection[2, 1], projection[2, 2]
            a00 = np.float32(p00*p00 + p01*p01 + p02*p02 + ridge_lambda)
            a01 = np.float32(p00*p10 + p01*p11 + p02*p12)
            a02 = np.float32(p00*p20 + p01*p21 + p02*p22)
            a10 = a01
            a11 = np.float32(p10*p10 + p11*p11 + p12*p12 + ridge_lambda)
            a12 = np.float32(p10*p20 + p11*p21 + p12*p22)
            a20 = a02
            a21 = a12
            a22 = np.float32(p20*p20 + p21*p21 + p22*p22 + ridge_lambda)
            i00, i01, i02 = np.float32(1.0), np.float32(0.0), np.float32(0.0)
            i10, i11, i12 = np.float32(0.0), np.float32(1.0), np.float32(0.0)
            i20, i21, i22 = np.float32(0.0), np.float32(0.0), np.float32(1.0)

            pivot = 0
            absolute0, absolute1, absolute2 = abs(a00), abs(a10), abs(a20)
            if absolute1 > absolute0 and absolute1 >= absolute2:
                pivot = 1
            elif absolute2 > absolute0 and absolute2 > absolute1:
                pivot = 2
            if pivot == 1:
                a00,a01,a02,a10,a11,a12 = a10,a11,a12,a00,a01,a02
                i00,i01,i02,i10,i11,i12 = i10,i11,i12,i00,i01,i02
            elif pivot == 2:
                a00,a01,a02,a20,a21,a22 = a20,a21,a22,a00,a01,a02
                i00,i01,i02,i20,i21,i22 = i20,i21,i22,i00,i01,i02
            pivot_value = a00 if abs(a00) > eps8 else (eps8 if a00 >= 0 else -eps8)
            inverse_pivot = np.float32(1.0) / pivot_value
            a00,a01,a02 = a00*inverse_pivot,a01*inverse_pivot,a02*inverse_pivot
            i00,i01,i02 = i00*inverse_pivot,i01*inverse_pivot,i02*inverse_pivot
            factor = a10
            a10,a11,a12 = a10-factor*a00,a11-factor*a01,a12-factor*a02
            i10,i11,i12 = i10-factor*i00,i11-factor*i01,i12-factor*i02
            factor = a20
            a20,a21,a22 = a20-factor*a00,a21-factor*a01,a22-factor*a02
            i20,i21,i22 = i20-factor*i00,i21-factor*i01,i22-factor*i02

            if abs(a21) > abs(a11):
                a10,a11,a12,a20,a21,a22 = a20,a21,a22,a10,a11,a12
                i10,i11,i12,i20,i21,i22 = i20,i21,i22,i10,i11,i12
            pivot_value = a11 if abs(a11) > eps8 else (eps8 if a11 >= 0 else -eps8)
            inverse_pivot = np.float32(1.0) / pivot_value
            a10,a11,a12 = a10*inverse_pivot,a11*inverse_pivot,a12*inverse_pivot
            i10,i11,i12 = i10*inverse_pivot,i11*inverse_pivot,i12*inverse_pivot
            factor = a01
            a01,a02,a00 = a01-factor*a11,a02-factor*a12,a00-factor*a10
            i01,i02,i00 = i01-factor*i11,i02-factor*i12,i00-factor*i10
            factor = a21
            a21,a22,a20 = a21-factor*a11,a22-factor*a12,a20-factor*a10
            i21,i22,i20 = i21-factor*i11,i22-factor*i12,i20-factor*i10

            pivot_value = a22 if abs(a22) > eps8 else (eps8 if a22 >= 0 else -eps8)
            inverse_pivot = np.float32(1.0) / pivot_value
            a20,a21,a22 = a20*inverse_pivot,a21*inverse_pivot,a22*inverse_pivot
            i20,i21,i22 = i20*inverse_pivot,i21*inverse_pivot,i22*inverse_pivot
            factor = a02
            a02,a01,a00 = a02-factor*a22,a01-factor*a21,a00-factor*a20
            i02,i01,i00 = i02-factor*i22,i01-factor*i21,i00-factor*i20
            factor = a12
            a12,a11,a10 = a12-factor*a22,a11-factor*a21,a10-factor*a20
            i12,i11,i10 = i12-factor*i22,i11-factor*i21,i10-factor*i20

            atb0 = np.float32(p00*hr0 + p01*hr1 + p02*hr2)
            atb1 = np.float32(p10*hr0 + p11*hr1 + p12*hr2)
            atb2 = np.float32(p20*hr0 + p21*hr1 + p22*hr2)
            gain0 = np.float32(i00*atb0 + i01*atb1 + i02*atb2)
            gain1 = np.float32(i10*atb0 + i11*atb1 + i12*atb2)
            gain2 = np.float32(i20*atb0 + i21*atb1 + i22*atb2)
            for band in range(31):
                output[row, col, band] = np.float32(
                    prior[band]
                    * (
                        phi[band, 0] * gain0
                        + phi[band, 1] * gain1
                        + phi[band, 2] * gain2
                    )
                )

    return output


def fuse_cpu(
    padded_hsi: np.ndarray,
    hr_msi: np.ndarray,
    lr_normalized: np.ndarray,
    srf: np.ndarray,
    phi: np.ndarray,
    scale_factor: int,
    tau: float,
    ridge_lambda: float,
) -> np.ndarray:
    """Run the published strictly single-threaded CPU implementation."""

    numba.set_num_threads(1)
    return _fuse_serial_kernel(
        padded_hsi,
        hr_msi,
        lr_normalized,
        srf,
        phi,
        scale_factor,
        np.float32(tau),
        np.float32(ridge_lambda),
    )
