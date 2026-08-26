"""Fully GPU-resident fused Numba-CUDA backend for CDMAP."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


def _bind_matching_windows_cuda_runtime() -> str | None:
    """Bind Numba to an installed CUDA 13.0 toolkit before CUDA is imported.

    The RTX 5090 workstation also has pip CUDA runtime files newer than its
    system linker. Selecting the matching system NVVM/cudart/libdevice avoids
    PTX-version mismatches. Other hosts simply keep Numba's normal discovery.
    """

    if os.name != "nt":
        return None
    cuda_root = Path(
        os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")
    )
    nvvm = cuda_root / "nvvm" / "bin" / "x64" / "nvvm64_40_0.dll"
    cudart = cuda_root / "bin" / "x64" / "cudart64_13.dll"
    libdevice = cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    if not all(path.is_file() for path in (nvvm, cudart, libdevice)):
        return None
    try:
        from numba.cuda.cudadrv import libs

        if not getattr(libs, "_cdmap_runtime_bound", False):
            original_get_cudalib = libs.get_cudalib

            def get_cudalib(library: str, static: bool = False) -> str:
                if library == "nvvm":
                    return str(nvvm)
                if library == "cudart":
                    return str(cudart)
                return original_get_cudalib(library, static)

            libs.get_cudalib = get_cudalib
            libs.get_libdevice = lambda: str(libdevice)
            libs._cdmap_runtime_bound = True
        return str(cuda_root)
    except Exception:
        return None


CUDA_RUNTIME_OVERRIDE = _bind_matching_windows_cuda_runtime()

from numba import cuda, float32, int32


def probe_cuda() -> tuple[bool, str]:
    """Check the backend CDMAP actually uses, including context creation."""

    try:
        if not cuda.is_available():
            return False, "numba.cuda.is_available() returned False"
        device = cuda.get_current_device()
        cuda.current_context()
        name = device.name.decode(errors="replace") if isinstance(device.name, bytes) else str(device.name)
        detail = name
        if CUDA_RUNTIME_OVERRIDE:
            detail += f"; toolkit={CUDA_RUNTIME_OVERRIDE}"
        return True, detail
    except Exception as exc:  # CUDA driver/runtime exceptions vary by Numba release.
        return False, f"{type(exc).__name__}: {exc}"


@cuda.jit
def _fuse_resident_kernel(hsi_pad, hr_msi, lr_normalized, srf, phi, output, sf, tau, ridge):
    row, col = cuda.grid(2)
    height, width, _ = output.shape
    if row >= height or col >= width:
        return

    eps12 = float32(1e-12)
    eps20 = float32(1e-20)
    eps8 = float32(1e-8)
    hr0, hr1, hr2 = hr_msi[row, col, 0], hr_msi[row, col, 1], hr_msi[row, col, 2]
    hmax = hr0
    if hr1 > hmax:
        hmax = hr1
    if hr2 > hmax:
        hmax = hr2
    hn0 = hr0 / (hmax + eps12)
    hn1 = hr1 / (hmax + eps12)
    hn2 = hr2 / (hmax + eps12)
    hnorm2 = hn0 * hn0 + hn1 * hn1 + hn2 * hn2
    base_i, base_j = row // sf, col // sf

    best_angle = cuda.local.array(3, float32)
    best_index = cuda.local.array(3, int32)
    prior = cuda.local.array(31, float32)
    best_angle[0] = best_angle[1] = best_angle[2] = float32(1e30)
    best_index[0] = best_index[1] = best_index[2] = 0

    for candidate in range(16):
        di, dj = candidate // 4, candidate % 4
        l0 = lr_normalized[base_i + di, base_j + dj, 0]
        l1 = lr_normalized[base_i + di, base_j + dj, 1]
        l2 = lr_normalized[base_i + di, base_j + dj, 2]
        dot = hn0 * l0 + hn1 * l1 + hn2 * l2
        lnorm2 = l0 * l0 + l1 * l1 + l2 * l2
        norm_product = hnorm2 * lnorm2
        if norm_product < eps20:
            norm_product = eps20
        cosine = dot / math.sqrt(norm_product)
        if cosine > float32(1.0):
            cosine = float32(1.0)
        elif cosine < float32(-1.0):
            cosine = float32(-1.0)
        angle = math.acos(cosine)
        if angle < best_angle[0]:
            best_angle[2], best_index[2] = best_angle[1], best_index[1]
            best_angle[1], best_index[1] = best_angle[0], best_index[0]
            best_angle[0], best_index[0] = angle, candidate
        elif angle < best_angle[1]:
            best_angle[2], best_index[2] = best_angle[1], best_index[1]
            best_angle[1], best_index[1] = angle, candidate
        elif angle < best_angle[2]:
            best_angle[2], best_index[2] = angle, candidate

    weight0 = float32(1.0)
    weight1 = math.exp(-(best_angle[1] - best_angle[0]) / tau)
    weight2 = math.exp(-(best_angle[2] - best_angle[0]) / tau)
    weight_sum = weight0 + weight1 + weight2 + eps12
    weight0, weight1, weight2 = weight0 / weight_sum, weight1 / weight_sum, weight2 / weight_sum
    index0, index1, index2 = best_index[0], best_index[1], best_index[2]
    for band in range(31):
        prior[band] = (
            weight0 * hsi_pad[base_i + index0 // 4, base_j + index0 % 4, band]
            + weight1 * hsi_pad[base_i + index1 // 4, base_j + index1 % 4, band]
            + weight2 * hsi_pad[base_i + index2 // 4, base_j + index2 % 4, band]
        )

    p00=p01=p02=p10=p11=p12=p20=p21=p22=float32(0.0)
    for band in range(31):
        value = prior[band]
        basis0, basis1, basis2 = value*phi[band,0], value*phi[band,1], value*phi[band,2]
        response0, response1, response2 = srf[band,0], srf[band,1], srf[band,2]
        p00+=basis0*response0; p01+=basis0*response1; p02+=basis0*response2
        p10+=basis1*response0; p11+=basis1*response1; p12+=basis1*response2
        p20+=basis2*response0; p21+=basis2*response1; p22+=basis2*response2

    a00=p00*p00+p01*p01+p02*p02+ridge; a01=p00*p10+p01*p11+p02*p12; a02=p00*p20+p01*p21+p02*p22
    a10=a01; a11=p10*p10+p11*p11+p12*p12+ridge; a12=p10*p20+p11*p21+p12*p22
    a20=a02; a21=a12; a22=p20*p20+p21*p21+p22*p22+ridge
    i00,i01,i02=float32(1),float32(0),float32(0)
    i10,i11,i12=float32(0),float32(1),float32(0)
    i20,i21,i22=float32(0),float32(0),float32(1)
    if abs(a10)>abs(a00) and abs(a10)>=abs(a20):
        a00,a01,a02,a10,a11,a12=a10,a11,a12,a00,a01,a02
        i00,i01,i02,i10,i11,i12=i10,i11,i12,i00,i01,i02
    elif abs(a20)>abs(a00) and abs(a20)>abs(a10):
        a00,a01,a02,a20,a21,a22=a20,a21,a22,a00,a01,a02
        i00,i01,i02,i20,i21,i22=i20,i21,i22,i00,i01,i02
    pivot=a00 if abs(a00)>eps8 else (eps8 if a00>=0 else -eps8); inv=float32(1)/pivot
    a00*=inv;a01*=inv;a02*=inv;i00*=inv;i01*=inv;i02*=inv
    factor=a10;a10-=factor*a00;a11-=factor*a01;a12-=factor*a02;i10-=factor*i00;i11-=factor*i01;i12-=factor*i02
    factor=a20;a20-=factor*a00;a21-=factor*a01;a22-=factor*a02;i20-=factor*i00;i21-=factor*i01;i22-=factor*i02
    if abs(a21)>abs(a11):
        a10,a11,a12,a20,a21,a22=a20,a21,a22,a10,a11,a12
        i10,i11,i12,i20,i21,i22=i20,i21,i22,i10,i11,i12
    pivot=a11 if abs(a11)>eps8 else (eps8 if a11>=0 else -eps8);inv=float32(1)/pivot
    a10*=inv;a11*=inv;a12*=inv;i10*=inv;i11*=inv;i12*=inv
    factor=a01;a00-=factor*a10;a01-=factor*a11;a02-=factor*a12;i00-=factor*i10;i01-=factor*i11;i02-=factor*i12
    factor=a21;a20-=factor*a10;a21-=factor*a11;a22-=factor*a12;i20-=factor*i10;i21-=factor*i11;i22-=factor*i12
    pivot=a22 if abs(a22)>eps8 else (eps8 if a22>=0 else -eps8);inv=float32(1)/pivot
    a20*=inv;a21*=inv;a22*=inv;i20*=inv;i21*=inv;i22*=inv
    factor=a02;a00-=factor*a20;a01-=factor*a21;a02-=factor*a22;i00-=factor*i20;i01-=factor*i21;i02-=factor*i22
    factor=a12;a10-=factor*a20;a11-=factor*a21;a12-=factor*a22;i10-=factor*i20;i11-=factor*i21;i12-=factor*i22
    target0=p00*hr0+p01*hr1+p02*hr2
    target1=p10*hr0+p11*hr1+p12*hr2
    target2=p20*hr0+p21*hr1+p22*hr2
    gain0=i00*target0+i01*target1+i02*target2
    gain1=i10*target0+i11*target1+i12*target2
    gain2=i20*target0+i21*target1+i22*target2
    for band in range(31):
        output[row,col,band]=prior[band]*(phi[band,0]*gain0+phi[band,1]*gain1+phi[band,2]*gain2)


def fuse_cuda(
    padded_hsi: np.ndarray,
    hr_msi: np.ndarray,
    lr_normalized: np.ndarray,
    srf: np.ndarray,
    phi: np.ndarray,
    scale_factor: int,
    tau: float,
    ridge_lambda: float,
    block_size: int = 16,
) -> np.ndarray:
    """Keep all inputs and intermediates resident until the final output copy."""

    device_arrays = [cuda.to_device(array) for array in (padded_hsi, hr_msi, lr_normalized, srf, phi)]
    device_output = cuda.device_array(
        (hr_msi.shape[0], hr_msi.shape[1], padded_hsi.shape[2]), dtype=np.float32
    )
    threads = (block_size, block_size)
    blocks = (
        (hr_msi.shape[0] + block_size - 1) // block_size,
        (hr_msi.shape[1] + block_size - 1) // block_size,
    )
    _fuse_resident_kernel[blocks, threads](
        *device_arrays,
        device_output,
        scale_factor,
        np.float32(tau),
        np.float32(ridge_lambda),
    )
    cuda.synchronize()
    return device_output.copy_to_host()
