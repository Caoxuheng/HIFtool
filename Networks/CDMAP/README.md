# CDMAP for HIFTool

This directory contains one training-free CDMAP implementation with two
numerically matched execution backends:

- `cuda_backend.py`: fully GPU-resident fused Numba-CUDA kernel. One CUDA
  thread completes one HR pixel; the global angular tensor is not materialized.
- `cpu_backend.py`: strictly serial Numba implementation (`parallel=False`, no
  `prange`), completing one HR pixel at a time.

`CDMAP` uses `backend=auto`: CUDA is preferred, and an unavailable CUDA driver
or context triggers a visible warning and a CPU fallback. `CDMAP-CUDA` forces
CUDA and reports an error when unavailable. `CDMAP-CPU` forces serial CPU.
Runtime CUDA kernel failures are not silently hidden by a CPU retry.

## Nominal algorithm parameters

| Parameter | Value |
|---|---:|
| Scale factor | 32 |
| Candidate support | 4 x 4 (16 LR candidates) |
| Top K | 3 |
| Temperature tau | 0.1 |
| Ridge coefficient | 1e-6 |
| Precision | float32 |
| Spectral range | 400--700 nm, 31 bands |
| MSI channels | 3 |
| CPU threads | 1 |
| CUDA block | 16 x 16 |

The implementation uses the existing
`Dataloader_tool/srflib/NikonD700.npy` response matrix. No checkpoint or
pretrained weight is required.

## HIFTool commands

```bash
# CUDA preferred; serial CPU fallback when CUDA cannot initialize
python Network_eval_VL.py --method CDMAP --dataset CAVE --dataset-root D:/dataset
python Network_eval_VL.py --method CDMAP --dataset HARVARD --dataset-root D:/dataset

# Controlled backends
python Network_eval_VL.py --method CDMAP --dataset CAVE --cdmap-backend cuda
python Network_eval_VL.py --method CDMAP --dataset CAVE --cdmap-backend cpu
```

Use `--save-output` to save reconstructed cubes. The evaluator prints the
actual backend and the mean/sample-standard-deviation inference time. It runs
one untimed full-image warm-up by default; change this with `--cdmap-warmups`.

## Python API

```python
from Networks.CDMAP import CDMAP, CDMAPConfig

model = CDMAP(CDMAPConfig(backend="auto"))
reconstruction = model(lr_hsi_hwc, hr_msi_hwc)
print(model.backend_name)
```

## Scope

The released CUDA kernel is the validated CAVE/Harvard nominal configuration:
31 HSI bands, three MSI channels, 4x4 support, and Top-3 selection. Change the
implementation and add new CPU/CUDA parity tests before publishing other band
counts or support sizes.


The CUDA release was validated on Windows with an RTX 5090, driver 581.80,
CUDA Toolkit 13.0, and Numba 0.66.0. On that workstation the backend binds
Numba to the matching system CUDA 13.0 runtime when it is installed.
