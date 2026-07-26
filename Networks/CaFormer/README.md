# CaFormer replacement for HIFTool

This folder is a drop-in replacement for `Networks/CaFormer`. It keeps the
released CaFormer parameter names and the existing HIFTool call
`model(lr_hsi, hr_msi)`, while adding:

- integer spatial scales selected by `--sf`;
- exact output-size interpolation instead of scale-factor rounding;
- internal padding/cropping for spatial sizes that are not multiples of 16;
- aligned tiled inference for large CAVE/Harvard images;
- optional blind fusion using uHNTC `SpaDNet` and `SpeDNet`.

## 1. Supervised/known-degradation use

The existing HIFTool registry remains valid:

```python
model, opt = Networks.model_generator("CaFormer_3", "cuda")
prediction = model(lr_hsi, hr_msi)
```

Select the scale on the command line:

```text
python Network_training_VL.py --sf 4
python Network_training_VL.py --sf 8
python Network_training_VL.py --sf 16
python Network_training_VL.py --sf 32
```

A checkpoint trained at one scale should not be presented as trained at a
different scale. The architecture accepts a new scale, but fair evaluation
still requires the corresponding supervised checkpoint unless the original
method explicitly supports cross-scale evaluation.

Standalone construction:

```python
from Networks.CaFormer import build_caformer, forward_tiled

model = build_caformer(scale=32, total_stages=3, device="cuda")
prediction = forward_tiled(
    model, lr_hsi, hr_msi, scale=32, tile_size=256, overlap=32
)
```

## 2. Blind fusion

Blind fusion starts from the supervised general CaFormer checkpoint. For each
test observation, uHNTC estimates the spatial and spectral degradation from
the LR-HSI/HR-MSI pair. The frozen estimated degraders supervise per-image
CaFormer adaptation through:

```text
Lblind = |SpaDNet(Xhat) - LR-HSI|_1
       + |SpeDNet(Xhat) - HR-MSI|_1
```

This uses neither ground truth nor the true SRF/PSF. “Supervision” here means
observation supervision supplied by the estimated uHNTC degraders.

```python
from Networks.CaFormer import UHNTCConfig, build_blind_caformer

config = UHNTCConfig(
    scale=32,
    adaptation_steps=1000,
    adaptation_lr=3e-5,
    tile_size=256,
    tile_overlap=32,
)
blind_model = build_blind_caformer(
    "PretrainModel/CaFormer/CAVE/best.pth", config, device="cuda"
)
result = blind_model(lr_hsi, hr_msi)
prediction = result.reconstruction
print(result.diagnostics)
```

The base model is deep-copied per observation, so adaptation of one scene
cannot contaminate another. uHNTC currently requires batch size 1 for blind
operator estimation. When `tile_size` is set, blind adaptation cycles through
aligned LR/HR patch pairs and final inference uses overlap averaging; this is
the memory-safe path for full Harvard scenes.

For non-31-band HSI or non-3-band MSI, pass explicit spectral supports:

```python
config = UHNTCConfig(
    scale=8,
    hsi_channels=64,
    msi_channels=4,
    spectral_spans=(
        tuple(range(0, 22)),
        tuple(range(14, 38)),
        tuple(range(30, 54)),
        tuple(range(46, 64)),
    ),
)
```

## 3. Installation

1. Stop jobs that import `Networks/CaFormer`.
2. Back up the existing directory.
3. Replace the complete `Networks/CaFormer` directory with this folder.
4. Keep `Networks/FeafusFormer` installed; its degradation modules are the
   uHNTC dependency used only in blind mode.
5. Run `python Networks/CaFormer/tests/smoke_test.py`.

No change to `Networks/__init__.py` is required for the existing supervised
`CaFormer_3` entry. Blind mode is intentionally explicit through
`Networks.CaFormer.build_blind_caformer`, so an experiment cannot silently
switch from supervised inference to test-time adaptation.

## 4. Reproducibility defaults

The blind defaults match the currently evaluated HIFTool version:

- SpeDNet initialization: 500 steps;
- SpaDNet initialization: 500 steps;
- coupled uHNTC initialization: 500 steps;
- frozen degradation estimators during CaFormer adaptation;
- CaFormer adaptation: 1000 steps, Adam, learning rate `3e-5`;
- spectral supports for CAVE/Harvard Nikon-D700 simulation:
  `[18:31]`, `[10:23]`, `[0:12]`.

All counts are configurable. Changing them must be recorded in the experiment
metadata rather than silently changing the comparison protocol.
