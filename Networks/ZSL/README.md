# HIFTool ZSL integration

`model_generator('ZSL')` now returns a per-image self-supervised ZSL model.
Unlike a checkpoint model, its `forward(lr_hsi, hr_msi)` performs zero-shot
optimisation on that image and therefore must not be wrapped in `torch.no_grad()`.

## Scale routing

- `sf=4`: one literal released ZSL x4 stage.
- `sf>16`: automatic high-scale cascade. HIFTool first applies a fixed bicubic
  bridge of `sf / 16`, then trains two independent, unchanged ZSL x4 stages.
  Thus x32 is exactly `bicubic x2 -> ZSL x4 -> ZSL x4`.

The integrated protocol entry point is:

```powershell
python Networks/ZSL/run_protocol.py --sf 32 --zsl-steps 1000 `
  --output-root Benchmarks/CAVE_sequential_60_40/ZSL
```

It uses the HIFTool `sequential_60_40` test split by default and never feeds
HR-HSI ground truth into the zero-shot optimisation. `--zsl-steps` is the
per-scene total, split evenly between the two learned x4 stages.

## Verified x32 reference

The completed run in
`Benchmarks/CAVE_sequential_60_40/comparisons_original_x32/self_supervised_1000_zsl_cascade/ZSL`
uses 1,000 updates per scene (`500 + 500`) and reports over 13 scenes:

| PSNR | SAM | ERGAS | SSIM |
|---:|---:|---:|---:|
| 36.782 | 6.181 | 0.523 | 0.971 |
