# Replacement checklist

Target directory: `HIFtool-main/Networks/CaFormer`

| File | Role |
|---|---|
| `net.py` | checkpoint-compatible CaFormer and resolution-safe forward path |
| `Config.py` | legacy arguments plus explicit supervised/blind options |
| `__init__.py` | public package exports |
| `inference.py` | aligned overlapping tiled inference |
| `factory.py` | model/checkpoint constructors |
| `blind.py` | uHNTC operator estimation and observation-supervised adaptation |
| `README.md` | usage, protocol, and reproducibility defaults |
| `MANIFEST.json` | machine-readable package description |
| `tests/smoke_test.py` | scales 2/4/8/16/32 and tiled inference |
| `tests/check_checkpoint.py` | strict released-checkpoint loading |
| `tests/check_numerical_compatibility.py` | comparison against installed CaFormer |
| `tests/blind_smoke_test.py` | end-to-end uHNTC blind path |

Before replacing the live package:

1. Wait until running CaFormer processes finish.
2. Back up the current `Networks/CaFormer`.
3. Copy this directory as the new `Networks/CaFormer`.
4. Keep `Networks/FeafusFormer` because `blind.py` imports its uHNTC modules.
5. Run all four tests in `tests`.
6. Keep the existing `Networks/__init__.py` for supervised `CaFormer_3`.
7. Invoke blind mode explicitly through `build_blind_caformer`; do not relabel
   ordinary checkpoint inference as blind fusion.
