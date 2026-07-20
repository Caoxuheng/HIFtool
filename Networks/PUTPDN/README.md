# PUT-PDN adapter

The official PUT implementation is a hyperspectral **pansharpening** network.
This HIFTool baseline widens every PAN-dependent operation to preserve the
three native HR-MSI channels (`3 -> 6 -> 12` through the encoder) instead of
collapsing MSI into a synthetic PAN image.
The architecture otherwise retains PUT's unfolding stages and learnable spatial/
spectral degradation operators. Its training objective is `L1(HR-HSI) + 0.1
L1(reconstructed HR-MSI, HR-MSI) + 0.1 L1(reconstructed LR-HSI, LR-HSI)`.
Results must be reported as **PUT-PDN-MSI**, not as the original PUT-PDN paper
protocol.

The official source is expected at `external/comparison_official/PUT-PDN` and is
loaded by `model_generator('PUTPDN')`.

Run the CAVE x32 protocol with:

```powershell
python comparison_benchmarks/run_putpdn.py --max-steps 10000
```
