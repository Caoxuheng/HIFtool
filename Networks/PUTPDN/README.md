# PUT-PDN adapter

The official PUT implementation is a hyperspectral **pansharpening** network:
it requires LR-HSI and a one-channel HR-PAN image. HIFTool supplies LR-HSI and
three-channel HR-MSI, so this adapter deterministically constructs `PAN = mean(MSI)`.
The PUT architecture, supervision objective and learnable degradation operators are
otherwise unchanged. Results under this adapter must be reported as **MSI-to-PAN proxy**
adaptation, not as the original PUT-PDN paper protocol.

The official source is expected at `external/comparison_official/PUT-PDN` and is
loaded by `model_generator('PUTPDN')`.
