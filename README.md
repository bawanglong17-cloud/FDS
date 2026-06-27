# FDS-MedSeg

This repository contains the core PyTorch implementation of the **Feature Decoupling and Dual-Branch Separation (FDS)** module for medical image segmentation.

The code is organized from the original single-file implementation into independent modules so that each component can be inspected, reused, and extended more easily.

## Main components

```text
fds_medseg/
  models/
    chebyshev.py      # Chebyshev structure-detail split
    layers.py         # Basic Conv/Down/Up/U-Net layers
    attention.py      # Channel attention, ECA, CBAM
    wavkan.py         # Grouped Dense WAV-KAN and bottleneck WAV-KAN branch
    fds_block.py      # Complete FDS block
    unet.py           # Baseline U-Net and FDS-UNet
    swin_adapter.py   # FDS adapter for Swin-style bottleneck tokens
    builder.py        # build_model() helper
examples/
  quick_check.py      # Minimal forward-pass test
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick check

Run the following command from the repository root:

```bash
python examples/quick_check.py
```

Expected output shape examples:

```text
FDS-UNet output: (1, 1, 64, 64)
FDS block output: (1, 64, 16, 16)
Swin adapter output: (1, 256, 64)
```

## Basic usage

```python
import torch
from fds_medseg.models import build_model

model = build_model(
    "fds_unet",
    in_channels=3,
    n_classes=1,
    base_ch=64,
    kan_reduction=4,
    kan_groups=8,
    kan_spatial_chunk=512,
    fusion_attention="ours",  # choices: ours, eca, cbam
)

x = torch.randn(1, 3, 384, 384)
logits = model(x)
print(logits.shape)
```

## FDS block only

```python
import torch
from fds_medseg.models import FDSBlockLite

fds = FDSBlockLite(channels=512, kan_groups=8)
feature = torch.randn(1, 512, 24, 24)
out = fds(feature)
```

## Swin/Transformer bottleneck adapter

For Swin-style token features `[B, L, C]`, provide the spatial size `h` and `w`:

```python
import torch
from fds_medseg.models import FDSForSwinBottleneckLite

adapter = FDSForSwinBottleneckLite(channels=768, kan_groups=8)
tokens = torch.randn(1, 24 * 24, 768)
out = adapter(tokens, h=24, w=24)
```

## Notes

- `UNet` is the baseline model without FDS.
- `FDSUNetLite` inserts the FDS block into the U-Net bottleneck layer.
- `FDSBlockLite` contains Chebyshev splitting, CNN detail branch, bottleneck grouped WAV-KAN structure branch, 1x1 fusion, attention recalibration, and residual connection.
- Large datasets, trained weights, logs, and checkpoints are intentionally excluded from this repository.

## Citation

If this code is useful for your research, please cite the corresponding paper when available.
