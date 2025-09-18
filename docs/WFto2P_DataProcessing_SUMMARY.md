# Real Microscopy Data Processing Summary

## Overview
Successfully processed your real microscopy data into a format ready for PKL-guided diffusion training. The data includes WF/2P paired images and bead calibration data.

## Data Processing Results

### 1. WF/2P Paired Data
**Source Files:**
- `../data_wf_tp/wf.tif`: Wide-field microscopy data (51 frames, 2048×2048, uint16)
- `../data_wf_tp/tp_reg.tif`: Two-photon registered data (51 frames, 2048×2048, uint16)

**Processing:**
- Extracted **11,475 patch pairs** (256×256 pixels each, 225 patches per frame)
- Normalized to uint8 with 99.5% percentile clipping
- Applied intensity thresholding to avoid dark background patches
- Split into train/val/test: **9,000 / 1,125 / 1,350 pairs**

**Output Structure:**
```
data/real_microscopy/
├── splits/
│   ├── train/
│   │   ├── wf/      # 9,000 WF patches
│   │   └── 2p/      # 9,000 2P patches
│   ├── val/
│   │   ├── wf/      # 1,125 WF patches  
│   │   └── 2p/      # 1,125 2P patches
│   └── test/
│       ├── wf/      # 1,350 WF patches
│       └── 2p/      # 1,350 2P patches
```

### 2. Bead Calibration Data
**Source Files:**
- `../data_wf_tp/beads/1-2um_bead_1_cover_glass_Zstep_0.1um.tif`: Bead without AO (241 z-slices, 256×256)
- `../data_wf_tp/beads/2-2um_bead_1_cover_glass_Zstep_0.1um_after_AO.tif`: Bead with AO (241 z-slices, 256×256)

**Processing:**
- Created max-intensity projections for both conditions
- Saved middle z-slices for reference
- Extracted z-slice series (every 10th slice) for analysis
- Z-range: 0.0 - 24.0 μm (0.1 μm steps)

**Output Structure:**
```
data/real_microscopy/beads/
├── bead_no_AO_projection.png
├── bead_with_AO_projection.png
├── bead_no_AO_middle_slice.png
├── bead_with_AO_middle_slice.png
├── bead_no_AO_slices/     # 25 z-slices
├── bead_with_AO_slices/   # 25 z-slices
└── metadata.txt
```

## Dataset Classes Created

### 1. RealPairsDataset
- **Location**: `pkl_dg/data/real_pairs.py`
- **Purpose**: Load WF/2P paired training data
- **Features**:
  - Automatic train/val/test split loading
  - Intensity normalization to [-1, 1] range
  - Compatible with existing transform pipeline
  - Returns (clean_2P, noisy_WF) pairs for diffusion training

### 2. BeadDataset  
- **Location**: `pkl_dg/data/real_pairs.py`
- **Purpose**: Load bead calibration data
- **Features**:
  - Support for AO/no-AO conditions
  - Z-projection and slice loading
  - Useful for PSF analysis and calibration

## Training Configuration

### New Configuration Files
1. **`configs/data/real_pairs.yaml`**: Data configuration for real microscopy pairs
2. **`configs/config_real.yaml`**: Complete training configuration for real data

### Training Script
- **`scripts/train_real_data.py`**: Dedicated training script for real data
- **Features**:
  - Direct loss computation (bypasses Lightning trainer issues)
  - Mixed precision training support
  - Model checkpointing and validation
  - TensorBoard and W&B logging

## Usage Instructions

### 1. Train on Real Data
```bash
# Basic training
python scripts/train_real_data.py

# Custom settings
python scripts/train_real_data.py \
  training.max_epochs=100 \
  training.batch_size=8 \
  experiment.name=my_real_experiment

# Disable W&B logging
python scripts/train_real_data.py wandb.mode=disabled
```

### 2. Visualize Data
```bash
# Generate visualization plots
python scripts/visualize_real_data.py

# Custom number of samples
python scripts/visualize_real_data.py --num-samples 6
```

### 3. Load Data Programmatically
```python
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel

# Create transform
transform = IntensityToModel(minIntensity=0, maxIntensity=255)

# Load training data
train_dataset = RealPairsDataset(
    data_dir='data/real_microscopy',
    split='train',
    transform=transform,
    image_size=256
)

# Get a sample
clean_2p, noisy_wf = train_dataset[0]
```

## Data Statistics

- **Total WF/2P pairs**: 11,475
- **Training pairs**: 9,000 (40 frames × 225 patches)
- **Validation pairs**: 1,125 (5 frames × 225 patches)  
- **Test pairs**: 1,350 (6 frames × 225 patches)
- **Image size**: 256×256 pixels
- **Data type**: uint8, normalized to [-1, 1] for training
- **Bead z-stacks**: 241 slices each (0.1 μm steps)

## Next Steps for ICLR Research

With your real data now processed and ready:

1. **Complete DDIM Sampler** (Day 1 priority from research plan)
2. **Train diffusion model** on real data using PKL guidance
3. **Compare with baselines** (L2 guidance, Richardson-Lucy)
4. **Generate quantitative results** (PSNR, SSIM, FRC metrics)
5. **Create visual comparisons** for paper figures

The processed dataset provides a solid foundation for your 18-day sprint to ICLR 2025 submission. You now have **~9.5K high-quality WF/2P pairs** from real microscopy data, which is more than sufficient for demonstrating PKL guidance effectiveness.

## Files Created

### Scripts
- `scripts/process_microscopy_data.py`: Data processing pipeline
- `scripts/train_real_data.py`: Real data training script  
- `scripts/visualize_real_data.py`: Data visualization script

### Dataset Classes
- `pkl_dg/data/real_pairs.py`: RealPairsDataset and BeadDataset

### Configurations
- `configs/data/real_pairs.yaml`: Real data configuration
- `configs/config_real.yaml`: Complete real data training config

### Outputs
- `data/real_microscopy/`: Processed data directory
- `data/real_microscopy/*.png`: Visualization plots
- Metadata files with processing details
