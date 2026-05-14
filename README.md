# Shadow Removal for TRI Industrial PCB X-ray Images

Classical signal processing pipeline for removing metal-sheet illumination shadows from TRI PCB X-ray images. Handles two artifact types:
- **Horizontal banding** — per-row illumination variation
- **Large-scale L/R gradient** — smooth left-right brightness drift

## Method

`aggressive_local` — a weighted ensemble of local mean equalization (45%), local z-score normalization (35%), and multi-scale retinex (20%), with structure-masked background estimation and component-aware parameters.

Evaluated on 32 images across 4 component types (J1, J4, U29, U30):
- **87% average reduction** in quadrant brightness variance
- **69% average reduction** in row banding
- **Edge energy ratio: 1.71** — solder ball structure preserved and enhanced

## Requirements

```
pip install -r requirements.txt
```

Requires Python 3.8+. No GPU needed.

## Usage

### Process a full directory

```python
from classical_shadow_removal_v2 import process_directory

process_directory(
    input_dir="path/to/input/images",
    output_dir="path/to/output",
    component="J1",          # component type: J1, J4, U29, U30
    save_comparison=True     # saves before|after side-by-side images
)
```

### Run the script directly

Edit the `__main__` block at the bottom of `classical_shadow_removal_v2.py` to set your input/output paths, then:

```
python classical_shadow_removal_v2.py
```

## Component Parameter Reference

| Component | Image Size | sigma_frac | correction_strength |
|---|---|---|---|
| J1 | ~2148×2304 | 0.070 | 1.65 |
| J4 | ~2148×2304 | 0.070 | 1.65 |
| U29 | Small | 0.090 | 1.85 |
| U30 | Small | 0.090 | 1.85 |

## Results

Quantitative metrics for all 32 test images are in [`results/metrics.csv`](results/metrics.csv).

## Output Files

- `output_dir/<filename>.jpg` — corrected images
- `output_dir/_comparisons/<filename>_compare.png` — side-by-side before/after
- `output_dir/_metrics.csv` — per-image metrics

## Citation

Research conducted at NTU in collaboration with TRI (Taiwan Research Institute).
