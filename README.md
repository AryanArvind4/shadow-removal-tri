# TRI PCB X-ray Inspection-View Pipeline

Classical image processing for shadow correction and grayscale inspection of PCB
X-ray images. **V5a is the selected inspection-view pipeline** used for the latest
progress report and new TRI image batches. It is not a trained neural network or
an automatic defect classifier. The original classical shadow-removal code is
retained as an earlier baseline.

## Setup

Tested with Python 3.12 on Windows. Processing uses NumPy, SciPy, Pillow and
OpenCV on the CPU; no NVIDIA GPU or model checkpoint is required.

```powershell
python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

Input images and generated results are kept locally, not supplied with the code.
Ignore rules exclude new images, datasets, archives, checkpoints, virtual
environments and presentation files. Images already committed in older revisions
remain in Git history; this update does not remove or re-upload them.

## Run V5a

From the repository root, provide a directory of **8-bit grayscale** PNG/JPG/JPEG
images. RGB and higher-bit-depth inputs require an explicit preprocessing decision.
Use a new output directory for each run to avoid overwriting earlier results.

```powershell
.\.venv312\Scripts\python.exe competitor_target_pipeline\competitor_target_v5a.py --input "path\to\grayscale_images" --output "competitor_target_pipeline\result\my_v5a_run"
```

Optionally append `--files image1.png image2.png` to select exact filenames. The
default input, when omitted, is the local
`TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset_Concised` folder; it is not included
in a fresh clone.

### G2/G3 and Other BMP Batches

The BMP adapter converts 8-bit grayscale BMPs to pixel-identical PNGs, runs the
unchanged selected V5a, and verifies output dimensions and unchanged originals.
It refuses an existing output directory. The bundled `v5a_source_manifest.json`
checks the three processing modules without requiring an old results folder.

```powershell
.\.venv312\Scripts\python.exe competitor_target_pipeline\run_v5a_bmp_batch.py --input "competitor_target_pipeline\G2G3" --output "competitor_target_pipeline\G2G3\v5a_results"
```

For another batch, change `--input` and `--output`. Filenames do not select a
component-specific processing mode. `run_tri_new_images.py` is a separate,
dataset-specific adapter for `TRI_New_Image/image001.png` and `image002.png`; it
converts those RGB screenshots to grayscale without removing their overlays.

## How V5a Works

1. Estimate image noise and apply guarded, blended non-local-means denoising.
2. Estimate feature scale and conditionally correct supported row-band patterns.
3. Reduce broad brightness variation with a bounded background correction.
4. Apply limited local gain and a continuous grayscale display curve.
5. Look for a repeated rectangular-pad layout using opposing edges and spacing.
6. Only for supported layouts, apply bounded per-pad brightness/slope correction
   and partial attenuation of repeated small bright spots outside guarded pads.

If rectangular-layout evidence is insufficient, V5a retains the automatic V5
baseline output exactly. It does not force a rectangular grid onto every image.
No manual annotations are needed at runtime. Final images retain native input
dimensions; reduced-resolution analysis is used for some correction estimates.

## Which Outputs to Review

| Folder / file | Purpose |
| --- | --- |
| `reference_gray/` | Main native-resolution enhanced PNGs |
| `input_output/` | Input and V5a side-by-side comparisons |
| `comparison_sheets/` | Input, automatic V5 baseline and V5a |
| `pad_only_gray/` | V5a without repeated-dot attenuation |
| `selection_overlay/` | Proposed pad boundaries and attenuated spots |
| `pad_mask/`, `dot_attenuation_mask/`, `pad_correction/` | Diagnostic maps, not inspection results |
| `numeric/` | Pre-tone corrected arrays and region labels |
| `per_region_metrics.csv` | Measurements within automatic proposed regions |
| `run.json` | Input/code hashes and processing decisions |

The BMP adapter also saves `prepared_inputs/` and `source_preparation.json` with
conversion checks, output validation and end-to-end execution time.

## Evaluation and Limitations

- `interior_std` and `interior_iqr` measure intensity variation inside proposed
  pad interiors. Lower is not necessarily better: defects and genuine structure
  also contribute to variation, and contrast enhancement changes the scale.
- `input_correlation` measures association with the original pixels. High values
  do not certify defect preservation.
- `clipping_fraction` measures near-black/near-white pixels. Low clipping avoids
  one failure mode but does not prove good inspection quality.
- Regions are unreviewed proposals, not confirmed healthy solder. When no layout
  is accepted, the per-region CSV can contain only its header.
- Background grain, overlapping structures and uneven visibility can remain.
  Denoising, pad correction and display choices can affect faint defects.
- Competitor photographs are qualitative references, not matched ground truth.
  Matching board, acquisition and reconstruction depth are not confirmed.
  Enhancement cannot reliably recover cavities absent from the source data.

Next validation priorities are reviewed healthy/defect/background regions,
matched acquisition information, and testing across unseen TRI components.
Do not claim competitor parity or defect-detection accuracy from images alone.

## Verification and Timing

Run the synthetic regression suite (no TRI image files required):

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s competitor_target_pipeline -p "test_competitor_target_v*.py"
```

The latest local G2/G3 run processed 13 images (9 G2, 4 G3) in approximately
386 seconds including subprocess startup and exports. All 13 used the automatic
fallback because the rectangular-pad branch was not supported. This is a
single-machine measurement, not a general performance guarantee.

`benchmark_v5a.py` reproduces timing against a **local** selected run and verifies
pixel equality with its outputs. It requires the original images and
`competitor_target_pipeline/result/competitor_target_v5a/run.json`, plus its saved
`reference_gray` outputs; these are intentionally not uploaded.

## Repository Guide

- [V5a and earlier pipeline details](competitor_target_pipeline/README.md)
- `competitor_target_pipeline/competitor_target_v5a.py`: selected V5a entry point.
- `competitor_target_pipeline/automatic_v5.py`: automatic baseline processing.
- `competitor_target_pipeline/competitor_target_v3.py`: shared image/metric helpers.
- `competitor_target_pipeline/competitor_target_v2.py`, `competitor_target_v4.py`
  and `competitor_target_v5.py`: earlier methods and baseline comparisons.
- `competitor_target_pipeline/evaluate_reviewed_regions.py` and
  `review_workbench.py`: optional human-reviewed evaluation tools.
- `classical_shadow_removal_v2.py`: earlier classical shadow-removal pipeline.
- `results/classical_v2/metrics.csv`: historical classical metrics, not V5a scores.
