# Masked Background Research Direction

This folder contains the next research direction for the TRI PCB X-ray shadow
removal project.

The main idea is different from the older decomposition model:

1. Detect real PCB structure first: solder balls, pads, square grids, traces.
2. Mask those features so they are not used as background.
3. Estimate the shadow/background field only from the remaining background pixels.
4. Inpaint and smooth that field.
5. Render both:
   - a natural restoration view
   - an aggressive inspection view

This is meant to prevent the shadow model from absorbing solder balls or square
pads into the estimated background.

## Scripts

```powershell
.\.venv312\Scripts\python.exe .\masked_background_research\masked_background_v1.py
```

V1 is the first hard-mask experiment. It is useful as a failure case: the binary
mask becomes too large and inpainting can create a contaminated background field.

```powershell
.\.venv312\Scripts\python.exe .\masked_background_research\masked_background_v2.py
```

V2 replaces the hard mask with soft background confidence weights:

```text
background = GaussianBlur(image * weight) / GaussianBlur(weight)
```

This is faster and less aggressive, but still needs stronger component-specific
feature masking/priors before it can match TRI-style results.

Default input:

```text
TRI_Dataset-20260511T190518Z-3-001\TRI_Dataset_Concised
```

Default V1 output:

```text
masked_background_research\result\masked_background_v1
```

Default V2 output:

```text
masked_background_research\result\masked_background_v2
```

Important V1 result folders:

- `masked_flat_natural`: first output to check for realistic shadow reduction
- `masked_flat_strong`: stronger version, may be better for J4 seams
- `inspection_masked`: TRI-style high-contrast inspection view
- `debug_structure_mask`: white pixels are protected PCB structures
- `debug_background_field`: estimated shadow/background field
- `comparison_sheets`: side-by-side review sheets

Important V2 result folders:

- `soft_flat_natural`: gentler natural restoration
- `soft_flat_strong`: stronger additive flattening
- `inspection_soft`: aggressive feature/inspection view
- `debug_background_weight`: bright pixels are trusted as background
- `debug_background_field`: estimated background/shadow field
- `comparison_sheets`: side-by-side review sheets
