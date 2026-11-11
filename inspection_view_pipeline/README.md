Inspection-View PCB X-ray Enhancement
=====================================

This folder contains inspection-view pipelines for TRI PCB X-ray images.

V1
--

V1 is the first aggressive Mode 2 pipeline:

    aggressive background flattening + feature enhancement

Goal:

- dark / flattened background
- bright solder balls, vias, pads, and traces
- reduced visible shadow and background mottle
- not necessarily a natural-looking grayscale restoration

Run from the repository root:

    .\.venv312\Scripts\python.exe .\inspection_view_pipeline\inspection_view_v1.py

Output:

    inspection_view_pipeline/result/inspection_view_v1/

V2
--

V2 responds to TRI feedback:

- solder regions should have more uniform grayscale after shadow removal
- V1 can saturate solder regions, causing loss of contrast
- boosted texture can look like false solder fragments

V2 outputs:

- `inspection_clean`: dark inspection view with reduced saturation
- `balanced_gray`: less aggressive grayscale view for shadow-free review
- `feature_mask_clean`: debug mask for enhanced features/fragments
- `background_field`: estimated broad background/shadow field
- `flattened_residual`: flattened image before dark rendering

Run from the repository root:

    .\.venv312\Scripts\python.exe .\inspection_view_pipeline\inspection_view_v2.py

Output:

    inspection_view_pipeline/result/inspection_view_v2/

V3
--

V3 is defect-focused. It keeps V2's shadow flattening direction but avoids
using only a binary-looking dark image for review.

V3 outputs:

- `inspection_clean`: dark inspection view with solder/pad features emphasized
- `balanced_gray_clean`: first output to inspect for defect detection work
- `feature_mask`: debug feature mask used by the defect-focused renderer
- `background_field`: estimated broad background/shadow field

Run from the repository root:

    .\.venv312\Scripts\python.exe .\inspection_view_pipeline\inspection_view_v3.py

Output:

    inspection_view_pipeline/result/inspection_view_v3/

Input
-----

Both scripts default to:

    TRI_Dataset-20260511T190518Z-3-001/TRI_Dataset_Concised
