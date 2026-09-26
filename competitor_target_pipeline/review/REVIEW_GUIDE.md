# Human review workflow

Eight candidates are prepared: two images each from J1, J4, U29 and U30.
They are previously viewed images, not an unseen test set. All annotations start
empty and all review states start pending. No defect or healthy region was
automatically asserted to be ground truth.

## Open the annotation window

From the repository root:

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\review_workbench.py annotate
```

1. Start with the Input view. Zoom and scroll to inspect the native pixels.
2. Choose `healthy`, enter a pad ID, and click vertices around an area TRI agrees
   is healthy solder. Right-click or choose Finish polygon to close the polygon.
3. For a confirmed defect in that pad, choose `defect` and use the SAME pad ID.
   Mark its outline. Dark spots are not automatically voids; leave uncertain
   regions unlabelled and pending for expert review.
4. Mark unwanted background with `background`. Use an ID such as `background_1`.
   Do not include actual traces or small components unless TRI confirms that
   those elements should be suppressed for this inspection task.
5. Optionally trace `pad_boundary` for visual inspection of proposed rectangles.
   This is currently an annotation aid, not an automatic boundary accuracy score.
6. Use V4 and V5 views to inspect the same coordinates. No image is registered,
   resized on disk, or independently normalized during annotation. Zoom changes
   display size only; polygon coordinates remain in source pixels.
7. Enter the reviewer name and choose Mark image reviewed after checking all
   polygons. This requires healthy and background polygons; each defect also
   needs a healthy reference with the same pad ID. Defect annotations are optional
   when none can be confirmed. Missing defect labels do not prove absence of defects.

Save draft preserves completed polygons. Finish an in-progress polygon before
navigating or closing. Edits invalidate review status. Preparing the set again
retains existing annotations and refuses changed source images.

## Produce reviewed metrics

```powershell
.\.venv312\Scripts\python.exe .\competitor_target_pipeline\review_workbench.py evaluate
```

Only reviewed entries with matching source SHA-256 and dimensions are evaluated.
Drafts do not produce reviewed metrics. Outputs appear beside this guide as
`reviewed_metrics.csv` and `evaluation_provenance.json`.

- Healthy standard deviation measures only manually selected healthy solder.
  Lower is desirable only when genuine structural and defect information remains.
- Signed healthy-minus-defect contrast preserves contrast polarity; do not
  maximize its absolute value without checking whether polarity changed.
- CNR divides absolute contrast by pooled within-region standard deviation.
  It is blank if variance is effectively zero. A larger CNR is useful but cannot
  validate defect geometry or guarantee downstream detection accuracy.
- Background spread and high-frequency RMS describe manually marked background.
  They do not justify deleting real structural details.

These are fixed-polygon measurements. Actual defect area and boundary preservation
still need native-resolution visual review or a separately validated detector.

`matched_crops` contains 24 source-coordinate crop sheets for input, V4, and V5.
Each crop uses the same pixel coordinates and no per-crop contrast normalization.
These are inspection windows, not labelled defects.

## What remains human-dependent

TRI or a qualified reviewer must identify healthy solder, confirmed defects, and
unwanted background. The software prepares and records this review; it cannot
certify its own guesses as ground truth. Until then V5 is an experimental output,
not a validated replacement for V4.
