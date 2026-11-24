# First V5c review: 11 November 2026

## Recommendation

Use this as a **U29-focused candidate for the 19 November discussion**, not as a
blanket replacement for V5b. The boundary treatment is improved in some inspected
crops, but real-defect preservation is not yet validated. Retain V5b and the
previous targeted outputs for the other components.

## Start here

- [U29_032 full comparison](results_v5c/U29/comparison_sheets/U29_032.png)
- [U29_032 row 2, column 3: pixel-replicated comparison](results_v5c/U29/pixel_crops/U29_032_pad_12.png)
- [U29_032 row 2, column 5: unresolved boundary example](results_v5c/U29/pixel_crops/U29_032_pad_14.png)
- [U29_034 full comparison](results_v5c/U29/comparison_sheets/U29_034.png)
- [U76 z201 comparison](results_v5c/TRI_1008/comparison_sheets/Component_U76_1_2048_708_636_201_z201.png)
- [Round-ball center comparison](results_v5c/round_ball_review/review_crops/Component_G2_1_1152_2472_2140_165_z0060_center.png)

Every comparison includes the original input. Full-resolution final images are in
each dataset's `reference_gray/` and `enhanced/` folders. Pixel crops replicate
each original pixel 5x without smoothing; ordinary comparison sheets use Lanczos.

## What the comparisons show

**U29:** The inspected lower-edge dark feature in row 2, column 3 is separated more
clearly from the outline in V5c. V5b cut that area into the boundary. V5c also
retains more source texture and a softer outline. However, several other second-row
features remain notch-like or ambiguous. Some dark regions become more conspicuous;
that is not proof that they are physical voids rather than overlapping structure.
Do not describe every notch as fixed, or the shape as reconstructed.

**U76 z141:** The grid branch runs successfully and the inspected comparison shows
more internal dark-feature detail in some pads. Boundary ambiguity remains.

**U76 z201:** The conservative selection leaves the previously problematic lower
background unchanged in the inspected comparison. It also skips irregular plate
regions, so their contrast is lower than in the targeted V5b result. The separate
`targeted_comparisons/` image makes this trade-off visible.

**Round balls:** Many accepted balls remain enhanced, but stricter selection misses
some weak or irregular balls that V5b enhanced. This is a coverage regression;
retain the established V5b output for presentation unless this trade-off is agreed.

**TRI_new:** Only 5/121 and 1/145 proposals are accepted. These already contrasted,
low-resolution inputs therefore change little. Retain the previous targeted V5b
outputs; do not present V5c here as an improvement.

## Completed checks

- All 14 originals processed: eight U29, three U76, two TRI_new, one round-ball.
- All eight U29 images accepted 72 pad proposals each. This is proposal coverage,
  **not** a measured detection accuracy or a confirmed count of healthy pads.
- Seventeen focused synthetic regression tests passed.
- All 14 recomputed frozen V5b outputs exactly match the saved automatic outputs.
- All 14 saved V5c images retain native dimensions and original pixels outside the
  enhancement support, with no added grayscale darkening.
- SHA-256 checks confirm the copied inputs and retained V5b files are unchanged.
- Complete export took 358.0 seconds, including comparisons, all-pad crops and
  diagnostics. This is not the standalone deployment runtime of V5c.

See `results_v5c/run.json`, `audit.json` and `metrics.csv` for the recorded checks.
Exact preservation outside the mask does not prove that all background was excluded
from the mask. Synthetic tests do not certify every real defect.

## Next review questions

1. Does the professor agree that the retained source-visible rims improve the
   U29 edge-void presentation without obscuring genuine open defects?
2. Which still-ambiguous second-row features are confirmed closed voids in a
   matching acquisition, and which are genuine boundary openings?
3. Can a small reviewed set distinguish background patches from irregular plates
   and weak balls before relaxing the conservative generic detector?

No source-informed algorithm can guarantee the competitor's closed-void geometry
when matched board, acquisition depth and source information are unconfirmed.
