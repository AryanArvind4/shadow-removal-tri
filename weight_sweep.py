"""
Weight Sweep for aggressive_local_combo
========================================
Finds the optimal (w_mean, w_zscore, w_retinex) weights for each component
group by grid-searching all valid combinations with step=0.10.

Strategy:
  - Compute the 3 sub-method outputs ONCE per image (slow part, cached)
  - Blend with each weight combination and re-run smooth_profile_correction
  - Score each combo using a component-aware composite score
  - Report best weights for J1/J4 and U29/U30 separately

Run:
    python weight_sweep.py

Output:
    Prints best weights and metric improvement for each component group.
    Also saves results/weight_sweep/metrics.csv for full audit.
"""

import csv
import itertools
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classical_shadow_removal_v2 import (
    component_from_name,
    edge_energy_ratio,
    image_paths,
    local_mean_equalized,
    local_zscore_equalized,
    lr_diff,
    mean_std_match,
    multi_scale_retinex_like,
    params_for,
    quadrant_std,
    read_gray,
    row_band_std,
    smooth_profile_correction,
)


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(
    BASE,
    "TRI_Dataset-20260511T190518Z-3-001",
    "TRI_Dataset_Concised",
)
OUTPUT_DIR = os.path.join(BASE, "results", "weight_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Weight grid ───────────────────────────────────────────────────────────────

STEP = 0.10

def weight_grid(step: float):
    """All (w1, w2, w3) with w1+w2+w3=1, each >= step, rounded to 2dp."""
    combos = []
    vals = [round(v, 2) for v in np.arange(step, 1.0, step)]
    for w1 in vals:
        for w2 in vals:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= step:
                combos.append((w1, w2, w3))
    return combos


WEIGHTS = weight_grid(STEP)
print(f"Weight combinations to test: {len(WEIGHTS)}")
print(f"e.g. first few: {WEIGHTS[:5]}")


# ── Scoring ───────────────────────────────────────────────────────────────────

def composite_score(
    lrd: float,
    band: float,
    edge: float,
    component: str,
) -> float:
    """
    Lower = better.
    Edge penalty kicks in hard if we drop below 1.0 (structure loss).

    J1/J4: large L/R gradient is the dominant problem → weight lr_diff more
    U29/U30: banding is the dominant problem → weight band more
    """
    edge_penalty = max(0.0, 1.0 - edge) * 15.0

    if component in ("J1", "J4"):
        return lrd * 2.0 + band * 0.4 + edge_penalty
    else:
        return lrd * 0.4 + band * 2.0 + edge_penalty


# ── Main sweep ────────────────────────────────────────────────────────────────

paths = image_paths(INPUT_DIR)
if not paths:
    raise FileNotFoundError(f"No images found in {INPUT_DIR}")

print(f"\nImages: {len(paths)}")
print("Step 1: Computing sub-method outputs (cached per image)...\n")

# Collect per-component best results
# Structure: {component: {combo: [scores]}}
component_scores: dict = {}  # component -> list of (score, w1, w2, w3)
all_rows = []

CURRENT_WEIGHTS = (0.45, 0.35, 0.20)  # baseline to compare against

for path in tqdm(paths, desc="Images"):
    filename = os.path.basename(path)
    component = component_from_name(filename)
    params = params_for(filename)
    original = read_gray(path)

    # ── Cache the 3 sub-method outputs (slow, done once per image) ────────────
    mean_eq  = local_mean_equalized(original, params).astype(np.float32)
    z_eq     = local_zscore_equalized(original, params).astype(np.float32)
    ret      = multi_scale_retinex_like(original, params).astype(np.float32)

    if component not in component_scores:
        component_scores[component] = {}

    # ── Sweep all weight combinations ─────────────────────────────────────────
    for w1, w2, w3 in WEIGHTS:
        blended = w1 * mean_eq + w2 * z_eq + w3 * ret
        blended = smooth_profile_correction(
            blended, params, correct_rows=True, correct_cols=True
        )
        corrected = mean_std_match(blended, original)

        lrd  = lr_diff(corrected)
        band = row_band_std(corrected)
        qstd_before = quadrant_std(original)
        qstd_after  = quadrant_std(corrected)
        q_red = (qstd_before - qstd_after) / (qstd_before + 1e-8)
        edge = edge_energy_ratio(original, corrected)
        score = composite_score(lrd, band, edge, component)

        combo_key = (w1, w2, w3)
        if combo_key not in component_scores[component]:
            component_scores[component][combo_key] = []
        component_scores[component][combo_key].append(score)

        all_rows.append({
            "filename"        : filename,
            "component"       : component,
            "w_mean"          : w1,
            "w_zscore"        : w2,
            "w_retinex"       : w3,
            "lr_diff_after"   : round(lrd,  4),
            "row_band_after"  : round(band, 4),
            "quad_reduction"  : round(q_red, 4),
            "edge_ratio"      : round(edge, 4),
            "score"           : round(score, 4),
        })


# ── Save full CSV ─────────────────────────────────────────────────────────────

csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nFull results saved: {csv_path}")


# ── Find best weights per component ──────────────────────────────────────────

print("\n" + "=" * 65)
print("WEIGHT SWEEP RESULTS")
print("=" * 65)

# Group J1/J4 and U29/U30 together (same params, treat as one group)
GROUPS = {
    "J1/J4" : ["J1", "J4"],
    "U29/U30": ["U29", "U30"],
}

best_per_group = {}

for group_name, components in GROUPS.items():
    # Aggregate scores across all components in this group
    group_combos: dict = {}
    for comp in components:
        if comp not in component_scores:
            continue
        for combo, scores in component_scores[comp].items():
            if combo not in group_combos:
                group_combos[combo] = []
            group_combos[combo].extend(scores)

    if not group_combos:
        continue

    # Best = lowest mean score
    sorted_combos = sorted(group_combos.items(), key=lambda x: np.mean(x[1]))
    best_combo, best_scores = sorted_combos[0]
    best_per_group[group_name] = best_combo

    # Also get baseline score
    baseline_scores = group_combos.get(CURRENT_WEIGHTS, [])
    baseline_mean = np.mean(baseline_scores) if baseline_scores else float("nan")
    best_mean = np.mean(best_scores)
    improvement = (baseline_mean - best_mean) / (baseline_mean + 1e-8) * 100

    print(f"\n{group_name}")
    print(f"  Current  (0.45 / 0.35 / 0.20): score = {baseline_mean:.4f}")
    print(f"  Best     {best_combo}:  score = {best_mean:.4f}  ({improvement:+.1f}%)")
    print(f"  → w_mean={best_combo[0]:.2f}, w_zscore={best_combo[1]:.2f}, w_retinex={best_combo[2]:.2f}")

    # Show top 5
    print(f"\n  Top 5 combos for {group_name}:")
    print(f"  {'w_mean':>7} {'w_zscore':>9} {'w_retinex':>10} {'score':>8}")
    for combo, scores in sorted_combos[:5]:
        print(f"  {combo[0]:>7.2f} {combo[1]:>9.2f} {combo[2]:>10.2f} {np.mean(scores):>8.4f}")


# ── Print update instructions ──────────────────────────────────────────────

print("\n" + "=" * 65)
print("UPDATE classical_shadow_removal_v2.py line ~457:")
print("=" * 65)
j1j4  = best_per_group.get("J1/J4",  CURRENT_WEIGHTS)
u29u30= best_per_group.get("U29/U30", CURRENT_WEIGHTS)

print(f"""
    # Component-aware weights (optimized by weight_sweep.py)
    if component in ("J1", "J4"):
        w_mean, w_zscore, w_retinex = {j1j4[0]:.2f}, {j1j4[1]:.2f}, {j1j4[2]:.2f}
    else:
        w_mean, w_zscore, w_retinex = {u29u30[0]:.2f}, {u29u30[1]:.2f}, {u29u30[2]:.2f}
    corrected = w_mean * mean_eq + w_zscore * z_eq + w_retinex * ret
""")

print("Done. Review results/weight_sweep/metrics.csv for full audit.")
