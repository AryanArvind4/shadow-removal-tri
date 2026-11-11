"""
Quick test: run only aggressive_local with hybrid post-processing
and save results to results/aggressive_local_hybrid/ for comparison.
"""

import csv
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classical_shadow_removal_v2 import (
    aggressive_local_combo,
    component_from_name,
    edge_energy_ratio,
    image_paths,
    lr_diff,
    metric_row,
    params_for,
    quadrant_std,
    read_gray,
    row_band_std,
    save_gray,
)

BASE       = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE, "TRI_Dataset-20260511T190518Z-3-001", "TRI_Dataset_Concised")
OUTPUT_DIR = os.path.join(BASE, "results", "aggressive_local_hybrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

paths = image_paths(INPUT_DIR)
print(f"Running aggressive_local + hybrid post-process on {len(paths)} images...")
print(f"Output: {OUTPUT_DIR}\n")

rows = []
for path in tqdm(paths, desc="aggressive_local_hybrid"):
    filename = os.path.basename(path)
    params   = params_for(filename)
    original = read_gray(path)

    corrected = aggressive_local_combo(original, params)
    save_gray(os.path.join(OUTPUT_DIR, filename), corrected)
    rows.append(metric_row(filename, "aggressive_local_hybrid", original, corrected))

# Save metrics
csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Print summary
print("\n" + "=" * 60)
print("RESULTS — aggressive_local + hybrid post-process")
print("=" * 60)
print(f"{'Component':>10} {'LR_after':>10} {'Quad_red%':>10} {'Band_after':>11} {'Edge_ratio':>11}")
print("-" * 55)
for comp in ["J1", "J4", "U29", "U30"]:
    subset = [r for r in rows if r["component"] == comp]
    lr   = np.mean([float(r["lr_diff_after"])       for r in subset])
    qr   = np.mean([float(r["quad_reduction"])       for r in subset]) * 100
    band = np.mean([float(r["row_band_std_after"])   for r in subset])
    edge = np.mean([float(r["edge_energy_ratio"])    for r in subset])
    print(f"{comp:>10} {lr:>10.4f} {qr:>10.1f}% {band:>11.4f} {edge:>11.4f}")

lr   = np.mean([float(r["lr_diff_after"])     for r in rows])
qr   = np.mean([float(r["quad_reduction"])     for r in rows]) * 100
band = np.mean([float(r["row_band_std_after"]) for r in rows])
edge = np.mean([float(r["edge_energy_ratio"])  for r in rows])
print("-" * 55)
print(f"{'AVERAGE':>10} {lr:>10.4f} {qr:>10.1f}% {band:>11.4f} {edge:>11.4f}")

print(f"\nImages saved : {OUTPUT_DIR}")
print(f"Metrics saved: {csv_path}")

# Compare against previous optimized results
print("\n" + "=" * 60)
print("COMPARISON vs optimized (no post-process)")
print("=" * 60)
prev_path = os.path.join(BASE, "results", "aggressive_local_optimized", "metrics.csv")
if os.path.exists(prev_path):
    with open(prev_path) as f:
        prev = list(csv.DictReader(f))
    metrics = ["lr_diff_after", "quad_reduction", "row_band_std_after", "edge_energy_ratio"]
    labels  = ["LR_after", "Quad_red", "Band_after", "Edge_ratio"]
    print(f"{'Metric':>15} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 48)
    for m, label in zip(metrics, labels):
        before = np.mean([float(r[m]) for r in prev])
        after  = np.mean([float(r[m]) for r in rows])
        delta  = after - before
        sign   = "+" if delta > 0 else ""
        print(f"{label:>15} {before:>10.4f} {after:>10.4f} {sign}{delta:>9.4f}")
else:
    print("(optimized baseline not found for comparison)")
