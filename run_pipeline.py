"""
MASTER RUN SCRIPT — SDEdit PCB X-ray Shadow Removal
====================================================

Run this file to execute the full pipeline stage by stage.
Each stage can also be run independently.

STAGE 1 : Train diffusion prior on clean X-rays      (run once)
STAGE 2 : Run SDEdit inference on shadowed images    (run per dataset)
STAGE 3 : Evaluate results                           (run after Stage 2)

FOLDER STRUCTURE EXPECTED:
    data/
      clean/          ← shadow-free TRI X-rays   (for Stage 1 training)
      shadowed/       ← shadowed TRI X-rays       (for Stage 2 inference)
      clean_ref/      ← paired clean refs         (for Stage 3 eval, optional)
      masks/          ← binary shadow masks       (for masked SDEdit, optional)
    checkpoints/      ← model checkpoints saved here
    results/
      t0_sweep/       ← t₀ tuning outputs
      shadow_removed/ ← final shadow-removed results
"""

import os
import torch
from stage1_train import train
from stage1_model import UNet
from stage1_train import VPSDESchedule
from stage1_dataset import get_shadowed_dataloader
from stage2_inference import SDEditRemover, t0_sweep, run_inference
from stage3_evaluate import evaluate, evaluate_t0_sweep
from PIL import Image
import numpy as np


def main():

    # ── CONFIGURATION ────────────────────────────────────────────────────────
    # Paths to the actual data folders on this machine.
    # Clean data is spread across two source folders — both are used for training.
    # Shadowed data is likewise split across two source folders.

    BASE = os.path.dirname(os.path.abspath(__file__))

    _DATASETS = os.path.join(
        BASE,
        'Datasets_Shadow_Removal_Transfer-20260511T190707Z-3-001',
        'Datasets_Shadow_Removal_Transfer'
    )
    _TRI = os.path.join(
        BASE,
        'TRI_Dataset-20260511T190518Z-3-001',
        'TRI_Dataset'
    )

    # Stage 1 training: clean_auto (62) + clean_manual_fixed (23) = 85 images
    CLEAN_DIR = [
        os.path.join(_DATASETS, 'clean_auto'),
        os.path.join(_DATASETS, 'clean_manual_fixed'),
    ]

    # Stage 2 inference: noisy_auto (217) + noisy_manual_fixed (256) = 473 images
    # To run on TRI images instead, set SHADOW_DIR = _TRI
    SHADOW_DIR = [
        os.path.join(_DATASETS, 'noisy_auto'),
        os.path.join(_DATASETS, 'noisy_manual_fixed'),
    ]

    CLEAN_REF   = None        # no paired clean ground-truth available
    MASK_DIR    = os.path.join(BASE, 'data', 'masks')   # optional, not yet generated
    CKPT_DIR    = os.path.join(BASE, 'checkpoints')
    RESULTS_DIR = os.path.join(BASE, 'results')

    IMAGE_SIZE   = 128    # 4× faster than 256; still captures shadow gradients well
    BATCH_SIZE   = 8
    EPOCHS       = 500    # need ~500 steps/image for a usable prior; ~30 min at 128×128
    LEARNING_RATE= 2e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"SDEdit PCB X-ray Shadow Removal")
    print(f"Device : {device}")
    print(f"{'='*60}\n")

    # ── STAGE 1: Train Diffusion Prior ───────────────────────────────────────
    print("STAGE 1: Training diffusion prior on clean X-rays")
    print("-" * 60)

    stage1_config = {
        'clean_dir'    : CLEAN_DIR,
        'save_dir'     : CKPT_DIR,
        'image_size'   : IMAGE_SIZE,
        'batch_size'   : BATCH_SIZE,
        'epochs'       : EPOCHS,
        'lr'           : LEARNING_RATE,
        'base_channels': 64,
        'num_workers'  : 0,         # keep 0 for Windows
        'resume'       : os.path.join(CKPT_DIR, 'ckpt_epoch100.pt'),  # resume from epoch 100
    }

    FINAL_CKPT = os.path.join(CKPT_DIR, 'diffusion_prior_final.pt')

    if os.path.exists(FINAL_CKPT):
        print(f"[Stage 1] Checkpoint found at {FINAL_CKPT}, skipping training.")
        print(f"          Delete it to force re-training.\n")
    else:
        train(stage1_config)
        print(f"[Stage 1] Complete. Model saved to {FINAL_CKPT}\n")

    # ── t₀ SWEEP: Find the best t₀ before full inference ─────────────────────
    print("t₀ SWEEP: Finding optimal noise level for shadow removal")
    print("-" * 60)

    sweep_dir = os.path.join(RESULTS_DIR, 't0_sweep')
    os.makedirs(sweep_dir, exist_ok=True)

    # Load model
    model = UNet(image_channels=1, base_channels=64, image_size=IMAGE_SIZE).to(device)
    ckpt = torch.load(FINAL_CKPT, map_location=device)
    model.load_state_dict(ckpt.get('ema_model', ckpt['model']))
    model.eval()

    schedule = VPSDESchedule()
    remover  = SDEditRemover(model, schedule, device)

    # Pick one representative shadowed image for the sweep
    _shadow_dirs = [SHADOW_DIR] if isinstance(SHADOW_DIR, str) else SHADOW_DIR
    sample_path = None
    for sd in _shadow_dirs:
        for f in sorted(os.listdir(sd)):
            if f.lower().endswith(('.png', '.jpg', '.tiff', '.bmp')):
                sample_path = os.path.join(sd, f)
                break
        if sample_path:
            break

    if sample_path:
        import torchvision.transforms as T

        transform = T.Compose([
            T.Grayscale(1),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        sample_img = transform(Image.open(sample_path)).unsqueeze(0)

        # Run sweep
        t0_results = t0_sweep(
            remover,
            sample_img,
            x_clean_ref=None,
            t0_values=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            save_dir=sweep_dir
        )

        evaluate_t0_sweep(sweep_dir, sample_path, None, IMAGE_SIZE)

        print(f"\n[Sweep] Visual results saved to: {sweep_dir}")
        print(f"[Sweep] Review images and set T0 below before proceeding.\n")
    else:
        print("[Sweep] No shadowed images found for sweep.")

    # ────────────────────────────────────────────────────────────────────────
    # IMPORTANT: After reviewing the t₀ sweep results,
    # set T0 to the value that gives the best shadow removal
    # while preserving PCB structure.
    # Typical range for X-ray gradients: 0.35–0.45
    # ────────────────────────────────────────────────────────────────────────
    T0 = 0.4   # ← CHANGE THIS after reviewing sweep

    # ── STAGE 2: SDEdit Shadow Removal Inference ─────────────────────────────
    print(f"STAGE 2: Running SDEdit inference (t₀={T0})")
    print("-" * 60)

    stage2_config = {
        'shadowed_dir' : SHADOW_DIR,
        'output_dir'   : os.path.join(RESULTS_DIR, 'shadow_removed'),
        'checkpoint'   : FINAL_CKPT,
        'image_size'   : IMAGE_SIZE,
        'base_channels': 64,
        't0'           : T0,
        'N'            : 500,
        'use_mask'     : os.path.isdir(MASK_DIR),  # auto-enable if masks exist
        'mask_dir'     : MASK_DIR,
    }

    run_inference(stage2_config)
    print(f"[Stage 2] Complete.\n")

    # ── STAGE 3: Evaluation ───────────────────────────────────────────────────
    print("STAGE 3: Evaluating results")
    print("-" * 60)

    stage3_config = {
        'shadowed_dir': SHADOW_DIR,
        'output_dir'  : os.path.join(RESULTS_DIR, 'shadow_removed'),
        'clean_dir'   : CLEAN_REF,   # None — no paired ground-truth available
        'image_size'  : IMAGE_SIZE,
        'save_report' : os.path.join(RESULTS_DIR, 'eval_report.csv')
    }

    metrics = evaluate(stage3_config)
    print(f"[Stage 3] Complete. Report: {stage3_config['save_report']}\n")

    print("=" * 60)
    print("FULL PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
