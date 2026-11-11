Component-Aware PCB X-ray Restoration Scripts
============================================

This folder contains newer thesis-stage restoration experiments.

The older root-level scripts are kept for history:

- classical_shadow_removal.py
- classical_shadow_removal_v2.py

Current focus:

- J4: remove horizontal seams / banding while preserving solder-ball rows.
- U29: make square pads clearer and reduce noise between square-grid cells.
- J1/U30: keep generic shadow and band correction as secondary targets.

Run the first component-aware experiment:

    python component_restoration/component_restoration_v1.py

Outputs are written to:

    component_restoration/result/component_restoration_v1/

Current recommended experiment:

    py component_restoration/component_restoration_v2_stack.py

This version uses the 8-image stack per component family to estimate a
repeatable artifact/shading template before applying component-specific
cleanup. By default, the current version estimates templates from the full
TRI_Dataset folder and saves restored outputs only for TRI_Dataset_Concised.

Outputs are written to:

    component_restoration/result/component_restoration_v2_stack/

Research-library experiment:

    py -m pip install BaSiCPy PyWavelets
    py component_restoration/component_restoration_v3_research.py

This version tries BaSiCPy for real stack-based flat-field/dark-field
estimation and PyWavelets for a Muench-style wavelet/Fourier J4 destriping
branch. If the libraries are missing, the script prints a clear fallback
message and still produces fallback row-profile outputs.

Outputs are written to:

    component_restoration/result/component_restoration_v3_research/
