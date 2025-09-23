# CE-DDIM: Conditional Efficient DDIM for CBCT-to-CT Enhancement

# CBCT → sCT with Diffusion Models and Baselines

This repository contains code to generate **synthetic CT (sCT)** from **CBCT** with a diffusion-based model (dual‑head with DDIM sampling).
---

## Table of Contents

* [Environment](#environment)
* [Data & Preprocessing](#data--preprocessing)
* [Project Structure](#project-structure)
* [Quickstart](#quickstart)
* [Training](#training)
* [Evaluation](#evaluation)
* [Visualization](#visualization)
* [Reproducibility](#reproducibility)
* [Notes on Windowing (HU)](#notes-on-windowing-hu)
* [Results Tracking](#results-tracking)
* [Citations](#citations)
* [License](#license)

---

## Environment

Tested with Python 3.10+, PyTorch 2.2+, CUDA 12.x.

```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>

# (Recommended) Conda
conda create -n sct python=3.10 -y
conda activate sct

# Core deps
pip install -r requirements.txt
# or explicitly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nibabel SimpleITK monai[all] torchio scikit-image tqdm matplotlib pyyaml einops
```

Optional: `wandb` or `tensorboard` for logging; `git-lfs` if you store large checkpoints.

---

## Data & Preprocessing

**Expected layout** (NIfTI suggested):

```
DATA_ROOT/
  P001/
    ct.nii.gz         # planning CT in HU
    cbct.nii.gz       # CBCT in HU
    mask.nii.gz       # body/ROI mask (binary)
  P002/
    ...
```

Add split files (IDs per line):

```
splits/
  train.txt
  val.txt
  test.txt
```

**Preprocessing** (default):

* Resample to common spacing (e.g., 1×1×1 mm) and align CBCT↔CT (rigid/Deformable as per your protocol).
* **HU clip** base window to **\[−1000, 3000]** and normalize to **\[0, 1]** for model input.
* Masks are binary (0/1); errors are computed **inside mask** unless noted.

Scripts:

* `scripts/prepare_data.py` — optional helper for resampling, cropping, and normalization metadata.

---

## Project Structure

```
<REPO>/
  configs/
    dualhead_default.yaml
    ddim.yaml
    pix2pix.yaml
    unet.yaml
  scripts/
    train.py
    eval.py
    viz_compare.py
    prepare_data.py
  models/
    dualhead/
    pix2pix/
    unet/
  utils/
    data.py            # datasets & transforms
    metrics.py         # MAE, RMSE, PSNR, SSIM, etc.
    windowing.py       # HU↔[0,1] conversions
    vis.py             # plotting (multi‑panel, ROI insets, colorbars)
  outputs/
    ...                # logs, checkpoints, predictions
```

---

## Quickstart

Train diffusion (dual‑head) 
