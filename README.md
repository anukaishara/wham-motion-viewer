# WHAM — Extended Pipeline with Motion Viewer

An extended implementation of [WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion](https://arxiv.org/abs/2312.07531) (CVPR 2024), built for the IPCV course project.

This repo adds a **GPU-accelerated video preprocessing stage**, a **unified three-stage pipeline**, and a **web-based Motion Viewer** on top of the original WHAM codebase.

---

## What it does

Given an ordinary video of a person, the pipeline reconstructs:

| Output | Description |
|--------|-------------|
| `betas.json` | 10 SMPL body-shape coefficients |
| `thetas.csv` | Per-frame joint pose (72 values × N frames) |
| `trans.csv` | Per-frame world-space root translation (3 × N) |
| `contact.csv` | Per-frame foot-contact probabilities (4 × N) |
| `camera_path.csv` | Per-frame camera pose in world space (7 × N) |
| `metadata.json` | Frame-sync metadata for the Motion Viewer |

Results are consumed by the included **Motion Viewer** — a React app that plays back the 3D reconstruction alongside the original video.

---

## Pipeline overview

```
Input video
    │
    ▼
Stage 1 — GPU Preprocessing (kornia / PyTorch)
    ego-motion compensation → motion saliency → adaptive gamma
    → CLAHE → guided filter → unsharp masking
    │
    ▼
Stage 2 — WHAM Inference
    2D detection (ViTPose / YOLO) → DPVO global SLAM
    → feature extraction → WHAM transformer → SMPL parameters
    │
    ▼
Stage 3 — Parameter Extraction
    quaternion continuity correction → Savitzky-Golay smoothing
    → betas / thetas / trans / contact / camera_path / metadata
```

Stage 1 runs in an isolated subprocess so its CUDA context is fully released before Stage 2 initialises (required to avoid illegal-memory-access errors when ultralytics fuses Conv2d layers after kornia).

---

## Requirements

- Ubuntu 20.04 / 22.04
- Python 3.9 (conda env `wham_dev`)
- CUDA-capable GPU (tested on RTX 5090 with CUDA 12.8)
- Node.js ≥ 18 (for the Motion Viewer frontend)

---

## Installation

### 1. Clone with submodules

```bash
git clone <repo-url> --recursive
cd IPCV_Prj
```

### 2. Create the conda environment

```bash
conda create -n wham_dev python=3.9 -y
conda activate wham_dev
```

### 3. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

> For a different CUDA version, replace `cu128` with your version (e.g. `cu121`) and pick matching torch/torchvision versions from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Install torch-scatter

```bash
pip install torch-scatter==2.1.2+pt27cu128 \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### 5. Install pytorch3d (optional — required for `--visualize`)

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
FORCE_CUDA=1 \
pip install "git+https://github.com/facebookresearch/pytorch3d.git" \
    --no-build-isolation
```

### 6. Install DPVO

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
pip install third-party/DPVO --no-build-isolation
```

### 7. Install ViTPose

```bash
pip install -v -e third-party/ViTPose
```

### 8. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 9. Download model checkpoints

Register at [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLify](https://smplify.is.tue.mpg.de/) (the download script prompts for these credentials), then run:

```bash
bash fetch_demo_data.sh
```

This downloads WHAM weights, ViTPose weights, YOLO weights, DPVO weights, and example videos into `checkpoints/` and `examples/`.

### 10. Download Motion Viewer body models

```bash
bash setup_models.sh
```

This downloads the SMPL GLB body models (female + male, ~152 MB total) into `motion_viewer/public/models/`.

### 11. Build the Motion Viewer frontend

```bash
cd motion_viewer
npm install
npm run build
cd ..
```

> **Note:** `start.sh` resolves Python at `$HOME/miniconda3/envs/wham_dev/bin/python`. If your conda is installed elsewhere (e.g. `~/anaconda3`), edit that line before running.

---

## Usage

### Run the web app (recommended)

```bash
./start.sh --prod
```

Open **http://localhost:8787** in your browser. Upload a video, watch the live progress, then explore the 3D reconstruction in the Motion Viewer.

For development (hot-reload frontend on port 5173):

```bash
./start.sh
```

### Run the pipeline directly

```bash
conda run -n wham_dev python pipeline.py --input inputs/your_video.mp4
```

Common options:

| Flag | Description |
|------|-------------|
| `--output_dir DIR` | Root output directory (default: `outputs/`) |
| `--visualize` | Render a 3D mesh overlay video (requires pytorch3d) |
| `--run_smplify` | Temporal SMPLify refinement — slower, more accurate |
| `--local_only` | Skip global SLAM trajectory (faster) |
| `--skip_preprocess` | Reuse an existing preprocessed video |
| `--calib FILE` | Camera calibration file (auto-estimated if omitted) |

### Run the original WHAM demo

```bash
conda run -n wham_dev python demo.py --video examples/IMG_9732.mov --visualize
```

With known camera intrinsics:

```bash
conda run -n wham_dev python demo.py \
    --video examples/drone_video.mp4 \
    --calib examples/drone_calib.txt \
    --visualize
```

---

## Output structure

```
outputs/<video_name>/
├── <video_name>_processed.mp4       enhanced video fed into WHAM
├── betas.json                       10 SMPL shape coefficients
├── thetas.csv                       N × 72 pose parameters
├── trans.csv                        N × 3 world-space root translation
├── contact.csv                      N × 4 foot-contact probabilities
├── camera_path.csv                  N × 7 camera pose [x,y,z,qx,qy,qz,qw]
├── metadata.json                    frame-sync info for the Motion Viewer
└── wham/
    └── <video_name>_processed/
        ├── wham_output.pkl          full WHAM results (all subjects)
        ├── output.mp4               mesh overlay (only with --visualize)
        ├── slam_results.pth         DPVO global trajectory
        └── tracking_results.pth     per-frame 2D tracking + keypoints
```

---

## Validation

Run the automated validation script against all output folders:

```bash
conda run -n wham_dev python validate.py
```

The script checks 10 properties per run without requiring a GPU or model weights:

| Check | What it verifies |
|-------|-----------------|
| Frame alignment | ≤ 40% of source frames dropped by tracker |
| Theta dimensions | Exactly 72 pose parameters per frame |
| Theta finite values | No NaN / Inf in pose output |
| Temporal smoothness | Mean jitter < 0.08 rad/frame |
| Beta plausibility | All 10 shape coefficients within ±2σ |
| Translation validity | Finite, non-zero world-space travel |
| Contact value range | Reports min/max (informational) |
| Contact plausibility | At least 1% of frames show foot contact |
| Cross-file consistency | thetas, trans, contact all have the same frame count |
| Camera path | 7-column output, quaternions are unit-length |

---

## Results

Pipeline tested on 5 distinct activity types across 17 runs.

### Per-video summary

| Video | Frames | FPS | Duration | Theta jitter (rad/f) | Root travel (m) | Foot contact | Status |
|-------|-------:|----:|----------:|---------------------:|----------------:|-------------:|--------|
| Sprint | 371 | 25 | 14.9 s | 0.02737 | 34.66 | 33% | PASS |
| Sprint 2 | 191 | 25 | 7.7 s | 0.01287 | 8.45 | 74% | PASS |
| Female Dancer | 459 | 25 | 18.4 s | 0.01440 | 4.69 | 99% | PASS |
| Male Dancer | 65 | 25 | 5.6 s | 0.01838 | 1.94 | 80% | FAIL† |
| Parkour | 302 | 24 | 18.7 s | 0.00801 | 4.85 | 76% | PASS |
| Taichi | 712 | 24 | 29.7 s | 0.00389 | 2.90 | 100% | PASS |

†Male Dancer failed frame alignment — tracker lost the subject for 54% of frames (person partially off-screen for extended periods).

### Physical plausibility

Results are physically consistent with the activity type:

- **Jitter scales with motion intensity** — sprint (0.027) > dancer (0.014) > parkour (0.008) > taichi (0.004)
- **Root travel matches expected displacement** — sprinter covers 34.7 m across 14.9 s (≈ 8.4 km/h, plausible for indoor sprint); taichi moves 2.9 m in 29.7 s (near stationary, correct)
- **Foot contact patterns reflect activity** — sprinter is airborne 67% of the time (low contact); taichi maintains ground contact ≥ 90% of frames; dancer at 99% (choreographed floor routine)
- **Body shape (betas) stays within ±2σ** across all subjects (max observed |β| = 1.45)

### Frame alignment

The preprocessed video and the theta/trans/contact CSVs are frame-aligned by construction. For well-tracked sequences, WHAM drops at most 1 frame:

| Video | Source frames | Theta frames | Dropped |
|-------|-------------:|-------------:|--------:|
| Sprint | 372 | 371 | 1 |
| Sprint 2 | 192 | 191 | 1 |
| Female Dancer | 460 | 459 | 1 |
| Parkour | 448 | 302 | 146 (tracker) |

### Temporal smoothness — effect of Savitzky-Golay post-processing

On the sprint sequence (371 frames), applying Savitzky-Golay smoothing (window=7, poly=3) after quaternion continuity correction significantly reduces temporal jitter:

| Metric | Raw WHAM output | After SG smoothing |
|--------|----------------:|-------------------:|
| Mean jitter | 0.0628 rad/frame | **0.0274 rad/frame** |
| Median jitter | 0.0321 rad/frame | **0.0182 rad/frame** |
| P95 jitter | 0.2069 rad/frame | **0.0910 rad/frame** |
| Smoothness score (↑ better) | 0.9409 | **0.9734** |

The pelvis joint benefits most: 0.510 → 0.021 rad/frame, which directly eliminates the barrel-roll artifact visible in unsmoothed output.

### Determinism

The same video processed twice produces bit-identical outputs. Three separate runs of `sprint_2` all yield `thetas` jitter = 0.01287 and travel = 8.446 m, confirming the pipeline is fully deterministic given the same input.

### Preprocessing visual comparison

Stage 1 GPU preprocessing improves contrast, sharpens joint landmarks, and suppresses background noise without altering the subject's appearance:

> `outputs/sprint_raw_vs_preprocessed.png` — 4 frames × 2 rows (raw top, preprocessed bottom)

---

## Training

Training follows the two-stage procedure from the original WHAM paper.

### Stage 1 — 2D-to-SMPL lifting on AMASS

```bash
python train.py --cfg configs/yamls/stage1.yaml
```

### Stage 2 — Feature integration on video datasets

```bash
python train.py --cfg configs/yamls/stage2.yaml \
    TRAIN.CHECKPOINT checkpoints/wham_stage1.tar.pth
```

See [docs/DATASET.md](docs/DATASET.md) for dataset preparation instructions.

---

## Evaluation

```bash
# 3DPW
python -m lib.eval.evaluate_3dpw \
    --cfg configs/yamls/demo.yaml \
    TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar

# RICH
python -m lib.eval.evaluate_rich \
    --cfg configs/yamls/demo.yaml \
    TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar

# EMDB (split 1 and 2)
python -m lib.eval.evaluate_emdb \
    --cfg configs/yamls/demo.yaml --eval-split 1 \
    TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar

python -m lib.eval.evaluate_emdb \
    --cfg configs/yamls/demo.yaml --eval-split 2 \
    TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar
```

---

## Project structure

```
IPCV_Prj/
├── pipeline.py          unified 3-stage pipeline (preprocess → WHAM → extract)
├── server.py            FastAPI backend for the web app
├── wham_api.py          Python API wrapper
├── demo.py              original WHAM demo script
├── train.py             training entry point
├── start.sh             launch script (dev and production modes)
├── fetch_demo_data.sh   download checkpoints and example data
├── setup_models.sh      download Motion Viewer GLB body models
├── configs/             YAML configs and constants
├── lib/
│   ├── core/            trainer and loss functions
│   ├── data/            dataset loaders and augmentation
│   ├── data_utils/      preprocessing utilities for training data
│   ├── eval/            evaluation scripts (3DPW, EMDB, RICH)
│   ├── models/          WHAM network, SMPL body model, preprocessors
│   ├── utils/           geometry, transforms, misc helpers
│   └── vis/             visualisation tools
├── third-party/
│   ├── ViTPose/         2D keypoint estimator
│   └── DPVO/            dense visual odometry (global SLAM)
├── motion_viewer/       React + Vite 3D viewer frontend
├── checkpoints/         pretrained model weights
├── examples/            sample videos and calibration files
├── inputs/              place your own videos here
└── outputs/             pipeline results written here
```

---

## Acknowledgements

The WHAM model and training code are from [Shin et al., CVPR 2024](https://arxiv.org/abs/2312.07531).  
The base implementation borrows from [VIBE](https://github.com/mkocabas/VIBE) and [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE).  
2D keypoints are estimated with [ViTPose](https://github.com/ViTAE-Transformer/ViTPose).  
Global camera motion is estimated with [DPVO](https://github.com/princeton-vl/DPVO).

## Citation

```bibtex
@InProceedings{shin2023wham,
  title   = {WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion},
  author  = {Shin, Soyong and Kim, Juyong and Halilaj, Eni and Black, Michael J.},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year    = {2024}
}
```

## License

See [LICENSE](./LICENSE) for details.
