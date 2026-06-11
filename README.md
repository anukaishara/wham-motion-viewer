<div align="center">

# WHAM — Extended Pipeline & Motion Viewer

**3D human motion reconstruction from video, with GPU preprocessing and an interactive web-based Motion Viewer**

[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=black)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

Built on top of [WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion](https://arxiv.org/abs/2312.07531) (CVPR 2024)  
as an IPCV course project.

</div>

---

## What this adds

| Feature | Description |
|---------|-------------|
| **GPU Preprocessing** | Kornia/PyTorch stage: ego-motion compensation, motion saliency, adaptive gamma, CLAHE, guided filter, unsharp masking |
| **Unified Pipeline** | Single `pipeline.py` entry point — preprocess → infer → extract, fully deterministic |
| **Parameter Extraction** | Quaternion continuity correction + Savitzky-Golay smoothing → clean betas / thetas / trans / contact / camera_path |
| **Web Motion Viewer** | React + Three.js app that plays back the 3D SMPL reconstruction frame-synced with the original video |
| **REST API** | FastAPI backend with live progress streaming for the web UI |

---

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                      Input Video                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Stage 1 — GPU Preprocessing                    │
│   ego-motion compensation → motion saliency             │
│   → adaptive gamma → CLAHE → guided filter              │
│   → unsharp masking                                     │
│                                                         │
│   Runs in isolated subprocess (CUDA context released    │
│   before Stage 2 to avoid ultralytics Conv2d conflicts) │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Stage 2 — WHAM Inference                       │
│   ViTPose / YOLO (2D detection)                         │
│   → DPVO (global SLAM)                                  │
│   → WHAM Transformer → SMPL parameters                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Stage 3 — Parameter Extraction                 │
│   quaternion continuity correction                      │
│   → Savitzky-Golay smoothing                            │
│   → betas · thetas · trans · contact · camera_path      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Web Motion Viewer                          │
│   3D SMPL playback synced with original video           │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/anukaishara/wham-motion-viewer.git --recursive
cd wham-motion-viewer

# 2. Setup environment & dependencies  (see Installation for full steps)
conda create -n wham_dev python=3.9 -y && conda activate wham_dev
pip install -r requirements.txt

# 3. Download weights & models
bash fetch_demo_data.sh   # WHAM / ViTPose / YOLO / DPVO checkpoints
bash setup_models.sh      # SMPL GLB body models for Motion Viewer

# 4. Build frontend & run
cd motion_viewer && npm install && npm run build && cd ..
./start.sh --prod         # → http://localhost:8787
```

---

## Requirements

| Requirement | Version |
|-------------|---------|
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.9 (conda env `wham_dev`) |
| GPU | CUDA-capable (tested on RTX 5090, CUDA 12.8) |
| Node.js | ≥ 18 |

---

## Installation

### 1. Clone with submodules

```bash
git clone https://github.com/anukaishara/wham-motion-viewer.git --recursive
cd wham-motion-viewer
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

> For a different CUDA version, replace `cu128` with yours (e.g. `cu121`) and pick matching versions from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Install torch-scatter

```bash
pip install torch-scatter==2.1.2+pt27cu128 \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### 5. Install pytorch3d *(optional — only needed for `--visualize`)*

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

Register at [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLify](https://smplify.is.tue.mpg.de/) (the script will prompt for credentials), then:

```bash
bash fetch_demo_data.sh
```

Downloads WHAM, ViTPose, YOLO, and DPVO weights into `checkpoints/`.

### 10. Download Motion Viewer body models

```bash
bash setup_models.sh
```

Downloads the custom SMPL GLB body models (~152 MB) into `motion_viewer/public/models/`.

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

### Web app (recommended)

```bash
./start.sh --prod
```

Open **http://localhost:8787** — upload a video, watch live progress, then explore the 3D reconstruction in the Motion Viewer.

For development with hot-reload (frontend on port 5173):

```bash
./start.sh
```

### Pipeline — command line

```bash
conda run -n wham_dev python pipeline.py --input inputs/your_video.mp4
```

| Flag | Description |
|------|-------------|
| `--output_dir DIR` | Root output directory (default: `outputs/`) |
| `--visualize` | Render 3D mesh overlay video — requires pytorch3d |
| `--run_smplify` | Temporal SMPLify refinement — slower, more accurate |
| `--local_only` | Skip global SLAM trajectory (faster) |
| `--skip_preprocess` | Reuse an existing preprocessed video |
| `--calib FILE` | Camera calibration file (auto-estimated if omitted) |

### Original WHAM demo

```bash
conda run -n wham_dev python demo.py --video examples/IMG_9732.mov --visualize

# With known camera intrinsics
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

```bash
conda run -n wham_dev python validate.py
```

Checks 10 properties per output folder — no GPU or weights required:

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

Pipeline tested on 5 activity types across 17 runs.

### Per-video summary

| Video | Frames | FPS | Duration | Jitter (rad/f) | Root travel | Foot contact | Status |
|-------|-------:|----:|---------:|---------------:|------------:|-------------:|:------:|
| Sprint | 371 | 25 | 14.9 s | 0.02737 | 34.66 m | 33% | ✅ PASS |
| Sprint 2 | 191 | 25 | 7.7 s | 0.01287 | 8.45 m | 74% | ✅ PASS |
| Female Dancer | 459 | 25 | 18.4 s | 0.01440 | 4.69 m | 99% | ✅ PASS |
| Male Dancer | 65 | 25 | 5.6 s | 0.01838 | 1.94 m | 80% | ❌ FAIL† |
| Parkour | 302 | 24 | 18.7 s | 0.00801 | 4.85 m | 76% | ✅ PASS |
| Taichi | 712 | 24 | 29.7 s | 0.00389 | 2.90 m | 100% | ✅ PASS |

†Male Dancer: tracker lost subject for 54% of frames (person partially off-screen).

### Physical plausibility

- **Jitter scales with motion intensity** — sprint (0.027) > dancer (0.014) > parkour (0.008) > taichi (0.004)
- **Root travel matches expected displacement** — sprinter covers 34.7 m in 14.9 s (≈ 8.4 km/h); taichi moves 2.9 m in 29.7 s (near-stationary, correct)
- **Foot contact reflects activity** — sprinter airborne 67% of frames; taichi grounded ≥ 90%; dancer at 99% (floor routine)
- **Body shape stays within ±2σ** across all subjects (max observed |β| = 1.45)

### Effect of Savitzky-Golay smoothing

Sprint sequence (371 frames), window=7, poly=3:

| Metric | Raw WHAM | After SG smoothing |
|--------|:--------:|:-----------------:|
| Mean jitter | 0.0628 rad/f | **0.0274 rad/f** |
| Median jitter | 0.0321 rad/f | **0.0182 rad/f** |
| P95 jitter | 0.2069 rad/f | **0.0910 rad/f** |
| Smoothness score (↑) | 0.9409 | **0.9734** |

Pelvis joint improves most: 0.510 → 0.021 rad/f, eliminating the barrel-roll artifact in raw output.

### Determinism

The same video processed twice produces bit-identical outputs. Three independent runs of `sprint_2` all yield jitter = 0.01287 rad/f and travel = 8.446 m.

---

## Project structure

```
wham-motion-viewer/
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
├── checkpoints/         pretrained model weights  (not in repo — run fetch_demo_data.sh)
├── examples/            sample calibration files
├── inputs/              place your own videos here
└── outputs/             pipeline results written here
```

---

## Training

### Stage 1 — 2D-to-SMPL lifting on AMASS

```bash
python train.py --cfg configs/yamls/stage1.yaml
```

### Stage 2 — Feature integration on video datasets

```bash
python train.py --cfg configs/yamls/stage2.yaml \
    TRAIN.CHECKPOINT checkpoints/wham_stage1.tar.pth
```

See [docs/DATASET.md](docs/DATASET.md) for dataset preparation.

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

# EMDB
python -m lib.eval.evaluate_emdb \
    --cfg configs/yamls/demo.yaml --eval-split 1 \
    TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar
```

---

## Acknowledgements

- [WHAM](https://arxiv.org/abs/2312.07531) — Shin et al., CVPR 2024
- [VIBE](https://github.com/mkocabas/VIBE) and [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE) — base implementation
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) — 2D keypoint estimation
- [DPVO](https://github.com/princeton-vl/DPVO) — dense visual odometry

## Citation

```bibtex
@InProceedings{shin2023wham,
  title     = {WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion},
  author    = {Shin, Soyong and Kim, Juyong and Halilaj, Eni and Black, Michael J.},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

## License

MIT — see [LICENSE](./LICENSE) for details.
