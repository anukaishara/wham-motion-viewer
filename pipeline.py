#!/usr/bin/env python3
# =============================================================================
# WHAM Unified Pipeline
# =============================================================================
#
# Three stages executed end-to-end:
#
#   Stage 1 — Video preprocessing
#               Motion-saliency masking, adaptive gamma, CLAHE, guided filter,
#               and unsharp masking — all GPU-accelerated via kornia/PyTorch.
#
#   Stage 2 — WHAM inference
#               2D pose detection (ViTPose/YOLO), optional DPVO global SLAM,
#               per-person feature extraction, WHAM transformer inference.
#
#   Stage 3 — Parameter extraction
#               Reads wham_output.pkl and writes betas.json, thetas.csv,
#               trans.csv, contact.csv, camera_path.csv, and metadata.json.
#
# REQUIREMENTS
#   Conda environment : wham_dev   (see README for setup)
#   Run from          : /home/sensor_readings/WHAM_dev/
#
# BASIC USAGE
#   conda run -n wham_dev python pipeline.py --input inputs/your_video.mp4
#
# COMMON OPTIONS
#   --output_dir DIR      Where to write all outputs  (default: outputs/)
#   --visualize           Also render a mesh-overlay video (requires pytorch3d)
#   --run_smplify         Temporal SMPLify refinement — slower, more accurate
#   --local_only          Skip global trajectory (no DPVO); faster
#   --skip_preprocess     Reuse an existing preprocessed video from a previous run
#   --calib FILE          Camera calibration file (optional, auto-estimated if absent)
#
# OUTPUT STRUCTURE
#   outputs/<video_name>/
#   ├── <video_name>_processed.mp4          enhanced video fed into WHAM
#   ├── betas.json                          10 SMPL body-shape coefficients
#   ├── thetas.csv                          N frames × 72 pose parameters
#   ├── trans.csv                           N frames × 3 world-space root translation
#   ├── contact.csv                         N frames × 4 foot-contact probabilities
#   ├── camera_path.csv                     N frames × 7 camera pose [x,y,z,qx,qy,qz,qw]
#   ├── metadata.json                       sync metadata for the Motion Viewer
#   └── wham/
#       └── <video_name>_processed/
#           ├── wham_output.pkl             full WHAM results (all subjects)
#           ├── output.mp4                  mesh overlay (only with --visualize)
#           ├── slam_results.pth            DPVO global trajectory
#           └── tracking_results.pth        per-frame 2D tracking + keypoints
#
# =============================================================================

import os
import sys
import json
import time
import argparse
import subprocess
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Server-mode progress reporting
# ─────────────────────────────────────────────────────────────────────────────
# When invoked by server.py, --progress_file points to a JSON file that the
# server's SSE endpoint reads and streams to the browser as live progress.

_PROGRESS_FILE = None   # set to a path string when called from server.py


def _write_progress(data):
    """Write a progress snapshot to _PROGRESS_FILE (no-op if not set)."""
    if not _PROGRESS_FILE:
        return
    data['ts'] = time.time()
    try:
        with open(_PROGRESS_FILE, 'w') as fh:
            json.dump(data, fh)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — GPU-Accelerated Video Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _box_filter(t, r):
    """Fast box filter via 2D average pooling (GPU-native)."""
    return torch.nn.functional.avg_pool2d(t, kernel_size=2*r+1, stride=1, padding=r)


def _guided_filter(I, p, r=3, eps=0.01):
    """
    Edge-preserving guided filter (He et al., 2013) — GPU version.

    I: guidance image  [B,C,H,W] float32 [0,1]
    p: input to filter [B,C,H,W] float32 [0,1]

    Computes per-pixel linear coefficients (a, b) from the guidance image so
    that edges in I are preserved while noise in p is suppressed.
    """
    mean_I  = _box_filter(I, r)
    mean_p  = _box_filter(p, r)
    mean_Ip = _box_filter(I * p, r)
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = _box_filter(I * I, r) - mean_I * mean_I
    a       = cov_Ip / (var_I + eps)
    b       = mean_p - a * mean_I
    return _box_filter(a, r) * I + _box_filter(b, r)


def _unsharp_mask_gpu(t, sigma=1.0, amount=0.5):
    """
    Unsharp masking on GPU.
    Sharpens high-frequency details (joint landmarks) to improve ViTPose accuracy.
    """
    import kornia
    blurred = kornia.filters.gaussian_blur2d(t, kernel_size=(5, 5), sigma=(sigma, sigma))
    return torch.clamp((1.0 + amount) * t - amount * blurred, 0.0, 1.0)


def _clahe_gpu(t, clip_limit=2.0, grid_size=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel.
    Enhances local contrast without blowing out highlights or crushing shadows.

    t: BGR tensor [B,3,H,W] float32 [0,1]
    """
    import kornia
    rgb = torch.flip(t, dims=[1])                       # BGR → RGB
    lab = kornia.color.rgb_to_lab(rgb)                  # L channel ∈ [0, 100]
    l_norm      = lab[:, :1] / 100.0
    lab[:, :1]  = kornia.enhance.equalize_clahe(
        l_norm, clip_limit=clip_limit, grid_size=grid_size
    ) * 100.0
    return torch.flip(kornia.color.lab_to_rgb(lab), dims=[1])   # RGB → BGR


def _estimate_camera_motion(prev_gray, curr_gray):
    """
    Estimate the camera's affine motion between two frames using
    Lucas-Kanade optical flow on background corners.

    Returns a 2×3 affine matrix that maps points from the previous frame's
    coordinate system into the current frame's.  Falls back to the identity
    (no motion) when fewer than 4 corner matches are found.

    This stabilisation step is used before frame-differencing so that a
    panning/tracking camera does not flood the background with false positives
    in the motion-saliency mask.
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,
                                 qualityLevel=0.01, minDistance=30)
    if p0 is None:
        return np.eye(2, 3, dtype=np.float32)

    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    if p1 is None:
        return np.eye(2, 3, dtype=np.float32)

    good_new = p1[status == 1]
    good_old = p0[status == 1]
    if len(good_new) < 4:
        return np.eye(2, 3, dtype=np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    return matrix if matrix is not None else np.eye(2, 3, dtype=np.float32)


def preprocess_video(input_path, output_path, progress_path=None):
    """
    GPU preprocessing pipeline applied per frame (6 sequential stages):

      1. Ego-motion compensation  — LK optical flow estimates the affine
                                    camera motion; the previous frame is warped
                                    to align with the current view so that
                                    the subsequent frame diff is camera-stable.

      2. Motion saliency mask     — absolute pixel difference on the stabilised
                                    pair + max-pooling dilation isolates the
                                    region occupied by the moving subject.

      3. Person-focused gamma     — gamma 1.5 applied selectively to the
                                    salient (person) region; a brightness-
                                    adaptive base gamma is used for the
                                    background to avoid over-brightening.

      4. CLAHE (kornia, GPU)      — adaptive local contrast on the L channel
                                    of the Lab colour space; recovers shadow
                                    and highlight detail without blowout.

      5. Guided filter (GPU)      — edge-preserving smoothing that suppresses
                                    compression noise while keeping the sharp
                                    body-silhouette contours needed for 2D
                                    pose estimation.

      6. Unsharp masking (GPU)    — sharpens joint-landmark regions to improve
                                    ViTPose keypoint localisation accuracy.

    The pipeline runs entirely on CUDA (~10–20× faster than a CPU equivalent).
    Falls back to CPU automatically when CUDA is unavailable.
    """
    import kornia

    # Empty-string CUDA_VISIBLE_DEVICES silently hides all GPUs — clear it.
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        del os.environ['CUDA_VISIBLE_DEVICES']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), f"Cannot open video: {input_path}"

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    def to_tensor(frame):
        """BGR uint8 HWC → float32 BCHW [0, 1] on device."""
        return (torch.from_numpy(frame)
                     .to(device).float()
                     .permute(2, 0, 1).unsqueeze(0) / 255.0)

    def to_frame(t):
        """float32 BCHW [0, 1] → BGR uint8 HWC numpy array."""
        return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

    logger.info(
        f"Stage 1 — GPU preprocessing  {width}x{height} @ {fps:.1f} fps  "
        f"({total} frames)  device={device}"
    )

    progress_file = progress_path or _PROGRESS_FILE
    t_start       = time.time()

    def write_stage1_progress(frame_idx):
        if not progress_file:
            return
        elapsed  = time.time() - t_start
        rate     = frame_idx / elapsed if elapsed > 0 else 0
        eta_s    = (total - frame_idx) / rate if rate > 0 else None
        pct      = 5 + int(frame_idx / max(total, 1) * 20)  # maps [0, total] → [5%, 25%]
        try:
            with open(progress_file, 'w') as fh:
                json.dump({
                    'stage': 1, 'stage_name': 'Preprocessing video',
                    'frame': frame_idx, 'total_frames': total - 1,
                    'pct': pct, 'eta_s': round(eta_s, 1) if eta_s else None,
                    'status': 'running', 'ts': time.time(),
                }, fh)
        except Exception:
            pass

    write_stage1_progress(0)

    ret, first_frame = cap.read()
    assert ret, "Video has no frames"
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_t    = to_tensor(first_frame)

    # inference_mode is faster than no_grad — fully disables autograd overhead.
    with torch.inference_mode():
        for frame_idx, _ in enumerate(
            tqdm(range(total - 1), desc="Stage 1  preprocess"), start=1
        ):
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_t    = to_tensor(frame)

            # ── 1. Ego-motion compensation ──────────────────────────────────
            # Warp the previous frame into the current camera pose so that
            # background pixels cancel out in the stabilised diff.
            affine_matrix = _estimate_camera_motion(prev_gray, curr_gray)
            affine_t      = torch.from_numpy(affine_matrix).to(device).unsqueeze(0).float()
            prev_t_aligned = kornia.geometry.transform.warp_affine(
                prev_t, affine_t, dsize=(height, width), align_corners=True
            )

            # ── 2. Motion saliency mask ─────────────────────────────────────
            diff     = torch.abs(curr_t - prev_t_aligned)
            saliency = (torch.max(diff, dim=1, keepdim=True)[0] > 0.1).float()
            # Dilate to cover the full body contour, not just moving edges.
            saliency = torch.nn.functional.max_pool2d(
                saliency, kernel_size=21, stride=1, padding=10
            )

            # ── 3. Person-focused adaptive gamma ───────────────────────────
            mean_brightness = curr_t.mean().item()
            base_gamma      = float(np.clip(1.5 - mean_brightness, 0.8, 1.5))
            gamma_map       = base_gamma + saliency * 0.5  # person: +0.5 extra boost
            enhanced        = torch.pow(curr_t.clamp(min=1e-6), 1.0 / gamma_map)

            # ── 4. CLAHE — adaptive local contrast ─────────────────────────
            dynamic_clip = float(np.clip(2.5 - mean_brightness * 2.0, 0.8, 2.0))
            enhanced     = _clahe_gpu(enhanced, clip_limit=dynamic_clip)

            # ── 5. Guided filter — edge-preserving smoothing ───────────────
            # Use the original (unenhanced) frame as guidance to anchor edges.
            enhanced = _guided_filter(curr_t, enhanced, r=3, eps=0.01)

            # ── 6. Unsharp masking — sharpen keypoint landmarks ────────────
            enhanced = _unsharp_mask_gpu(enhanced, sigma=1.0, amount=0.5)

            out.write(to_frame(enhanced))

            prev_t    = curr_t
            prev_gray = curr_gray

            if frame_idx % 20 == 0:
                write_stage1_progress(frame_idx)

    cap.release()
    out.release()
    logger.info(f"Stage 1 — Done  →  {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — WHAM Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_wham(cfg, video, output_pth, network,
             calib=None, run_global=True, save_pkl=True,
             visualize=False, run_smplify=False,
             progress_base_pct=(25, 80)):
    """
    Full WHAM inference pipeline (mirrors demo.py::run).

    Two-pass design:
      Pass 1 — frame-by-frame detection (ViTPose / YOLO bounding boxes),
               DPVO SLAM (optional, for global trajectory), and per-box
               feature extraction.  Results are cached to disk so the pass
               can be skipped on re-runs.
      Pass 2 — WHAM transformer inference over the full sequence per subject,
               with optional flip-augmentation averaging and Temporal SMPLify
               refinement.

    Outputs wham_output.pkl containing SMPL parameters for every tracked
    subject.
    """
    from lib.data.datasets import CustomDataset
    from lib.utils.imutils import avg_preds
    from lib.utils.transforms import matrix_to_axis_angle
    from lib.models.preproc.detector import DetectionModel
    from lib.models.preproc.extractor import FeatureExtractor
    from lib.models.smplify import TemporalSMPLify

    try:
        from lib.models.preproc.slam import SLAMModel
        slam_available = True
    except Exception:
        logger.warning("DPVO not available — running in local-coordinates mode only")
        slam_available = False

    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f"Failed to open video: {video}"
    fps    = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    run_global = run_global and slam_available

    # Progress range allocation: ~55% of stage 2's budget goes to detection.
    pct_det_start, pct_wham_end = progress_base_pct
    pct_det_end    = pct_det_start + int((pct_wham_end - pct_det_start) * 0.55)
    pct_wham_start = pct_det_end

    with torch.no_grad():
        tracking_cache = osp.join(output_pth, 'tracking_results.pth')
        slam_cache     = osp.join(output_pth, 'slam_results.pth')

        if not (osp.exists(tracking_cache) and osp.exists(slam_cache)):
            _write_progress({'stage': 2, 'stage_name': 'Detection & tracking',
                             'pct': pct_det_start, 'status': 'running'})

            detector  = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            slam      = (SLAMModel(video, output_pth, width, height, calib)
                         if run_global else None)

            # Pass 1: per-frame detection and SLAM tracking.
            t_detect_start = time.time()
            for frame_idx, _ in enumerate(
                tqdm(range(length), desc="Stage 2  detection + SLAM"), start=1
            ):
                flag, img = cap.read()
                if not flag:
                    break
                detector.track(img, fps, length)
                if slam is not None:
                    slam.track()

                if frame_idx % 30 == 0:
                    elapsed   = time.time() - t_detect_start
                    rate      = frame_idx / elapsed if elapsed > 0 else 0
                    eta_s     = (length - frame_idx) / rate if rate > 0 else None
                    pct       = pct_det_start + int(
                        frame_idx / max(length, 1) * (pct_det_end - pct_det_start)
                    )
                    _write_progress({
                        'stage': 2, 'stage_name': 'Detection & tracking',
                        'frame': frame_idx, 'total_frames': length,
                        'pct': pct, 'eta_s': round(eta_s, 1) if eta_s else None,
                        'status': 'running',
                    })

            _write_progress({'stage': 2, 'stage_name': 'Extracting image features',
                             'pct': pct_det_end, 'status': 'running'})

            tracking_results = detector.process(fps)
            slam_results     = slam.process() if slam is not None else _zero_slam(length)

            # Pass 1b: extract image features for each tracked bounding box.
            tracking_results = extractor.run(video, tracking_results)
            logger.info("Stage 2 — Preprocessing complete")

            joblib.dump(tracking_results, tracking_cache)
            joblib.dump(slam_results,     slam_cache)
            logger.info(f"Stage 2 — Cached tracking + SLAM to {output_pth}")

        else:
            tracking_results = joblib.load(tracking_cache)
            slam_results     = joblib.load(slam_cache)
            logger.info(f"Stage 2 — Loaded existing cache from {output_pth}")

    cap.release()

    # Pass 2: WHAM transformer inference over the full sequence per subject.
    _write_progress({'stage': 2, 'stage_name': 'WHAM inference',
                     'pct': pct_wham_start, 'status': 'running'})

    dataset  = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    results  = defaultdict(dict)
    n_subjs  = len(dataset)
    total_frames = 0
    t_wham_start = time.time()

    for subj_idx in range(n_subjs):
        pct_subj = pct_wham_start + int(
            (subj_idx / max(n_subjs, 1)) * (pct_wham_end - pct_wham_start)
        )
        _write_progress({
            'stage': 2,
            'stage_name': f'WHAM inference (person {subj_idx + 1}/{n_subjs})',
            'pct': pct_subj, 'status': 'running',
        })

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Average forward-pass and horizontally-flipped predictions.
                # Flip augmentation reduces left-right bias in the pose estimator.
                flipped_batch = dataset.load_data(subj_idx, True)
                subj_id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask,
                                       init_root=init_root, cam_angvel=cam_angvel,
                                       return_y_up=True, **kwargs)

                batch = dataset.load_data(subj_idx)
                subj_id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask,
                               init_root=init_root, cam_angvel=cam_angvel,
                               return_y_up=True, **kwargs)

                flipped_pose  = flipped_pred['pose'].squeeze(0).reshape(-1, 24, 6)
                flipped_shape = flipped_pred['betas'].squeeze(0)
                pose          = pred['pose'].squeeze(0).reshape(-1, 24, 6)
                shape         = pred['betas'].squeeze(0)

                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose    = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2

                network.pred_pose    = avg_pose.view_as(network.pred_pose)
                network.pred_shape   = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred   = network.refine_trajectory(output, cam_angvel, return_y_up=True)

            else:
                batch = dataset.load_data(subj_idx)
                subj_id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask,
                               init_root=init_root, cam_angvel=cam_angvel,
                               return_y_up=True, **kwargs)

        # Optional: refine with Temporal SMPLify (2D keypoint reprojection fitting).
        if run_smplify:
            from lib.models import build_body_model as _bbm
            smpl     = _bbm(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
            smplify  = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            pred     = smplify.fit(pred, dataset.tracking_results[subj_id]['keypoints'], **kwargs)
            with torch.no_grad():
                network.pred_pose  = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam   = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred   = network.refine_trajectory(output, cam_angvel, return_y_up=True)

        # Convert rotation matrices → axis-angle for compact storage.
        pred_body_pose  = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root_cam   = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)

        results[subj_id]['pose']        = np.concatenate((pred_root_cam,   pred_body_pose), axis=-1)
        results[subj_id]['pose_world']  = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        results[subj_id]['trans']       = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        results[subj_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[subj_id]['betas']       = pred['betas'].cpu().squeeze(0).numpy()
        results[subj_id]['verts']       = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[subj_id]['frame_ids']   = frame_id
        # 4 contact probabilities per frame: [left_heel, right_heel, left_toe, right_toe]
        results[subj_id]['contact']     = pred['contact'].squeeze(0).cpu().numpy()
        total_frames += len(frame_id)

    wham_elapsed = time.time() - t_wham_start
    logger.info(
        f"Total WHAM Inference Time: {wham_elapsed:.2f} s  "
        f"({int(wham_elapsed // 60)}m {wham_elapsed % 60:.1f}s)"
    )
    logger.info(f"Average WHAM FPS: {total_frames / wham_elapsed:.2f}  ({total_frames} frames)")

    if save_pkl:
        pkl_out = osp.join(output_pth, 'wham_output.pkl')
        joblib.dump(results, pkl_out)
        logger.info(f"Stage 2 — Saved WHAM output to {pkl_out}")

    if visualize:
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)

    return results


def _zero_slam(length):
    """Fallback SLAM result (identity quaternion) used when DPVO is unavailable."""
    slam = np.zeros((length, 7))
    slam[:, 3] = 1.0   # unit quaternion: w=1, xyz=0
    return slam


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Parameter Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_axis_angle(pose, window=7, poly=3):
    """
    Savitzky-Golay temporal smoothing for N×D pose/translation data.
    Preserves motion peaks better than a plain moving average by fitting a
    local polynomial through each window rather than averaging values.
    Falls back to a uniform moving average when scipy is unavailable.
    """
    try:
        from scipy.signal import savgol_filter
        w = min(window, pose.shape[0])
        if w % 2 == 0:
            w -= 1
        if w >= 3:
            return savgol_filter(pose, window_length=w,
                                 polyorder=min(poly, w - 1), axis=0)
    except ImportError:
        pass
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(pose, size=window, axis=0)


def _normalize_quaternion_continuity(pose):
    """
    Eliminate barrel-flip discontinuities in axis-angle pose sequences.

    When a rotation magnitude crosses π, the two equivalent quaternion
    representations (q and −q) can swap sign between frames, causing a sudden
    large-angle jump in axis-angle space.  This function:
      1. Converts each joint's axis-angle trajectory to quaternions.
      2. Enforces sign continuity: if dot(q_{t-1}, q_t) < 0, negates q_t.
      3. Converts back to axis-angle.

    Fixing the discontinuity at the source means downstream consumers never
    see the artifact, regardless of interpolation method.
    """
    from scipy.spatial.transform import Rotation

    n_frames = pose.shape[0]
    n_joints = 24
    out_pose = pose.copy()

    for j in range(n_joints):
        aa   = pose[:, j*3 : j*3+3]                   # (N, 3) axis-angle
        quat = Rotation.from_rotvec(aa).as_quat()     # (N, 4) xyzw

        # Walk forward, negating any quaternion whose sign disagrees with the previous.
        for i in range(1, n_frames):
            if np.dot(quat[i-1], quat[i]) < 0:
                quat[i] = -quat[i]

        out_pose[:, j*3 : j*3+3] = Rotation.from_quat(quat).as_rotvec()

    return out_pose


def extract_params(pkl_path, output_dir, slam_path=None):
    """
    Read wham_output.pkl and write the following output files:

      betas.json        — mean SMPL shape coefficients (10 values), clipped to ±2σ
      thetas.csv        — per-frame pose parameters (72 cols), continuity-corrected
                          and Savitzky-Golay smoothed
      trans.csv         — per-frame world-coordinate root translation (3 cols),
                          Savitzky-Golay smoothed
      contact.csv       — per-frame foot-contact probabilities (4 cols)
      camera_path.csv   — per-frame camera pose [x, y, z, qx, qy, qz, qw]
                          derived from the world↔camera root transform
    """
    logger.info(f"Stage 3 — Extracting parameters from {pkl_path}")

    data      = joblib.load(pkl_path)
    first_key = list(data.keys())[0]
    subject   = data[first_key] if isinstance(data[first_key], dict) else data

    def to_numpy(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)

    # ── Body shape (betas) ────────────────────────────────────────────────────
    betas     = to_numpy(subject['betas'])
    avg_betas = np.mean(betas, axis=0) if betas.ndim > 1 else betas
    # Clip to ±2σ: values beyond this range are implausible body shapes.
    avg_betas = np.clip(avg_betas, -2.0, 2.0)

    beta_path = osp.join(output_dir, 'betas.json')
    with open(beta_path, 'w') as fh:
        json.dump({"betas": avg_betas.tolist()}, fh, indent=4)
    logger.info(f"Stage 3 — betas.json  ({len(avg_betas)} shape coefficients)")

    # ── Pose (thetas) ─────────────────────────────────────────────────────────
    # IMPORTANT: use pose_world (gravity-aligned root), NOT pose (camera-space root).
    #
    # subject['pose']       root is relative to the camera   → OpenCV Y-down convention
    #                        encodes a ~π rotation that produces large discontinuous
    #                        jumps every time the angle crosses the π boundary.
    # subject['pose_world'] root is gravity-aligned           → small, stable values
    #                        for a standing person the root values are near zero.
    #
    # Mixing camera-space root with world-space translation is the root cause
    # of the barrel-roll artifact seen in earlier pipeline versions.
    pose = to_numpy(subject['pose_world'])
    num_frames, num_thetas = pose.shape
    if num_thetas != 72:
        logger.warning(f"Expected 72 theta columns, got {num_thetas} — check WHAM config")

    logger.info("Stage 3 — Normalizing quaternion continuity …")
    pose = _normalize_quaternion_continuity(pose)

    logger.info("Stage 3 — Smoothing thetas with Savitzky-Golay filter …")
    pose = _smooth_axis_angle(pose, window=7, poly=3)

    theta_path = osp.join(output_dir, 'thetas.csv')
    pd.DataFrame(pose).to_csv(theta_path, header=False, index=False)
    logger.info(f"Stage 3 — thetas.csv  ({num_frames} frames × {num_thetas} params)")

    # ── Root translation (world space) ────────────────────────────────────────
    trans_world = to_numpy(subject['trans_world'])

    logger.info("Stage 3 — Smoothing translation …")
    trans_world = _smooth_axis_angle(trans_world, window=9, poly=3)

    trans_path = osp.join(output_dir, 'trans.csv')
    pd.DataFrame(trans_world).to_csv(trans_path, header=False, index=False)
    logger.info(f"Stage 3 — trans.csv   ({trans_world.shape[0]} frames × 3)")

    # ── Foot contact probabilities ────────────────────────────────────────────
    # Columns: [left_heel, right_heel, left_toe, right_toe] (sigmoid outputs).
    contact_path = None
    if 'contact' in subject:
        contact = to_numpy(subject['contact'])
        if contact.ndim == 1:
            contact = contact.reshape(-1, 4)
        contact_path = osp.join(output_dir, 'contact.csv')
        pd.DataFrame(contact).to_csv(contact_path, header=False, index=False)
        logger.info(f"Stage 3 — contact.csv ({contact.shape[0]} frames × 4)")
    else:
        logger.warning("Stage 3 — contact data not found in PKL; skipping contact.csv")

    # ── Camera path (world space) ─────────────────────────────────────────────
    # Reconstruct the camera's world-space pose so the Motion Viewer can
    # recreate the original video perspective exactly.
    #
    # Given:  R_c = root rotation in camera space
    #         R_w = root rotation in world space
    # Then:   R_cw = R_w @ R_c^T          (camera-to-world rotation)
    #         T_cw = T_w − R_cw @ T_c     (camera-to-world translation)
    cam_path = None
    try:
        from scipy.spatial.transform import Rotation as R

        pose_c  = to_numpy(subject['pose'])[:, :3]       # root in camera space
        pose_w  = to_numpy(subject['pose_world'])[:, :3] # root in world space
        trans_c = to_numpy(subject['trans'])

        R_c  = R.from_rotvec(pose_c).as_matrix()
        R_w  = R.from_rotvec(pose_w).as_matrix()
        R_cw = R_w @ R_c.transpose(0, 2, 1)
        T_cw = trans_world - np.einsum('bij,bj->bi', R_cw, trans_c)
        Q_cw = R.from_matrix(R_cw).as_quat()   # [x, y, z, w]

        cam_path_data = np.concatenate((T_cw, Q_cw), axis=-1)
        cam_path_data = _smooth_axis_angle(cam_path_data, window=5, poly=2)

        cam_path = osp.join(output_dir, 'camera_path.csv')
        pd.DataFrame(cam_path_data).to_csv(cam_path, header=False, index=False)
        logger.info(f"Stage 3 — camera_path.csv ({cam_path_data.shape[0]} frames × 7)")
    except Exception as exc:
        logger.warning(f"Stage 3 — failed to export camera path: {exc}")

    return beta_path, theta_path, trans_path, contact_path, cam_path


# ─────────────────────────────────────────────────────────────────────────────
# Sync-metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_video_metadata(video_path):
    """
    Read basic video properties via OpenCV.
    Returns a dict with fps, frame_count, width, height, or None if the file
    cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps":         fps if fps > 0 else None,
        "frame_count": frame_count,
        "width":       width,
        "height":      height,
    }


def count_csv_rows(csv_path):
    """Count non-blank data rows in a headerless CSV file."""
    if csv_path is None or not osp.exists(csv_path):
        return None
    try:
        with open(csv_path, 'r') as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return None


def write_sync_metadata(source_video, processed_video, run_output_dir,
                        theta_path, trans_path,
                        contact_path=None, cam_path=None):
    """
    Write outputs/<video_name>/metadata.json so the Motion Viewer can align
    video frames 1-to-1 with SMPL reconstruction frames.

    Frame-count mismatches between the CSV files and the processed video are
    non-fatal: a 'warnings' key is added to the JSON so the viewer can surface
    the discrepancy to the user.  WHAM can legitimately drop frames during
    detection if the tracker loses the subject.
    """
    src_meta  = get_video_metadata(source_video)
    proc_meta = get_video_metadata(processed_video)

    source_fps            = src_meta["fps"]          if src_meta  else None
    source_frame_count    = src_meta["frame_count"]  if src_meta  else None
    processed_fps         = proc_meta["fps"]         if proc_meta else None
    processed_frame_count = proc_meta["frame_count"] if proc_meta else None

    # Prefer processed FPS → source FPS → hard-coded fallback.
    if processed_fps and processed_fps > 0:
        export_fps = processed_fps
    elif source_fps and source_fps > 0:
        export_fps = source_fps
    else:
        export_fps = 30.0

    duration_seconds = (
        source_frame_count / source_fps
        if (source_fps and source_fps > 0 and source_frame_count) else None
    )

    theta_frame_count   = count_csv_rows(theta_path)
    trans_frame_count   = count_csv_rows(trans_path)
    contact_frame_count = count_csv_rows(contact_path)
    cam_frame_count     = count_csv_rows(cam_path)

    warnings = []

    if theta_frame_count is not None and processed_frame_count is not None:
        if theta_frame_count != processed_frame_count:
            msg = (
                f"Frame count mismatch: thetas.csv has {theta_frame_count} frames "
                f"but processed video has {processed_frame_count} frames. "
                f"WHAM may have dropped frames during tracking."
            )
            warnings.append(msg)
            logger.warning(f"metadata.json — {msg}")

    if trans_frame_count is not None and theta_frame_count is not None:
        if trans_frame_count != theta_frame_count:
            msg = (
                f"Frame count mismatch: trans.csv has {trans_frame_count} frames "
                f"but thetas.csv has {theta_frame_count} frames."
            )
            warnings.append(msg)
            logger.warning(f"metadata.json — {msg}")

    metadata = {
        "source_video":            str(source_video),
        "processed_video":         osp.basename(str(processed_video)),
        "source_fps":              source_fps,
        "processed_fps":           processed_fps,
        "export_fps":              export_fps,
        "source_frame_count":      source_frame_count,
        "processed_frame_count":   processed_frame_count,
        "theta_frame_count":       theta_frame_count,
        "trans_frame_count":       trans_frame_count,
        "contact_frame_count":     contact_frame_count,
        "camera_path_frame_count": cam_frame_count,
        "duration_seconds":        duration_seconds,
        "sync_mode":               "frame_index",
        "notes": (
            "Each row in thetas.csv corresponds to one processed video frame "
            "unless WHAM dropped frames during tracking."
        ),
    }
    if warnings:
        metadata["warnings"] = warnings

    meta_path = osp.join(run_output_dir, 'metadata.json')
    with open(meta_path, 'w') as fh:
        json.dump(metadata, fh, indent=4)
    logger.info(f"Stage 3 — metadata.json  →  {meta_path}")
    return meta_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess → WHAM → Extract beta/theta',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input',           type=str, required=True,
                        help='Input video path  (e.g. inputs/sprint.mp4)')
    parser.add_argument('--output_dir',      type=str, default='outputs',
                        help='Root output directory  (default: outputs/)')
    parser.add_argument('--calib',           type=str, default=None,
                        help='Camera calibration file  (auto-estimated if omitted)')
    parser.add_argument('--visualize',       action='store_true',
                        help='Render 3D mesh overlay video  (requires pytorch3d)')
    parser.add_argument('--run_smplify',     action='store_true',
                        help='Temporal SMPLify post-processing  (slower, more accurate)')
    parser.add_argument('--local_only',      action='store_true',
                        help='Skip global trajectory  (faster, no DPVO SLAM)')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip Stage 1 and reuse an existing preprocessed video')
    # Internal flags used by the server / subprocess modes (hidden from --help)
    parser.add_argument('--_stage1_only', type=str, default=None,
                        help=argparse.SUPPRESS, metavar='OUTPUT_PATH')
    parser.add_argument('--progress_file', type=str, default=None,
                        help=argparse.SUPPRESS, metavar='PROGRESS_JSON')
    args = parser.parse_args()

    if args.progress_file:
        globals()['_PROGRESS_FILE'] = args.progress_file

    # Subprocess entry-point: run Stage 1 in isolation and exit.
    # Stage 1 is launched as a child process so its CUDA context is fully
    # torn down before Stage 2 initialises.  Without isolation, kornia's
    # GPU pipeline leaves state that triggers an illegal-memory-access when
    # ultralytics later calls model.fuse() → Conv2d(...).to(device).
    if args._stage1_only is not None:
        preprocess_video(args.input, args._stage1_only, progress_path=args.progress_file)
        sys.exit(0)

    t_pipeline_start = time.perf_counter()

    # ── Resolve output paths ────────────────────────────────────────────────
    video_name      = osp.splitext(osp.basename(args.input))[0]
    run_output_dir  = osp.join(args.output_dir, video_name)
    processed_video = osp.join(run_output_dir, f'{video_name}_processed.mp4')
    wham_out_dir    = osp.join(run_output_dir, 'wham', f'{video_name}_processed')
    pkl_path        = osp.join(wham_out_dir, 'wham_output.pkl')

    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(wham_out_dir,   exist_ok=True)

    # ── Stage 1: video preprocessing ───────────────────────────────────────
    if args.skip_preprocess:
        assert osp.exists(processed_video), (
            f"--skip_preprocess set but preprocessed video not found:\n  {processed_video}"
        )
        logger.info(f"Stage 1 — Skipped  (using {processed_video})")
    else:
        logger.info("Stage 1 — launching isolated preprocessing subprocess …")
        wham_dir   = osp.dirname(osp.abspath(__file__))
        stage1_cmd = [
            sys.executable, osp.abspath(__file__),
            '--input',        osp.abspath(args.input),
            '--_stage1_only', osp.abspath(processed_video),
        ]
        if args.progress_file:
            stage1_cmd += ['--progress_file', osp.abspath(args.progress_file)]
        subprocess.run(
            stage1_cmd,
            check=True,
            cwd=wham_dir,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        logger.info("Stage 1 — subprocess complete")

    # ── Stage 2: WHAM inference ─────────────────────────────────────────────
    logger.info("Stage 2 — Loading WHAM models …")

    # Empty-string CUDA_VISIBLE_DEVICES silently hides all GPUs from PyTorch.
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        del os.environ['CUDA_VISIBLE_DEVICES']

    if not torch.cuda.is_available():
        logger.error(
            "No CUDA GPU detected — WHAM inference requires a GPU.\n"
            "  Diagnostics:\n"
            "    nvidia-smi                   # check driver status\n"
            "    sudo modprobe nvidia         # reload kernel module if missing\n"
            "    unset CUDA_VISIBLE_DEVICES   # clear if accidentally set to ''\n"
            "  A reboot is often needed after a kernel update or driver crash."
        )
        sys.exit(1)

    gpu_props = torch.cuda.get_device_properties('cuda')
    logger.info(f"GPU: {gpu_props.name}  ({gpu_props.total_memory // 1024**2} MB)")

    from configs.config import get_cfg_defaults
    from lib.models import build_network, build_body_model

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')

    smpl    = build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
    network = build_network(cfg, smpl)
    network.eval()

    _write_progress({'stage': 1, 'stage_name': 'Stage 1 complete, loading WHAM models',
                     'pct': 25, 'status': 'running'})

    run_wham(cfg, processed_video, wham_out_dir, network,
             calib=args.calib,
             run_global=not args.local_only,
             visualize=args.visualize,
             run_smplify=args.run_smplify)

    # ── Stage 3: parameter extraction ───────────────────────────────────────
    _write_progress({'stage': 3, 'stage_name': 'Extracting parameters',
                     'pct': 82, 'status': 'running'})

    _, theta_path, trans_path, contact_path, cam_path = extract_params(
        pkl_path, run_output_dir,
        slam_path=osp.join(wham_out_dir, 'slam_results.pth'),
    )

    _write_progress({'stage': 3, 'stage_name': 'Writing metadata',
                     'pct': 95, 'status': 'running'})

    meta_path = write_sync_metadata(
        source_video=osp.abspath(args.input),
        processed_video=processed_video,
        run_output_dir=run_output_dir,
        theta_path=theta_path,
        trans_path=trans_path,
        contact_path=contact_path,
        cam_path=cam_path,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed         = time.perf_counter() - t_pipeline_start
    minutes, seconds = divmod(elapsed, 60)
    logger.info("")
    logger.info("Pipeline complete")
    logger.info(f"  Total time          →  {int(minutes)}m {seconds:.1f}s")
    logger.info(f"  Preprocessed video  →  {processed_video}")
    logger.info(f"  WHAM pkl            →  {pkl_path}")
    logger.info(f"  Beta parameters     →  {osp.join(run_output_dir, 'betas.json')}")
    logger.info(f"  Theta parameters    →  {theta_path}")
    logger.info(f"  World translation   →  {trans_path}")
    logger.info(f"  Sync metadata       →  {meta_path}")
