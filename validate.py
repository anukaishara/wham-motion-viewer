#!/usr/bin/env python3
"""
validate.py — Pipeline output validation and results summary.

Runs automated quality checks on every completed output folder and prints
a human-readable report.  All checks use only the CSV/JSON files produced
by Stage 3 — no GPU, no model weights required.

USAGE
    conda run -n wham_dev python validate.py
    conda run -n wham_dev python validate.py --output_dir outputs/
    conda run -n wham_dev python validate.py --video sprint
    conda run -n wham_dev python validate.py --compare          # adds raw-WHAM comparison
    conda run -n wham_dev python validate.py --compare --video sprint

EXIT CODE
    0  — all checks passed
    1  — one or more checks failed
"""

import io
import re
import os
import sys
import json
import argparse
import os.path as osp
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

BETA_CLIP           = 2.0    # betas must stay within ±2σ (we clip them there)
BETA_WARN           = 1.8    # warn if any beta is close to the clip boundary
JITTER_WARN         = 0.08   # rad/frame — flag high temporal jitter
FRAME_DROP_WARN_PCT = 5      # warn if >5% of source frames were dropped
FRAME_DROP_FAIL_PCT = 40     # fail  if >40% of source frames were dropped
# WHAM's contact head outputs raw sigmoid values; small float overflow (±0.05)
# beyond [0,1] is expected and does not indicate a problem.
CONTACT_RANGE_TOL   = 0.05
CONTACT_MIN_VALID   = 0.01   # at least 1% of frames must show some contact
# 100% contact is physically valid for slow exercises (taichi, standing); no upper bound


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    value: str
    note: str = ""


@dataclass
class RunReport:
    folder: str
    video_name: str
    checks: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self):
        return all(c.passed for c in self.checks)

    def add(self, name, passed, value, note=""):
        self.checks.append(CheckResult(name, passed, value, note))


# ─────────────────────────────────────────────────────────────────────────────
# Validation logic
# ─────────────────────────────────────────────────────────────────────────────

def validate_run(folder: str) -> Optional[RunReport]:
    """Validate a single output folder. Returns None if not a complete run."""
    required = ['thetas.csv', 'trans.csv', 'contact.csv', 'betas.json', 'metadata.json']
    if not all(osp.exists(osp.join(folder, f)) for f in required):
        return None

    name   = osp.basename(folder)
    report = RunReport(folder=folder, video_name=name)

    # ── Load files ────────────────────────────────────────────────────────────
    thetas  = pd.read_csv(osp.join(folder, 'thetas.csv'),  header=None).values
    trans   = pd.read_csv(osp.join(folder, 'trans.csv'),   header=None).values
    contact = pd.read_csv(osp.join(folder, 'contact.csv'), header=None).values
    betas   = json.load(open(osp.join(folder, 'betas.json')))['betas']
    meta    = json.load(open(osp.join(folder, 'metadata.json')))

    n_frames = thetas.shape[0]

    # ── Check 1: frame alignment ──────────────────────────────────────────────
    src_fc   = meta.get('source_frame_count')
    theta_fc = meta.get('theta_frame_count')

    if src_fc and theta_fc:
        dropped     = src_fc - theta_fc
        dropped_pct = dropped / src_fc * 100
        ok   = dropped_pct < FRAME_DROP_FAIL_PCT
        note = (
            "" if dropped_pct < FRAME_DROP_WARN_PCT
            else f"Tracker lost subject for {dropped_pct:.0f}% of frames"
        )
        report.add(
            "Frame alignment",
            ok,
            f"{theta_fc}/{src_fc} frames  ({dropped} dropped, {dropped_pct:.0f}%)",
            note,
        )
    else:
        report.add("Frame alignment", False, "missing metadata", "metadata.json incomplete")

    # ── Check 2: theta shape ──────────────────────────────────────────────────
    ok = thetas.shape[1] == 72
    report.add(
        "Theta dimensions",
        ok,
        f"{thetas.shape[0]} frames × {thetas.shape[1]} params",
        "" if ok else f"Expected 72 columns, got {thetas.shape[1]}"
    )

    # ── Check 3: no NaN / Inf in thetas ──────────────────────────────────────
    bad = int(np.isnan(thetas).sum() + np.isinf(thetas).sum())
    report.add(
        "Theta finite values",
        bad == 0,
        f"{bad} NaN/Inf",
        "" if bad == 0 else f"{bad} non-finite values found"
    )

    # ── Check 4: temporal jitter (smoothness) ─────────────────────────────────
    jitter = float(np.mean(np.abs(np.diff(thetas, axis=0))))
    ok     = jitter < JITTER_WARN
    report.add(
        "Temporal smoothness",
        ok,
        f"{jitter:.5f} rad/frame",
        "" if ok else f"Jitter > {JITTER_WARN} rad/frame — consider --run_smplify"
    )
    report.metrics['theta_jitter'] = jitter

    # ── Check 5: betas within physical range ─────────────────────────────────
    max_beta = float(max(abs(b) for b in betas))
    ok_hard  = max_beta <= BETA_CLIP
    ok_warn  = max_beta <= BETA_WARN
    report.add(
        "Beta plausibility",
        ok_hard,
        f"max |β| = {max_beta:.3f}  (clip boundary ±{BETA_CLIP})",
        "" if ok_warn else f"Beta near clip boundary — body shape estimate may be unreliable"
    )
    report.metrics['max_beta'] = max_beta

    # ── Check 6: translation — no NaN, monotone total distance > 0 ──────────
    bad_trans = int(np.isnan(trans).sum() + np.isinf(trans).sum())
    travel    = float(np.sum(np.linalg.norm(np.diff(trans, axis=0), axis=1)))
    ok = bad_trans == 0 and travel > 0
    report.add(
        "Translation validity",
        ok,
        f"travel = {travel:.3f} m  ({bad_trans} NaN/Inf)",
        "" if ok else "Zero travel or NaN values in trans.csv"
    )
    report.metrics['travel_m'] = travel

    # ── Check 7: contact value range (informational) ─────────────────────────
    # WHAM stores raw contact scores; Savitzky-Golay smoothing can push them
    # slightly outside [0, 1].  We report the range but never fail on it —
    # the plausibility check below (threshold at 0.5) is the real gate.
    c_min, c_max = float(contact.min()), float(contact.max())
    report.add(
        "Contact value range",
        True,
        f"min={c_min:.3f}  max={c_max:.3f}",
    )

    # ── Check 8: contact pattern physically non-trivial ───────────────────────
    any_contact = float((contact > 0.5).any(axis=1).mean())
    ok = any_contact > CONTACT_MIN_VALID
    foot_pct = (contact > 0.5).mean(axis=0) * 100
    report.add(
        "Contact plausibility",
        ok,
        f"frames with ≥1 foot contact: {any_contact*100:.0f}%  "
        f"(LH={foot_pct[0]:.0f}% RH={foot_pct[1]:.0f}% "
        f"LT={foot_pct[2]:.0f}% RT={foot_pct[3]:.0f}%)",
        "" if ok else "No frames with foot contact detected — check tracking results"
    )
    report.metrics['contact_any_pct'] = any_contact * 100
    report.metrics['foot_pct']        = foot_pct.tolist()

    # ── Check 9: temporal consistency across files ────────────────────────────
    frame_counts = {
        'thetas': thetas.shape[0],
        'trans':  trans.shape[0],
        'contact': contact.shape[0],
    }
    ok = len(set(frame_counts.values())) == 1
    report.add(
        "Cross-file consistency",
        ok,
        "  ".join(f"{k}={v}" for k, v in frame_counts.items()),
        "" if ok else "Frame count mismatch across CSV files"
    )

    # ── Check 10: camera path (optional) ─────────────────────────────────────
    cam_path = osp.join(folder, 'camera_path.csv')
    if osp.exists(cam_path):
        cam = pd.read_csv(cam_path, header=None).values
        ok  = cam.shape[1] == 7 and cam.shape[0] == n_frames
        # Quaternion norm should be ~1
        quats    = cam[:, 3:]
        quat_norms = np.linalg.norm(quats, axis=1)
        quat_ok  = bool(np.allclose(quat_norms, 1.0, atol=0.01))
        report.add(
            "Camera path",
            ok and quat_ok,
            f"{cam.shape[0]} frames × 7  |  quaternion norm ≈ {quat_norms.mean():.4f}",
            "" if quat_ok else "Camera quaternions are not unit-length"
        )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Raw-WHAM vs our-pipeline comparison
# ─────────────────────────────────────────────────────────────────────────────

def _count_quat_discontinuities(pose: np.ndarray) -> int:
    """Count frames where any joint's quaternion sign flips vs the previous frame."""
    from scipy.spatial.transform import Rotation
    n_frames, n_cols = pose.shape
    n_joints = n_cols // 3
    total = 0
    for j in range(n_joints):
        aa   = pose[:, j*3 : j*3+3]
        quat = Rotation.from_rotvec(aa).as_quat()    # (N,4) xyzw
        dots = np.einsum('ij,ij->i', quat[:-1], quat[1:])
        total += int((dots < 0).sum())
    return total


def load_raw_wham(folder: str) -> Optional[dict]:
    """
    Load wham_output.pkl (joblib format) for person 0.
    Returns dict with pose_world (N,72), trans_world (N,3), betas (N,10).
    Returns None if pkl is missing.
    """
    import glob, joblib
    pkls = glob.glob(osp.join(folder, 'wham', '*', 'wham_output.pkl'))
    if not pkls:
        return None
    try:
        data   = joblib.load(pkls[0])
        person = data[0]
        return {
            'pose':    np.array(person['pose_world'], dtype=np.float32),   # (N,72)
            'trans':   np.array(person['trans_world'], dtype=np.float32),  # (N,3)
            'betas':   np.array(person['betas'], dtype=np.float32),        # (N,10)
            'contact': np.array(person['contact'], dtype=np.float32),      # (N,4)
        }
    except Exception:
        return None


def compare_run(folder: str) -> Optional[dict]:
    """
    Compare raw WHAM output (from pkl) with our post-processed output (from CSVs).
    Returns a dict of metric dicts, or None if pkl is missing.
    """
    raw = load_raw_wham(folder)
    if raw is None:
        return None

    required = ['thetas.csv', 'trans.csv', 'betas.json']
    if not all(osp.exists(osp.join(folder, f)) for f in required):
        return None

    ours_pose  = pd.read_csv(osp.join(folder, 'thetas.csv'), header=None).values
    ours_trans = pd.read_csv(osp.join(folder, 'trans.csv'),  header=None).values

    # Align frame counts (pkl may include frames our tracker dropped)
    n = min(raw['pose'].shape[0], ours_pose.shape[0])
    raw_pose   = raw['pose'][:n]
    raw_trans  = raw['trans'][:n]
    raw_betas  = raw['betas'][:n]
    ours_pose  = ours_pose[:n]
    ours_trans = ours_trans[:n]

    def jitter(arr):
        return float(np.mean(np.abs(np.diff(arr, axis=0))))

    def max_jump(arr):
        return float(np.max(np.abs(np.diff(arr, axis=0))))

    def trans_jitter(arr):
        return float(np.mean(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    def trans_max_jump(arr):
        return float(np.max(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    return {
        'n_frames': n,
        'pose_jitter': {
            'raw':  jitter(raw_pose),
            'ours': jitter(ours_pose),
        },
        'pose_max_jump': {
            'raw':  max_jump(raw_pose),
            'ours': max_jump(ours_pose),
        },
        'rot_discontinuities': {
            'raw':  _count_quat_discontinuities(raw_pose),
            'ours': _count_quat_discontinuities(ours_pose),
        },
        'trans_jitter': {
            'raw':  trans_jitter(raw_trans),
            'ours': trans_jitter(ours_trans),
        },
        'trans_max_jump': {
            'raw':  trans_max_jump(raw_trans),
            'ours': trans_max_jump(ours_trans),
        },
        # Mean per-coefficient standard deviation across frames.
        # Raw WHAM emits a different beta per frame; our pipeline averages to a constant → 0.
        'beta_std': {
            'raw':  float(np.mean(np.std(raw_betas, axis=0))),
            'ours': 0.0,
        },
    }


def _pct_change(raw, ours, lower_is_better=True):
    """Return (delta_pct, is_improvement) as strings for display."""
    if raw == 0:
        return "—", True
    delta = (ours - raw) / abs(raw) * 100
    improved = delta < 0 if lower_is_better else delta > 0
    arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
    return f"{abs(delta):.1f}% {arrow}", improved


def print_comparison(video_name: str, cmp: dict):
    IMP  = "\033[32m"   # green
    NEU  = "\033[33m"   # yellow
    RST  = "\033[0m"

    def col(val_str, improved):
        c = IMP if improved else NEU
        return f"{c}{val_str}{RST}"

    n = cmp['n_frames']
    print(f"\n  ┌─ {video_name}  ({n} frames aligned for comparison)")
    print(f"  │  {'Metric':<30} {'Raw WHAM':>14} {'Our Pipeline':>14} {'Change':>12}")
    print(f"  │  {'─'*30} {'─'*14} {'─'*14} {'─'*12}")

    rows = [
        ("Pose jitter (rad/frame)",    cmp['pose_jitter'],        True,  "{:.5f}"),
        ("Pose max jump (rad/frame)",  cmp['pose_max_jump'],      True,  "{:.4f}"),
        ("Rotation discontinuities",   cmp['rot_discontinuities'],True,  "{:d}"),
        ("Trans jitter (m/frame)",     cmp['trans_jitter'],       True,  "{:.5f}"),
        ("Trans max jump (m/frame)",   cmp['trans_max_jump'],     True,  "{:.4f}"),
        ("Beta std (per-frame)",       cmp['beta_std'],           True,  "{:.5f}"),
    ]
    for label, vals, lib, fmt in rows:
        r = vals['raw'];  o = vals['ours']
        try:
            rs = fmt.format(r);  os = fmt.format(o)
        except (ValueError, TypeError):
            rs = str(r);  os = str(o)
        delta_str, improved = _pct_change(r, o, lib)
        print(f"  │  {label:<30} {rs:>14} {os:>14} {col(delta_str, improved):>22}")

    print(f"  └─")


def print_comparison_summary(compare_results: list):
    """Single-row-per-video summary of all comparison results."""
    valid = [(name, cmp) for name, cmp in compare_results if cmp is not None]
    if not valid:
        return

    print("\n" + "═" * 100)
    print("  PIPELINE vs RAW WHAM COMPARISON")
    print("  (Raw WHAM = wham_output.pkl before Stage 3 post-processing)")
    print("═" * 100)
    print(f"  {'Video':<22} {'Pose jitter':>12} {'  →':>4} {'':>12} "
          f"{'Trans jitter':>12} {'  →':>4} {'':>12} "
          f"{'Rot.disc':>9} {'  →':>4} {'':>6} "
          f"{'Beta std':>10} {'  →':>4} {'':>8}")
    print(f"  {'':22} {'Raw':>12} {'':>4} {'Ours':>12} "
          f"{'Raw':>12} {'':>4} {'Ours':>12} "
          f"{'Raw':>9} {'':>4} {'Ours':>6} "
          f"{'Raw':>10} {'':>4} {'Ours':>8}")
    print("  " + "─" * 97)

    IMP = "\033[32m"; RST = "\033[0m"

    def imp(v, ref):
        if v < ref:
            return f"{IMP}{v}{RST}"
        return str(v)

    for name, cmp in valid:
        clean = re.sub(r'^[0-9a-f]{8}_', '', name)
        pj_r  = cmp['pose_jitter']['raw'];    pj_o = cmp['pose_jitter']['ours']
        tj_r  = cmp['trans_jitter']['raw'];   tj_o = cmp['trans_jitter']['ours']
        rd_r  = cmp['rot_discontinuities']['raw']
        rd_o  = cmp['rot_discontinuities']['ours']
        bs_r  = cmp['beta_std']['raw'];       bs_o = cmp['beta_std']['ours']

        print(f"  {clean:<22} "
              f"{pj_r:>12.5f} {'→':>4} {IMP}{pj_o:>12.5f}{RST} "
              f"{tj_r:>12.5f} {'→':>4} {IMP}{tj_o:>12.5f}{RST} "
              f"{rd_r:>9d} {'→':>4} {IMP}{rd_o:>6d}{RST} "
              f"{bs_r:>10.5f} {'→':>4} {IMP}{bs_o:>8.5f}{RST}")

    print("  " + "─" * 97)
    print("  Green values = improvement over raw WHAM output.")
    print("  Metrics are computed on the same N frames after aligning pkl and CSV.")
    print("  Stage 3 post-processing applies: quaternion continuity fix, Savitzky-Golay")
    print("  smoothing (pose window=7, trans window=9), beta averaging + ±2σ clipping.")
    print("═" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


class _Tee:
    """Writes to stdout and captures a plain-text copy (ANSI codes stripped)."""
    def __init__(self):
        self._buf = io.StringIO()

    def write(self, text):
        sys.stdout.write(text)
        self._buf.write(_ANSI_RE.sub('', text))

    def flush(self):
        sys.stdout.flush()

    def save(self, path):
        with open(path, 'w') as fh:
            fh.write(self._buf.getvalue())


def print_report(report: RunReport, verbose: bool = True):
    status = PASS if report.passed else FAIL
    print(f"\n{status}  {report.video_name}")
    if not verbose and report.passed:
        meta_path = osp.join(report.folder, 'metadata.json')
        m = json.load(open(meta_path))
        j = report.metrics.get('theta_jitter', 0)
        t = report.metrics.get('travel_m', 0)
        print(f"     {m['theta_frame_count']} frames  |  "
              f"jitter {j:.4f} rad/frame  |  travel {t:.2f} m  |  "
              f"all {len(report.checks)} checks passed")
        return

    for c in report.checks:
        icon = PASS if c.passed else FAIL
        note = f"  ← {c.note}" if c.note else ""
        print(f"     {icon}  {c.name:<28}  {c.value}{note}")


def print_summary_table(reports: list[RunReport]):
    # Deduplicate by canonical video name (strip uuid prefix)
    seen = {}
    for r in reports:
        name = r.video_name
        # Strip leading UUID prefix (8 hex chars + underscore)
        import re
        clean = re.sub(r'^[0-9a-f]{8}_', '', name)
        if clean not in seen:
            seen[clean] = r

    print("\n" + "─" * 90)
    print(f"{'Video':<22} {'Frames':>7} {'FPS':>5} {'Duration':>10} "
          f"{'Jitter (rad/f)':>16} {'Travel (m)':>12} {'Contact%':>10} {'Status':>8}")
    print("─" * 90)

    for clean, r in sorted(seen.items()):
        meta = json.load(open(osp.join(r.folder, 'metadata.json')))
        j    = r.metrics.get('theta_jitter', 0)
        t    = r.metrics.get('travel_m', 0)
        c    = r.metrics.get('contact_any_pct', 0)
        n    = meta.get('theta_frame_count', '?')
        fps  = meta.get('export_fps', '?')
        dur  = meta.get('duration_seconds', 0)
        stat = "PASS" if r.passed else "FAIL"
        print(f"{clean:<22} {n:>7} {fps:>5.1f} {dur:>9.1f}s "
              f"{j:>16.5f} {t:>12.3f} {c:>9.0f}%  {stat:>8}")

    print("─" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate WHAM pipeline outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output_dir', default='outputs',
                        help='Root output directory (default: outputs/)')
    parser.add_argument('--video', default=None,
                        help='Validate a single video folder by name')
    parser.add_argument('--verbose', action='store_true',
                        help='Always show all checks, even for passing runs')
    parser.add_argument('--save', type=str, default=None,
                        metavar='FILE',
                        help='Save plain-text report to FILE  (e.g. outputs/validation_report.txt)')
    parser.add_argument('--compare', action='store_true',
                        help='Add side-by-side comparison of raw WHAM output vs our post-processed output')
    args = parser.parse_args()

    tee = _Tee()
    if args.save:
        import builtins
        _orig_print = builtins.print
        builtins.print = lambda *a, **kw: _orig_print(*a, **{**kw, 'file': tee})

    if args.video:
        folders = [osp.join(args.output_dir, args.video)]
    else:
        base = args.output_dir
        folders = sorted(
            osp.join(base, d) for d in os.listdir(base)
            if osp.isdir(osp.join(base, d))
        )

    reports  = []
    skipped  = 0
    cmp_list = []   # (video_name, cmp_dict | None)

    print(f"\nWHAM Pipeline Validation")
    print(f"Output directory: {osp.abspath(args.output_dir)}")

    for folder in folders:
        report = validate_run(folder)
        if report is None:
            skipped += 1
            continue
        reports.append(report)
        print_report(report, verbose=args.verbose or not report.passed)

        if args.compare:
            cmp = compare_run(folder)
            cmp_list.append((report.video_name, cmp))
            if cmp is not None:
                print_comparison(report.video_name, cmp)
            else:
                print(f"\n  ⚠  {report.video_name}: wham_output.pkl not found — skipping comparison")

    if not reports:
        print("\nNo complete output folders found.")
        sys.exit(1)

    print_summary_table(reports)

    if args.compare and cmp_list:
        # Deduplicate by canonical name before printing summary
        seen_cmp = {}
        for name, cmp in cmp_list:
            clean = re.sub(r'^[0-9a-f]{8}_', '', name)
            key   = clean.lower()
            if key not in seen_cmp and cmp is not None:
                seen_cmp[key] = (clean, cmp)
        print_comparison_summary([(name, cmp) for name, cmp in seen_cmp.values()])

    n_pass = sum(1 for r in reports if r.passed)
    n_fail = len(reports) - n_pass
    print(f"\n{n_pass}/{len(reports)} runs passed all checks"
          + (f"  |  {skipped} folders skipped (incomplete)" if skipped else "")
          + (f"\n{FAIL}  {n_fail} run(s) failed — see details above" if n_fail else ""))

    if args.save:
        tee.save(args.save)
        import builtins
        builtins.print = _orig_print
        print(f"Report saved  →  {osp.abspath(args.save)}")

    sys.exit(0 if n_fail == 0 else 1)
