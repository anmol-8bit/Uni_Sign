#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import gzip
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

EXPECTED_KP = 133

def load_pickle(path: Path) -> Any:
    # Support both .pkl and .pkl.gz
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)

def shape_of(arr) -> Tuple:
    try:
        return np.asarray(arr).shape
    except Exception:
        return tuple()

def check_pose_dict(pose: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dict of validation signals + a 'status' field ('ok' or 'bad')."""
    out = {
        "status": "ok",
        "reason": "",
        "duration": None,
        "keypoints_shape0": None,
        "scores_shape0": None,
        "num_frames_keypoints": None,
        "num_frames_scores": None,
        "any_nan": False,
        "max_conf": None,
        "min_conf": None,
        "all_zero_conf": False,
        "kp_elem_shape": None,
        "score_elem_shape": None,
    }
    # keys exist
    if not isinstance(pose, dict):
        out.update(status="bad", reason="pose_not_dict")
        return out
    if "keypoints" not in pose or "scores" not in pose:
        out.update(status="bad", reason="missing_keys")
        return out

    kps = pose["keypoints"]
    scs = pose["scores"]

    # lengths / duration
    try:
        n_k = len(kps)
        n_s = len(scs)
    except Exception:
        out.update(status="bad", reason="non_iterables")
        return out

    out["num_frames_keypoints"] = n_k
    out["num_frames_scores"] = n_s
    if n_k == 0 or n_s == 0:
        out.update(status="bad", reason="zero_length")
        return out
    if n_k != n_s:
        out.update(status="bad", reason=f"length_mismatch_k={n_k}_s={n_s}")
        return out

    out["duration"] = n_k

    # inspect first non-empty frame shapes
    first_k = None
    first_s = None
    for i in range(n_k):
        if kps[i] is not None and scs[i] is not None:
            first_k = np.asarray(kps[i])
            first_s = np.asarray(scs[i])
            break
    if first_k is None or first_s is None:
        out.update(status="bad", reason="all_frames_none")
        return out

    out["keypoints_shape0"] = shape_of(kps)
    out["scores_shape0"] = shape_of(scs)
    out["kp_elem_shape"] = tuple(first_k.shape)
    out["score_elem_shape"] = tuple(first_s.shape)

    # Expected shapes: (133,2) for keypoints; (133,) or (1,133) for scores
    kp_ok = (first_k.ndim == 2 and first_k.shape[-1] == 2 and first_k.shape[0] == EXPECTED_KP) or             (first_k.ndim == 3 and first_k.shape[0] == 1 and first_k.shape[1] == EXPECTED_KP and first_k.shape[2] == 2)
    sc_ok = (first_s.ndim == 1 and first_s.shape[0] == EXPECTED_KP) or             (first_s.ndim == 2 and ((first_s.shape[0] == 1 and first_s.shape[1] == EXPECTED_KP) or (first_s.shape[0] == EXPECTED_KP and first_s.shape[1] == 1)))

    if not kp_ok or not sc_ok:
        out.update(status="bad", reason=f"unexpected_elem_shapes_kp={out['kp_elem_shape']}_sc={out['score_elem_shape']}")
        return out

    # NaNs / conf stats
    try:
        sc_arr = np.asarray(scs)
        out["any_nan"] = bool(np.isnan(sc_arr).any() or np.isnan(np.asarray(kps)).any())
        with np.errstate(all="ignore"):
            out["max_conf"] = float(np.nanmax(sc_arr))
            out["min_conf"] = float(np.nanmin(sc_arr))
        out["all_zero_conf"] = bool(np.nanmax(sc_arr) <= 1e-8)
    except Exception as e:
        out.update(status="bad", reason=f"conf_stats_error:{type(e).__name__}")
        return out

    # Frame-wise sanity: count frames with nontrivial conf
    num_good_frames = 0
    for i in range(n_k):
        try:
            sfi = np.asarray(scs[i]).reshape(-1)
            if np.nanmax(sfi) > 0.05:  # threshold can be tuned
                num_good_frames += 1
        except Exception:
            pass
    if num_good_frames == 0:
        out.update(status="bad", reason="no_frames_above_conf_threshold")
        return out

    return out

def validate_pair(pose_path: Path, video_path: Optional[Path]) -> Dict[str, Any]:
    row = {
        "pose_path": str(pose_path),
        "video_path": str(video_path) if video_path else "",
        "video_exists": False,
        "pose_exists": pose_path.exists(),
    }
    if video_path is not None:
        row["video_exists"] = video_path.exists()

    if not row["pose_exists"]:
        row.update(status="bad", reason="pose_missing")
        return row

    try:
        pose = load_pickle(pose_path)
    except Exception as e:
        row.update(status="bad", reason=f"pose_load_error:{type(e).__name__}")
        return row

    pose_check = check_pose_dict(pose)
    row.update(pose_check)
    return row

def main():
    ap = argparse.ArgumentParser(description="Validate YT-ASL/CSL pose+rgb dataset integrity")
    ap.add_argument("--json", type=str, help="Path to dataset JSON (list of {video, pose, text,...})")
    ap.add_argument("--pose-dir", type=str, required=False, help="Base dir for pose files, joined with JSON 'pose' value")
    ap.add_argument("--rgb-dir", type=str, required=False, help="Base dir for videos, joined with JSON 'video' value")
    ap.add_argument("--limit", type=int, default=None, help="Validate at most N items")
    ap.add_argument("--skip-rgb", action="store_true", help="Do not check video existence")
    ap.add_argument("--dump-csv", type=str, default=None, help="Write a CSV report here")
    ap.add_argument("--single", action="store_true", help="Validate a single pose/video pair instead of a JSON list")
    ap.add_argument("--pose-file", type=str, help="Pose PKL path for --single mode")
    ap.add_argument("--video-file", type=str, help="Video MP4 path for --single mode (optional)")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []

    if args.single:
        if not args.pose_file:
            print("--pose-file is required in --single mode", file=sys.stderr)
            sys.exit(2)
        pose_path = Path(args.pose_file)
        video_path = Path(args.video_file) if args.video_file else None
        rows.append(validate_pair(pose_path, video_path))
    else:
        if not args.json:
            print("--json is required", file=sys.stderr)
            sys.exit(2)
        data = json.loads(Path(args.json).read_text(encoding="utf-8"))
        n = len(data) if args.limit is None else min(args.limit, len(data))
        for i in range(n):
            item = data[i]
            pose_rel = item.get("pose") or item.get("pose_path") or ""
            vid_rel  = item.get("video") or item.get("video_path") or ""

            pose_path = Path(pose_rel)
            video_path = Path(vid_rel) if vid_rel else None

            if args.pose_dir:
                pose_path = Path(args.pose_dir) / pose_path
            if (not args.skip_rgb) and args.rgb_dir and video_path:
                video_path = Path(args.rgb_dir) / video_path

            rows.append(validate_pair(pose_path, None if args.skip_rgb else video_path))

    df = pd.DataFrame(rows)
    # Summary
    total = len(df)
    bad = int((df.get("status") == "bad").sum()) if "status" in df else 0
    ok  = total - bad
    print(f"\nSummary: total={total}  ok={ok}  bad={bad}")
    if total:
        print("Top reasons (bad rows):")
        if "reason" in df:
            print(df[df.get("status") == "bad"]["reason"].value_counts().head(10))

    # Per-check quick stats
    for col in ["duration", "max_conf", "min_conf", "any_nan", "all_zero_conf"]:
        if col in df.columns:
            print(f"\n{col} stats:\n{df[col].describe(include='all')}")

    if args.dump_csv:
        outp = Path(args.dump_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"\nWrote CSV: {outp}")

if __name__ == "__main__":
    main()
