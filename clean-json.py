#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd


def _to_numeric(series):
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return series


def build_bad_sets(df: pd.DataFrame,
                   pose_dir: Optional[Path],
                   max_len: Optional[int]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns three sets of pose identifiers we consider 'bad':
      - bad_abs: absolute posix paths (as in the CSV)
      - bad_rel: paths relative to pose_dir (primary matching key)
      - bad_base: just basenames (fallback)
    """
    df = df.copy()
    # Normalize columns to str where relevant
    for col in ("pose_path", "status", "reason"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    status = df["status"].fillna("ok") if "status" in df.columns else "ok"
    bad_mask = (status == "bad")

    if max_len is not None and "duration" in df.columns:
        dur = _to_numeric(df["duration"])
        bad_mask = bad_mask | (dur > max_len)

    bad_df = df[bad_mask].copy()

    bad_abs: Set[str] = set()
    bad_rel: Set[str] = set()
    bad_base: Set[str] = set()

    for p in bad_df["pose_path"].astype(str):
        p_abs = Path(p)
        bad_abs.add(p_abs.as_posix())
        bad_base.add(p_abs.name)

        # Try to build a relative path against pose_dir
        if pose_dir is not None:
            try:
                rel = p_abs.resolve().relative_to(pose_dir.resolve()).as_posix()
                bad_rel.add(rel)
            except Exception:
                # Fallback: store common suffixes for suffix-matching
                parts = p_abs.parts
                if len(parts) >= 2:
                    bad_rel.add("/".join(parts[-2:]))
                if len(parts) >= 3:
                    bad_rel.add("/".join(parts[-3:]))

    return bad_abs, bad_rel, bad_base


def is_bad_pose(json_pose: str,
                bad_abs: Set[str],
                bad_rel: Set[str],
                bad_base: Set[str]) -> bool:
    if not json_pose:
        return True
    jp = Path(json_pose).as_posix()

    # direct matches
    if jp in bad_rel or jp in bad_abs:
        return True

    # strip leading './'
    if jp.startswith("./") and jp[2:] in bad_rel:
        return True

    # suffix matches against any rel entry (handles nested dirs)
    for b in bad_rel:
        if jp.endswith(b):
            return True

    # basename fallback
    if Path(jp).name in bad_base:
        return True

    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Original dataset JSON (with 'pose' or 'pose_path' key)")
    ap.add_argument("--csv", required=True, help="Validator CSV (yt_asl_validation.csv)")
    ap.add_argument("--out", required=True, help="Output cleaned JSON path")
    ap.add_argument("--pose-dir", default=None, help="Root used during validation to normalize absolute pose paths")
    ap.add_argument("--max-len", type=int, default=None, help="Drop clips with duration > max-len frames")
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir).resolve() if args.pose_dir else None

    # Read CSV robustly (fix DtypeWarning)
    df = pd.read_csv(args.csv, low_memory=False, dtype=str)
    if "duration" in df.columns:
        df["duration"] = _to_numeric(df["duration"])

    bad_abs, bad_rel, bad_base = build_bad_sets(df, pose_dir, args.max_len)

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))

    kept = []
    removed = 0
    missing_pose_key = 0

    for item in data:
        jp = item.get("pose") or item.get("pose_path")
        if not jp:
            missing_pose_key += 1
            removed += 1
            continue
        if is_bad_pose(jp, bad_abs, bad_rel, bad_base):
            removed += 1
            continue
        kept.append(item)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"kept={len(kept)} removed={removed} (missing_pose_key={missing_pose_key})")
    print(f"bad_abs={len(bad_abs)} bad_rel={len(bad_rel)} bad_base={len(bad_base)}")


if __name__ == "__main__":
    main()
