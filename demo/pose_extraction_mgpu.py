import os, glob, pickle, argparse, threading, queue, time, json, subprocess, traceback
from multiprocessing import Process, Lock
import numpy as np
import cv2
import av 
from rtmlib import Wholebody


def iter_frames_opencv(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"fail to open: {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()

def iter_frames_pyav(path):
    # Robust CPU decode for anything FFmpeg supports, incl. AV1 (libdav1d)
    with av.open(path, mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                img = frame.to_ndarray(format="rgb24")   # RGB uint8
                yield img[..., ::-1]                     # RGB -> BGR

def prefetch_frames(frame_iter, maxsize=32):
    """Single producer thread to overlap decode/IO with inference."""
    q = queue.Queue(maxsize=maxsize)
    sentinel = object()
    def _worker():
        try:
            for f in frame_iter:
                q.put(f)
        except Exception as e:
            q.put((sentinel, e))
            return
        q.put((sentinel, None))
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if isinstance(item, tuple) and item and item[0] is sentinel:
            if item[1] is not None:
                raise item[1]
            break
        yield item

def safe_iter_frames(path):
    """
    Try OpenCV first (fast for H.264/H.265). If it errors (e.g., AV1),
    fall back to PyAV (libdav1d on CPU) — reliable.
    """
    tried_opencv_exc = None
    try:
        for f in iter_frames_opencv(path):
            yield f
        return
    except Exception as e:
        tried_opencv_exc = e
    # Fallback path
    try:
        for f in iter_frames_pyav(path):
            yield f
    except Exception as e2:
        # Re-raise with both error messages for better logging
        raise RuntimeError(f"OpenCV decode failed: {tried_opencv_exc}; PyAV failed: {e2}")

# ---------------------- Metadata & logging ----------------------

def ffprobe_codec(path):
    """Return codec name via ffprobe or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=codec_name","-of","json", path],
            stderr=subprocess.STDOUT
        )
        j = json.loads(out.decode("utf-8"))
        return j["streams"][0]["codec_name"]
    except Exception:
        return "unknown"

def append_jsonl(path, obj, lock: Lock):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ---------------------- Core processing ----------------------

def process_video(video_path, tgt_dir, wholebody, overwrite=False, gpu_id=0, log_codec=False):
    """
    Returns (success: bool, meta: dict)
    meta (success): {path, frames, fps, duration_s, width, height, gpu, codec}
    meta (fail):    {path, error, gpu, codec, trace}
    """
    out_pkl = os.path.join(tgt_dir, os.path.basename(video_path).rsplit(".", 1)[0] + ".pkl")
    if os.path.exists(out_pkl) and not overwrite:
        return True, {"path": video_path, "skipped": True, "reason": "exists"}

    codec = ffprobe_codec(video_path) if log_codec else None

    # Prefetch to hide decode latency
    frame_iter = prefetch_frames(safe_iter_frames(video_path), maxsize=32)

    data = {"keypoints": [], "scores": []}
    frames = 0
    t0 = time.time()
    width = height = None

    try:
        for frame in frame_iter:
            frame = np.uint8(frame)
            if height is None:
                height, width = frame.shape[:2]
            keypoints, scores = wholebody(frame)
            H, W = frame.shape[:2]
            data["keypoints"].append(keypoints / np.array([W, H])[None, None])
            data["scores"].append(scores)
            frames += 1
        t1 = time.time()
        with open(out_pkl, "wb") as f:
            pickle.dump(data, f, protocol=4)
        dur = max(1e-6, t1 - t0)
        fps = frames / dur if frames else 0.0
        return True, {
            "path": video_path, "frames": frames, "fps": round(fps, 2),
            "duration_s": round(dur, 3), "width": width, "height": height,
            "gpu": gpu_id, "codec": codec
        }
    except Exception as e:
        return False, {
            "path": video_path,
            "error": str(e),
            "trace": traceback.format_exc(limit=3),
            "gpu": gpu_id,
            "codec": codec
        }

def worker(gpu_id, files_for_this_worker, args, success_log, failed_log, lock):
    # Pin to a single GPU; rtmlib still wants device='cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    wholebody = Wholebody(
        to_openpose=args.openpose_skeleton,
        mode=args.mode,
        backend=args.backend,
        device="cuda" if args.device.startswith("cuda") else args.device,
    )
    print(f"[GPU{gpu_id}] started with {len(files_for_this_worker)} files", flush=True)

    processed = 0
    for video_path in files_for_this_worker:
        ok, meta = process_video(
            video_path=video_path,
            tgt_dir=args.tgt_dir,
            wholebody=wholebody,
            overwrite=args.overwrite,
            gpu_id=gpu_id,
            log_codec=True,
        )
        if ok:
            append_jsonl(success_log, meta, lock)
        else:
            append_jsonl(failed_log, meta, lock)

        processed += 1
        if processed % 50 == 0:
            print(f"[GPU{gpu_id}] processed {processed} files...", flush=True)

# ---------------------- Orchestration ----------------------

def build_file_list(src_dir, exts):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(src_dir, f"*.{ext}")))
    files.sort()
    return files

def shard_list(lst, n_shards):
    return [lst[i::n_shards] for i in range(n_shards)]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True, help="video dir path")
    parser.add_argument("--tgt_dir", required=True, help="pose dir path")

    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--openpose_skeleton", action="store_true", help="use openpose format")
    parser.add_argument("--mode", default="balanced", choices=["performance", "lightweight", "balanced"])

    parser.add_argument("--video_extensions", nargs='+', default=["mp4"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="comma-separated GPU IDs to use")
    parser.add_argument("--procs_per_gpu", type=int, default=1, help="use 2 if decode/IO bound and VRAM OK")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_dir", default=None, help="directory for success.jsonl / failed.jsonl (default: tgt_dir/logs)")

    args = parser.parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)

    all_videos = build_file_list(args.src_dir, args.video_extensions)
    print(f"found {len(all_videos)} videos", flush=True)

    pending = []
    for v in all_videos:
        out_pkl = os.path.join(args.tgt_dir, os.path.basename(v).rsplit(".",1)[0] + ".pkl")
        if not os.path.exists(out_pkl) or args.overwrite:
            pending.append(v)
    print(f"pending {len(pending)} videos", flush=True)
    if not pending:
        return

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    num_workers = max(1, len(gpu_ids) * max(1, args.procs_per_gpu))
    shards = shard_list(pending, num_workers)

    # Logging setup
    log_root = args.log_dir or os.path.join(args.tgt_dir, "logs")
    os.makedirs(log_root, exist_ok=True)
    success_log = os.path.join(log_root, "success.jsonl")
    failed_log  = os.path.join(log_root, "failed.jsonl")

    # process-safe lock for JSONL appends
    lock = Lock()

    procs = []
    idx = 0
    for gid in gpu_ids:
        for _ in range(args.procs_per_gpu):
            files_for_this_worker = shards[idx]
            p = Process(
                target=worker,
                args=(gid, files_for_this_worker, args, success_log, failed_log, lock),
                daemon=False
            )
            p.start()
            procs.append(p)
            idx += 1

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Interrupted, terminating workers...", flush=True)
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()

if __name__ == "__main__":
    main()
