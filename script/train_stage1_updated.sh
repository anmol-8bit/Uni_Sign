#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# Stage-1 Uni-Sign Pretraining
# ------------------------------

# >>> Edit these if needed
GPUS="localhost:0,1,2,3,4,5,6,7"
MASTER_PORT="${MASTER_PORT:-29511}"
OUT_DIR="out/stage1_pretraining"
DATASET="CSL_News"

# Effective batch = micro_batch_per_gpu * grad_accum * world_size
MICRO_BSZ=48
GRAD_ACCUM=2
EPOCHS=20
LR=3e-4
WARMUP_EPOCHS=1               # small warmup helps stability
PRINT_FREQ=50                 # matches utils.get_args_parser
NUM_WORKERS=8                 # tune per machine
PREFETCH_FACTOR=4             # requires num_workers > 0
PERSISTENT_WORKERS=1          # set 0 to disable
MAX_LEN=256
ZERO_STAGE=2                  # ZeRO-2 as in config
DTYPE="bf16"                  # bf16 recommended on Ampere/L4/L40
QUICK_BREAK=2048              # periodic checkpoint trigger
# <<<

mkdir -p "${OUT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${OUT_DIR}/train_${TIMESTAMP}.log"

# Reasonable defaults for a single host, 8x GPUs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONWARNINGS=ignore                # silence Python warnings
export TORCH_SHOW_CPP_STACKTRACES=0         # don't try to symbolize C++ stack traces
export TORCH_DISABLE_ADDR2LINE=1            # hard-disable addr2line calls that print those lines
export TORCH_CPP_LOG_LEVEL=ERROR            # suppress PyTorch C++ INFO/WARN logs
export DEEPSPEED_LOG_LEVEL=ERROR            # suppress DeepSpeed INFO config dump
export NCCL_DEBUG=ERROR 
# If no Infiniband, avoid wasted probing:
export NCCL_IB_DISABLE=1

# Optional but helpful on Ampere+:
python - <<'PY' || true
import torch
try:
    torch.set_float32_matmul_precision("high")  # enables TF32 matmuls if supported
except Exception:
    pass
PY

cmd=(
deepspeed
  --include "${GPUS}"
  --master_port "${MASTER_PORT}"
  pre_training.py
    --dataset "${DATASET}"
    --output_dir "${OUT_DIR}"
    --epochs "${EPOCHS}"
    --batch-size "${MICRO_BSZ}"
    --gradient-accumulation-steps "${GRAD_ACCUM}"
    --opt AdamW
    --lr "${LR}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --quick_break "${QUICK_BREAK}"
    --dtype "${DTYPE}"
    --zero_stage "${ZERO_STAGE}"
    --max_length "${MAX_LEN}"
    --print-freq "${PRINT_FREQ}"
    --num_workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    $( [[ "${PERSISTENT_WORKERS}" == "1" ]] && echo --persistent-workers )
    # --rgb_support   # uncomment if using RGB branch
)

echo "[INFO] Launching:"
printf ' %q' "${cmd[@]}"; echo
echo "[INFO] Logs -> ${LOGFILE}"
echo "[INFO] Global batch = ${MICRO_BSZ} x ${GRAD_ACCUM} x $(echo "${GPUS}" | tr -cd , | wc -c | awk '{print $1+1}')"

nohup "${cmd[@]}" > "${LOGFILE}" 2>&1 &

echo "[INFO] PID $!   (tail -f ${LOGFILE})"
