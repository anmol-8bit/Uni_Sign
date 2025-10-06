#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# Stage-1 Uni-Sign Pretraining
# ------------------------------

# >>> Edit these if needed
GPUS="localhost:0,1,2,3,4,5,6,7"
MASTER_PORT="${MASTER_PORT:-29511}"
OUT_DIR="out/stage1_pretraining"
DATASET="MERGED_ASL_CSL"

# Effective batch = micro_batch_per_gpu * grad_accum * world_size
MICRO_BSZ=48
GRAD_ACCUM=1
EPOCHS=30
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

# --- W&B config (optional) ---
USE_WANDB=1                   # set 0 to disable
WANDB_PROJECT="unisign"
WANDB_ENTITY=""               # set if you use a team/org
WANDB_RUN_NAME="stage1-pretrain-augmentations_arch2-asl_csl_merged"
WANDB_GROUP="stage1"
WANDB_TAGS="pretrain deepspeed bf16"
WANDB_MODE="online"           # online|offline|disabled
WANDB_ID=""                   # set to resume a specific run id
# <<<

mkdir -p "${OUT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${OUT_DIR}/train_${TIMESTAMP}_augmentations_arch_2-asl_csl_merged.log"

# Reasonable defaults for a single host, 8x GPUs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONWARNINGS=ignore
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_CPP_LOG_LEVEL=ERROR
export DEEPSPEED_LOG_LEVEL=ERROR
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=1  # if you don't have IB
export TF_CPP_MIN_LOG_LEVEL=3       # hide INFO+WARN from TensorFlow
export TF_ENABLE_ONEDNN_OPTS=1 

# Optional but helpful on Ampere+:
python - <<'PY' || true
import torch
try:
    torch.set_float32_matmul_precision("high")  # enables TF32 matmuls if supported
except Exception:
    pass
PY

# W&B env (launcher level; non-main ranks will be disabled in code)
if [[ "${USE_WANDB}" == "1" ]]; then
  export WANDB_MODE="${WANDB_MODE}"
  [[ -n "${WANDB_ENTITY}" ]] && export WANDB_ENTITY="${WANDB_ENTITY}"
  export WANDB_PROJECT="${WANDB_PROJECT}"
fi

WORLD_SIZE=$(( $(tr -cd , <<<"${GPUS}" | wc -c) + 1 ))
echo "[INFO] Global batch = ${MICRO_BSZ} x ${GRAD_ACCUM} x ${WORLD_SIZE}"

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

# W&B flags passed to script
if [[ "${USE_WANDB}" == "1" ]]; then
  cmd+=( --wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${WANDB_RUN_NAME}" --wandb-group "${WANDB_GROUP}" )
  [[ -n "${WANDB_ENTITY}" ]] && cmd+=( --wandb-entity "${WANDB_ENTITY}" )
  [[ -n "${WANDB_TAGS}" ]] && cmd+=( --wandb-tags ${WANDB_TAGS} )
  [[ -n "${WANDB_MODE}" ]] && cmd+=( --wandb-mode "${WANDB_MODE}" )
  [[ -n "${WANDB_ID}" ]] && cmd+=( --wandb-id "${WANDB_ID}" )
fi

echo "[INFO] Launching:"
printf ' %q' "${cmd[@]}"; echo
echo "[INFO] Logs -> ${LOGFILE}"

nohup "${cmd[@]}" > "${LOGFILE}" 2>&1 &
echo $! > "${OUT_DIR}/run.pid"
echo "[INFO] PID $!   (tail -f ${LOGFILE})"

# helper to stop later:
# PGID=$(ps -o pgid= -p "$(cat ${OUT_DIR}/run.pid)" | tr -d ' ')
# kill -TERM -"$PGID"; sleep 3; kill -KILL -"$PGID"
