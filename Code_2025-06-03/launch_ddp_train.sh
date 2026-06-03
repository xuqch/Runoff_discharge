#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
RUN_DIR_VALUE="${RUN_DIR:-}"
NNODES_VALUE="${NNODES:-1}"
NODE_RANK_VALUE="${NODE_RANK:-0}"
MASTER_ADDR_VALUE="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_VALUE="${MASTER_PORT:-29500}"
RDZV_ID_VALUE="${RDZV_ID:-phaseh_ddp}"

if [ -n "${NPROC_PER_NODE:-}" ]; then
  NPROC_PER_NODE_VALUE="$NPROC_PER_NODE"
else
  if [ -z "$CUDA_VISIBLE_DEVICES_VALUE" ]; then
    CUDA_VISIBLE_DEVICES_VALUE="0,1"
  fi
  IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "$CUDA_VISIBLE_DEVICES_VALUE"
  NPROC_PER_NODE_VALUE="${#CUDA_DEVICE_ARRAY[@]}"
  if [ "$NPROC_PER_NODE_VALUE" -le 0 ]; then
    NPROC_PER_NODE_VALUE=1
  fi
fi

if [ -z "$CUDA_VISIBLE_DEVICES_VALUE" ]; then
  CUDA_VISIBLE_DEVICES_VALUE="$(seq -s, 0 $((NPROC_PER_NODE_VALUE - 1)))"
fi

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TORCHRUN_ARGS=(
  --nproc_per_node="$NPROC_PER_NODE_VALUE"
)

if [ "$NNODES_VALUE" -gt 1 ]; then
  TORCHRUN_ARGS+=(
    --nnodes="$NNODES_VALUE"
    --node_rank="$NODE_RANK_VALUE"
    --rdzv_backend=c10d
    --rdzv_id="$RDZV_ID_VALUE"
    --rdzv_endpoint="${MASTER_ADDR_VALUE}:${MASTER_PORT_VALUE}"
  )
else
  TORCHRUN_ARGS+=(--standalone)
fi

TRAIN_ARGS=()
if [ -n "$RUN_DIR_VALUE" ]; then
  TRAIN_ARGS+=(--run_dir "$RUN_DIR_VALUE")
fi
TRAIN_ARGS+=("$@")

exec torchrun \
  "${TORCHRUN_ARGS[@]}" \
  "$SCRIPT_DIR/train_global.py" \
  "${TRAIN_ARGS[@]}"
