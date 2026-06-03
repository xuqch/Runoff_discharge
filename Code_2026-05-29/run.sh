#!/usr/bin/env bash
set -euo pipefail
#CUDA_VISIBLE_DEVICES=0 bash run.sh
#CUDA_VISIBLE_DEVICES=0,1 bash run.sh
#CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash run.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/launch_ddp_train_phaseh.sh" \
  --use_amp \
  --num_workers 4 \
  --pin_memory \
  --persistent_workers \
  --basin_batch_size 1 \
  --target_block_size 768 \
  --target_block_stride 768 \
  --generator_chunk_size 2048 \
  --balanced_bucket_size 64 \
  --empty_cache_interval 200 \
  --epochs 30 \
  --Loss NSEstd \
  --max_lag -1 \
  --q_file 'basins_for_train.csv' \
  --data_dir '/share/home/dq083/Runoff/LSTM/Experiment_for_runoff/Global/' \
  --scalers_path 'scalers_Global.json' \
  "$@"
#  --use_checkpoint \
#  --use_checkpoint \
#  --run_dir '/share/home/dq083/Runoff/LSTM/Experiment_for_runoff/Code_2026-05-29/runs/run_phaseh_0528_2727_seed681/' \
#  --resume_latest \



#  BALANCED_BUCKET_SIZE_VALUE="${BALANCED_BUCKET_SIZE:-32}"
#BALANCED_SAMPLER_SEED_VALUE="${BALANCED_SAMPLER_SEED:-0}"
#TARGET_BLOCK_SIZE_VALUE="${TARGET_BLOCK_SIZE:-1024}"
#TARGET_BLOCK_STRIDE_VALUE="${TARGET_BLOCK_STRIDE:-1024}"768 1536
#GENERATOR_CHUNK_SIZE_VALUE="${GENERATOR_CHUNK_SIZE:-8192}"
#  --run_dir '/share/home/dq083/Runoff/LSTM/Experiment_for_runoff/Code_2026-05-22/runs/run_phaseh_0528_2727_seed681/' \
#  --resume_latest \

