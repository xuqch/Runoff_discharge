# Code_2026-06-03

06-03 is a simplified global runoff training version. It keeps the global basin nc workflow, qobs from nc `discharge`, checkpoint resume, validation, and inference capabilities from the 05-29 version, while moving the training path back toward the simpler PhaseH style.

## Main Files

- `config.py`: command line options and path normalization.
- `utils.py`: basin table reading, nc variable loading, date helpers, scalers.
- `build_h5.py`: compact per-basin H5 generation.
- `dataset_global.py`: `BlockPerBasinH5Dataset` / `PerBasinBlockH5Dataset`.
- `balanced_block_sampler.py`: block-level DDP load balancing.
- `model.py`: EA-LSTM runoff generator and Muskingum routing.
- `model_phaseh.py`: block sequence forward wrapper.
- `loss.py`: q loss, MFM/peak option, water balance terms.
- `train_global.py`: function-style training loop with checkpoint resume.
- `validation.py`: best/last checkpoint evaluation for train/validation/all.
- `inference_unit.py`, `inference_global.py`, `inference_global_parallel.py`: global yearly inference with previous-year context.

## What Comes From PhaseH

- Function-style training flow in `train_global.py`.
- `BlockPerBasinH5Dataset` sample unit: one basin target block.
- Block sequence forward: read `[P, prefix_len, D]`, generate runoff sequence, select target days, then route.
- Model/loss style using `precompute_inputs` and `precompute_time_chunk` only.

## What Comes From 05-29

- Basin ids are strings and are read from `basins_file.basin_id`.
- qobs is read from basin nc variable `--qobs_var`, default `discharge`.
- Basin `sdate` / `edate` is intersected with global train/eval date windows.
- NaN qobs remain on the original target timeline and are masked by `target_valid`.
- Resume, validation, global inference, parallel inference, and previous-year context are retained.

## Removed Complexity

- No separate per-basin streamflow csv inputs.
- No static array cache in dataset.
- No oversized `Trainer` class.
- No large nonfinite diagnosis/reporting path.
- No configurable DDP unused-parameter search; DDP uses `find_unused_parameters=False`.
- No periodic auxiliary-loss switches; q, balance, peak/MFM, and routing terms are computed every step.

## Basic Commands

## Training And Validation

06-03 does not run validation automatically inside `train_global.py`. Training only fits the model and saves checkpoints. Run validation separately after training with `validation.py` or `run_eval.sh`.

Build/train:

```bash
python train_global.py \
  --basins_file basins_for_train.csv \
  --data_dir /path/to/Global \
  --h5_dir ealstm_h5 \
  --qobs_var discharge \
  --epochs 30 \
  --use_amp
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash launch_ddp_train_phaseh.sh \
  --basins_file basins_for_train.csv \
  --data_dir /path/to/Global \
  --use_amp
```

Resume:

```bash
python train_global.py --run_dir /path/to/run --resume_latest
python train_global.py --resume_ckpt /path/to/run/checkpoints/ckpt_epoch010.pth
```

Validation:

```bash
python validation.py \
  --run_dir /path/to/run \
  --eval_model best \
  --eval_split validation \
  --basins_file basins_for_validation.csv
```

Global inference:

```bash
python inference_global.py \
  --run_dir /path/to/run \
  --eval_model best \
  --data_dir /path/to/Global
```

Parallel inference:

```bash
python inference_global_parallel.py \
  --run_dir /path/to/run \
  --eval_model best \
  --data_dir /path/to/Global \
  --cuda_devices 0,1
```

## Block-Level Balancing

Large basins and long target periods vary heavily in compute cost. The sampler estimates each block load as `p_count * prefix_len`, sorts blocks by load, and greedily assigns them to ranks. This keeps each GPU closer in total work than assigning entire basins to devices.
