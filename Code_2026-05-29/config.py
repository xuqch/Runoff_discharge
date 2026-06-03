from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np


def _collect_explicit_keys(parser: argparse.ArgumentParser) -> set[str]:
    explicit: set[str] = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            if opt in sys.argv:
                explicit.add(action.dest)
                break
    return explicit


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _bootstrap_run_defaults(cfg: Dict) -> None:
    if cfg.get("seed") is not None and cfg.get("run_dir") is not None:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    rank = int(os.environ.get("RANK", "0"))
    run_id = (
        os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("MASTER_PORT")
        or f"pid_{os.getppid()}"
    )
    repo_root = Path(__file__).resolve().parent
    bootstrap_dir = repo_root / "runs" / "temp" / "ddp_bootstrap"
    bootstrap_file = bootstrap_dir / f"{run_id}.json"

    if rank == 0:
        payload: Dict[str, object] = {}
        if cfg.get("seed") is None:
            payload["seed"] = int(np.random.uniform(low=0, high=1000))
        else:
            payload["seed"] = int(cfg["seed"])

        if cfg.get("run_dir") is None:
            now = datetime.now()
            run_name = (
                f"run_phaseh_{now.month:02d}{now.day:02d}_{now.hour + 8:02d}{now.minute:02d}_"
                f"seed{int(payload['seed'])}"
            )
            payload["run_dir"] = str(repo_root / "runs" / run_name)
        else:
            payload["run_dir"] = str(Path(cfg["run_dir"]))

        _save_json(bootstrap_file, payload)
        cfg["seed"] = int(payload["seed"])
        cfg["run_dir"] = Path(str(payload["run_dir"]))
        return

    start_time = time.time()
    while not bootstrap_file.exists():
        if time.time() - start_time > 300:
            raise TimeoutError(f"Timed out waiting for distributed run bootstrap file: {bootstrap_file}")
        time.sleep(0.2)

    payload = _load_json(str(bootstrap_file))
    if cfg.get("seed") is None:
        cfg["seed"] = int(payload["seed"])
    if cfg.get("run_dir") is None:
        cfg["run_dir"] = Path(str(payload["run_dir"]))


def get_args() -> Dict:
    parser = argparse.ArgumentParser(description="Train PhaseH with per-basin H5 storage and block-time basin training.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--data_dir", type=str, default="/data/xuqch3/LSTM_Runoff/Experiment_for_runoff/Global/")
    parser.add_argument("--qobs_var", type=str, default="discharge")
    parser.add_argument("--q_file", type=str, default="basins_for_train.csv")
    parser.add_argument("--h5_path", type=str, default="ealstm_h5")
    parser.add_argument("--eval_h5_path", type=str, default=None)
    parser.add_argument("--eval_ckpt", type=str, default=None)
    parser.add_argument(
        "--eval_model",
        type=str,
        choices=["best", "last"],
        default=None,
        help="Validation checkpoint selector. Use 'best' for best_model.pth or 'last' for latest/last epoch checkpoint.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "validation", "all"],
        default="validation",
        help="Which split to evaluate: training period, validation period, or both.",
    )
    parser.add_argument("--scalers_path", type=str, default="scalers.json")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--resume_latest", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--dyn_vars", nargs="+", default=["precip", "temp", "sp", "strd", "Q", "ssrd"])
    parser.add_argument("--stat_vars", nargs="+", default=[f"Band{i}" for i in range(1, 65)])
    parser.add_argument("--mask_var", type=str, default="elv")
    parser.add_argument("--dist_var", type=str, default="dist_map")
    parser.add_argument("--time_name", type=str, default="time")
    parser.add_argument("--precip_var", type=str, default="precip")
    parser.add_argument("--pet_var", type=str, default="pet")
    parser.add_argument("--pet_method", type=str, default="hamon")
    parser.add_argument("--area_var", type=str, default="area_m2")
    parser.add_argument("--fraction_var", type=str, default="fraction")

    parser.add_argument("--seq_len", type=int, default=270)
    parser.add_argument("--train_start_date", type=str, default="1994-10-01")
    parser.add_argument("--train_end_date", type=str, default="2022-09-30")
    parser.add_argument("--eval_start_date", type=str, default="1980-10-01")
    parser.add_argument("--eval_end_date", type=str, default="1993-08-30")
    parser.add_argument("--infer_start_year", type=int, default=1980)
    parser.add_argument("--infer_end_year", type=int, default=2023)
    parser.add_argument("--drop_invalid_targets", action="store_true")
    parser.add_argument("--Loss", type=str, default="NSEstd")

    parser.add_argument("--basin_batch_size", type=int, default=1)
    parser.add_argument("--eval_basin_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_gen", type=float, default=1e-3)
    parser.add_argument("--lr_vel", type=float, default=1e-4)
    parser.add_argument("--max_lag", type=int, default=60)
    parser.add_argument("--w_balance", type=float, default=0.1)
    parser.add_argument("--balance_loss", type=str, default="budyko_annual")
    parser.add_argument("--budyko_alpha", type=float, default=2.6)
    parser.add_argument("--budyko_year_start_month", type=int, default=10)
    parser.add_argument("--budyko_min_days_per_year", type=int, default=300)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--rebuild_h5", action="store_true")
    parser.add_argument("--rebuild_eval_h5", action="store_true")
    parser.add_argument(
        "--rebuild_validation_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rebuilding cached validation basin lists and per-basin validation H5 data under run_dir/ealstm_validation.",
    )
    parser.add_argument("--h5_build_workers", type=int, default=1)

    parser.add_argument("--generator_chunk_size", type=int, default=8192)
    parser.add_argument("--grid_batch_size", type=int, default=1)
    parser.add_argument("--inference_num_workers", type=int, default=0)
    parser.add_argument("--inference_tmp_dir", type=str, default=None)
    parser.add_argument("--cuda_devices", type=str, default=None)
    parser.add_argument("--keep_inference_parts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target_block_size", type=int, nargs="?", const=512, default=512)
    parser.add_argument("--target_block_stride", type=int, nargs="?", const=512, default=512)
    parser.add_argument("--target_block_shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_block_drop_last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precompute_inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precompute_time_chunk", type=int, default=0)
    parser.add_argument("--precompute_gate_positions_limit", type=int, default=1200000)
    parser.add_argument("--compile_generator", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--empty_cache_interval", type=int, default=20)
    parser.add_argument("--empty_cache_each_epoch", action="store_true")
    parser.add_argument("--clear_cache", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--balanced_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced_bucket_size", type=int, default=16)
    parser.add_argument("--balanced_sampler_drop_last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced_sampler_seed", type=int, default=0)
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_nonfinite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile_step_timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile_step_timing_csv", type=str, default="timing_step_phaseh.csv")
    parser.add_argument("--profile_step_timing_print_every", type=int, default=10)

    parser.add_argument("--mfm_p", type=float, default=1.0)
    parser.add_argument("--mfm_bins_suse", type=int, default=10)
    parser.add_argument("--mfm_bins_phi", type=int, default=10)
    parser.add_argument("--mfm_phase_penalty_scaling", type=float, default=4.0)
    parser.add_argument("--mfm_phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mfm_eps", type=float, default=1e-6)
    parser.add_argument("--mfm_soft_hist_sigma_scale", type=float, default=0.5)
    parser.add_argument("--w_peak", type=float, default=0.0)
    parser.add_argument("--peak_quantile", type=float, default=0.9)
    parser.add_argument("--peak_weight", type=float, default=2.0)
    parser.add_argument("--peak_eps", type=float, default=0.1)
    parser.add_argument("--peak_use_relative_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--info", type=str, default=None)

    args = parser.parse_args()
    cfg = vars(args)
    cfg["_explicit_keys"] = _collect_explicit_keys(parser)
    return cfg


def _merge_saved_config(cfg: Dict, saved_cfg: Dict) -> None:
    explicit_keys = cfg.get("_explicit_keys", set())
    transient_keys = {"resume_ckpt", "resume_latest"}
    for k, v in saved_cfg.items():
        if k.startswith("_"):
            continue
        if k in transient_keys:
            continue
        if k in explicit_keys:
            continue
        cfg[k] = v


def update_config(cfg: Dict) -> Dict:
    _bootstrap_run_defaults(cfg)

    if cfg["seed"] is None:
        cfg["seed"] = int(np.random.uniform(low=0, high=1000))

    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
        cfg_path = cfg["run_dir"] / "config_used.json"
        if cfg_path.exists():
            saved = _load_json(str(cfg_path))
            _merge_saved_config(cfg, saved)
    if cfg.get("resume_ckpt") is not None:
        cfg["resume_ckpt"] = Path(cfg["resume_ckpt"])

    if not Path(cfg["q_file"]).is_absolute():
        repo_local_q_file = Path(__file__).resolve().parent / "data" / cfg["q_file"]
        data_dir_q_file = Path(cfg["data_dir"]) / cfg["q_file"]
        if repo_local_q_file.exists():
            cfg["q_file"] = repo_local_q_file
        else:
            cfg["q_file"] = data_dir_q_file

    cfg["nc_dir"] = Path(cfg["data_dir"]) / "Basins_data"
    return cfg


def update_run_config(cfg: Dict) -> Dict:
    return update_config(cfg)
