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
        if any(opt in sys.argv for opt in action.option_strings):
            explicit.add(action.dest)
    return explicit


def _load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def get_args() -> Dict:
    parser = argparse.ArgumentParser(description="06-03 simplified global runoff training with per-basin H5 blocks.")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--data_dir", type=str, default="/data/xuqch3/LSTM_Runoff/Experiment_for_runoff/Global")
    parser.add_argument("--basins_file", type=str, default="basins_for_train.csv")
    parser.add_argument("--nc_dir", type=str, default=None)
    parser.add_argument("--h5_dir", type=str, default="ealstm_h5")
    parser.add_argument("--eval_h5_dir", type=str, default="ealstm_validation")
    parser.add_argument("--scalers_path", type=str, default="scalers_Global.json")
    parser.add_argument("--qobs_var", type=str, default="discharge")

    parser.add_argument("--dyn_vars", nargs="+", default=["precip", "temp", "sp", "strd", "Q", "ssrd"])
    parser.add_argument("--stat_vars", nargs="+", default=[f"Band{i}" for i in range(1, 65)])
    parser.add_argument("--mask_var", type=str, default="elv")
    parser.add_argument("--dist_var", type=str, default="dist_map")
    parser.add_argument("--time_name", type=str, default="time")
    parser.add_argument("--precip_var", type=str, default="precip")
    parser.add_argument("--pet_var", type=str, default="pet")
    parser.add_argument("--pet_method", type=str, default="hamon")
    parser.add_argument("--fraction_var", type=str, default="fraction")

    parser.add_argument("--train_start_date", type=str, default="1994-10-01")
    parser.add_argument("--train_end_date", type=str, default="2022-09-30")
    parser.add_argument("--eval_start_date", type=str, default="1980-10-01")
    parser.add_argument("--eval_end_date", type=str, default="1993-08-30")
    parser.add_argument("--infer_start_year", type=int, default=1980)
    parser.add_argument("--infer_end_year", type=int, default=2023)

    parser.add_argument("--seq_len", type=int, default=270)
    parser.add_argument("--target_block_size", type=int, default=512)
    parser.add_argument("--target_block_stride", type=int, default=512)
    parser.add_argument("--target_block_shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target_block_drop_last", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--basin_batch_size", type=int, default=1)
    parser.add_argument("--eval_basin_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--lr_gen', type=float, default=1e-3)
    parser.add_argument('--lr_vel', type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_lag", type=int, default=60)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")

    parser.add_argument("--precompute_inputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precompute_time_chunk", type=int, default=0)
    parser.add_argument(
        "--precompute_max_positions",
        type=int,
        default=1500000,
        help=(
            "Maximum P * prefix_len allowed for EA-LSTM input precomputation. "
            "If precompute_inputs=True but P*prefix_len exceeds this threshold, "
            "precompute is disabled for that basin-block. Set <=0 to disable this guard."
        ),
    )

    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--resume_latest", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--Loss", type=str, default="NSEstd", choices=["NSEstd", "MFM"])
    parser.add_argument("--w_q", type=float, default=1.0)
    parser.add_argument("--w_mfm", type=float, default=1.0)
    parser.add_argument("--w_peak", type=float, default=0.0)
    parser.add_argument("--w_balance", type=float, default=0.1)
    parser.add_argument("--w_route", type=float, default=1.0)
    parser.add_argument("--w_budyko", type=float, default=None)
    parser.add_argument("--balance_loss", type=str, default="budyko_annual")
    parser.add_argument("--budyko_alpha", type=float, default=2.6)
    parser.add_argument("--budyko_year_start_month", type=int, default=10)
    parser.add_argument("--budyko_min_days_per_year", type=int, default=300)
    parser.add_argument("--mfm_p", type=float, default=1.0)
    parser.add_argument("--mfm_bins_suse", type=int, default=10)
    parser.add_argument("--mfm_bins_phi", type=int, default=10)
    parser.add_argument("--mfm_phase_penalty_scaling", type=float, default=4.0)
    parser.add_argument("--mfm_phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mfm_eps", type=float, default=1e-6)
    parser.add_argument("--mfm_soft_hist_sigma_scale", type=float, default=0.5)
    parser.add_argument("--peak_quantile", type=float, default=0.9)
    parser.add_argument("--peak_weight", type=float, default=2.0)
    parser.add_argument("--peak_eps", type=float, default=0.1)
    parser.add_argument("--peak_use_relative_error", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--balanced_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced_bucket_size", type=int, default=16)
    parser.add_argument("--balanced_sampler_drop_last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--balanced_sampler_seed", type=int, default=0)

    parser.add_argument("--generator_chunk_size", type=int, default=8192)
    parser.add_argument("--grid_batch_size", type=int, default=1)
    parser.add_argument("--inference_num_workers", type=int, default=0)
    parser.add_argument("--inference_tmp_dir", type=str, default=None)
    parser.add_argument("--cuda_devices", type=str, default=None)
    parser.add_argument("--keep_inference_parts", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--rebuild_validation_data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h5_build_workers", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--eval_ckpt", type=str, default=None)
    parser.add_argument("--eval_model", type=str, choices=["best", "last"], default="best")
    parser.add_argument("--eval_split", type=str, choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--clear_cache", action="store_true")
    parser.add_argument("--empty_cache_interval", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--write_last_block_diagnostics", action="store_true", default=False)
    parser.add_argument("--last_block_diagnostics_file", type=str, default="")
    parser.add_argument("--debug_loss_threshold", type=float, default=0.0)
    parser.add_argument("--debug_loss_max_records", type=int, default=2000)
    parser.add_argument("--debug_loss_print_limit", type=int, default=50)
    parser.add_argument("--debug_loss_log_file", type=str, default="")
    parser.add_argument("--info", type=str, default=None)

    args = parser.parse_args()
    cfg = vars(args)
    cfg["_explicit_keys"] = _collect_explicit_keys(parser)
    return cfg


def _merge_saved_config(cfg: Dict, saved_cfg: Dict) -> None:
    explicit = cfg.get("_explicit_keys", set())
    transient = {"resume_ckpt", "resume_latest", "eval_ckpt", "eval_model", "eval_split"}
    for key, value in saved_cfg.items():
        if key.startswith("_") or key in transient or key in explicit:
            continue
        cfg[key] = value


def _resolve_file(path_like: str | Path, *roots: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return roots[0] / path if roots else path


def _bootstrap_run_defaults(cfg: Dict) -> None:
    code_dir = Path(__file__).resolve().parent
    repo_runs_dir = code_dir / "runs"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        rank = int(os.environ.get("RANK", "0"))
        run_id = (
            os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("RDZV_ID")
            or os.environ.get("MASTER_PORT")
            or f"pid_{os.getppid()}"
        )
        bootstrap_file = repo_runs_dir / "temp" / "ddp_bootstrap" / f"{run_id}.json"

        if rank == 0:
            payload: Dict[str, object] = {}
            payload["seed"] = int(cfg["seed"]) if cfg.get("seed") is not None else int(np.random.uniform(low=0, high=1000))

            raw_run_dir = cfg.get("run_dir") or cfg.get("output_dir")
            if raw_run_dir is None:
                now = datetime.now()
                run_name = f"run_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}_seed{int(payload['seed'])}"
                payload["run_dir"] = str(repo_runs_dir / run_name)
            else:
                path = Path(str(raw_run_dir)).expanduser()
                payload["run_dir"] = str(path if path.is_absolute() else repo_runs_dir / path)

            _save_json(bootstrap_file, payload)
            cfg["seed"] = int(payload["seed"])
            cfg["run_dir"] = Path(str(payload["run_dir"]))
            cfg["output_dir"] = cfg["run_dir"]
            return

        start_time = time.time()
        while not bootstrap_file.exists():
            if time.time() - start_time > 300:
                raise TimeoutError(f"Timed out waiting for distributed bootstrap file: {bootstrap_file}")
            time.sleep(0.2)

        payload = _load_json(bootstrap_file)
        cfg["seed"] = int(payload["seed"])
        cfg["run_dir"] = Path(str(payload["run_dir"]))
        cfg["output_dir"] = cfg["run_dir"]
        return

    if cfg.get("seed") is None:
        cfg["seed"] = int(np.random.uniform(low=0, high=1000))

    if cfg.get("run_dir") is None and cfg.get("output_dir") is None:
        now = datetime.now()
        run_name = f"run_phaseh_{now.month:02d}{now.day:02d}_{now.hour + 8:02d}{now.minute:02d}_seed{cfg['seed']}"
        cfg["run_dir"] = repo_runs_dir / run_name
    elif cfg.get("run_dir") is None:
        cfg["run_dir"] = cfg["output_dir"]
    cfg["output_dir"] = cfg["run_dir"]


def update_config(cfg: Dict) -> Dict:
    _bootstrap_run_defaults(cfg)
    code_dir = Path(__file__).resolve().parent
    repo_runs_dir = code_dir / "runs"
    data_dir = Path(cfg["data_dir"])

    run_dir = Path(cfg["run_dir"])
    cfg["run_dir"] = run_dir if run_dir.is_absolute() else repo_runs_dir / run_dir
    cfg["output_dir"] = cfg["run_dir"]
    cfg["out_dir"] = cfg["run_dir"]
    cfg["ckpt_dir"] = cfg["run_dir"] / "checkpoints"

    cfg_path = cfg["run_dir"] / "config_used.json"
    if cfg_path.exists():
        _merge_saved_config(cfg, _load_json(cfg_path))
        run_dir = Path(cfg["run_dir"])
        cfg["run_dir"] = run_dir if run_dir.is_absolute() else repo_runs_dir / run_dir
        cfg["output_dir"] = cfg["run_dir"]
        cfg["out_dir"] = cfg["run_dir"]
        cfg["ckpt_dir"] = cfg["run_dir"] / "checkpoints"

    cfg["basins_file"] = _resolve_file(cfg["basins_file"], code_dir / "data", data_dir, code_dir.parent)
    cfg["q_file"] = cfg["basins_file"]
    cfg["nc_dir"] = Path(cfg["nc_dir"]) if cfg.get("nc_dir") else data_dir / "Basins_data"

    explicit = cfg.get("_explicit_keys", set())
    h5_value = cfg.get("h5_dir")
    if "h5_dir" not in explicit and cfg.get("h5_path") is not None:
        h5_value = cfg["h5_path"]
    h5_dir = Path(h5_value)
    cfg["h5_dir"] = h5_dir if h5_dir.is_absolute() else repo_runs_dir / h5_dir
    cfg["h5_path"] = cfg["h5_dir"]

    if cfg.get("eval_h5_dir") is None:
        cfg["eval_h5_dir"] = None
    else:
        eval_h5_dir = Path(cfg["eval_h5_dir"])
        cfg["eval_h5_dir"] = eval_h5_dir if eval_h5_dir.is_absolute() else repo_runs_dir / eval_h5_dir
    cfg["eval_h5_path"] = cfg["eval_h5_dir"]

    scalers_path = Path(cfg["scalers_path"])
    cfg["scalers_path"] = scalers_path if scalers_path.is_absolute() else repo_runs_dir / scalers_path
    if cfg.get("resume_ckpt") is not None:
        cfg["resume_ckpt"] = Path(cfg["resume_ckpt"])

    if cfg.get("w_budyko") is not None:
        cfg["w_balance"] = float(cfg["w_budyko"])
    if cfg.get("lr_gen") is None:
        cfg["lr_gen"] = float(cfg["lr"])
    if cfg.get("lr_vel") is None:
        cfg["lr_vel"] = float(cfg["lr"])
    cfg["basin_batch_size"] = int(cfg.get("basin_batch_size") or cfg.get("batch_size") or 1)
    return cfg


def update_run_config(cfg: Dict) -> Dict:
    return update_config(cfg)
