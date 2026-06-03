from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from build_h5 import create_h5_files, find_missing_basin_h5s
from common import int64ns_to_datetime, resolve_eval_checkpoint, resolve_eval_output_dir
from config import get_args, update_config
from dataset_global import BlockPerBasinH5Dataset, collate_block_basin_batch
from model_phaseh import HydroAIBasinPhaseH
from utils import read_basin_ids


def _ensure_split_h5(cfg: Dict, split: str) -> Path:
    if split == "train":
        return Path(cfg["h5_dir"])

    h5_dir = Path(cfg.get("eval_h5_dir") or Path(cfg["run_dir"]) / "ealstm_validation")
    if h5_dir.exists():
        missing = find_missing_basin_h5s(cfg["basins_file"], h5_dir)
        if missing:
            report_path = h5_dir / "missing_basin_h5s.csv"
            pd.DataFrame(missing, columns=["basin_id", "expected_h5_path"]).to_csv(report_path, index=False)
            print(f"[H5] validation h5_dir exists: {h5_dir}")
            print(f"[H5] Missing basin H5 count: {len(missing)}")
            print(f"[H5] Full missing list written to: {report_path}")
            print("[H5] No basin H5 files were built because h5_dir already exists.")
        return h5_dir

    print(f"[H5] validation h5_dir does not exist, building all basin H5 files: {h5_dir}")
    h5_dir.mkdir(parents=True, exist_ok=True)

    create_h5_files(
        nc_dir=cfg["nc_dir"],
        basins_file=cfg["basins_file"],
        scalers_path=cfg["scalers_path"],
        h5_dir=h5_dir,
        dyn_vars=cfg["dyn_vars"],
        stat_vars=cfg["stat_vars"],
        seq_length=int(cfg["seq_len"]),
        mask_var=cfg["mask_var"],
        dist_var=cfg["dist_var"],
        time_name=cfg["time_name"],
        qobs_var=cfg["qobs_var"],
        precip_var=cfg["precip_var"],
        pet_var=cfg.get("pet_var"),
        pet_method=cfg.get("pet_method", "hamon"),
        target_start_date=cfg["eval_start_date"],
        target_end_date=cfg["eval_end_date"],
        fraction_var=cfg["fraction_var"],
        water_year_start_month=int(cfg.get("budyko_year_start_month", 10)),
        h5_build_workers=int(cfg.get("h5_build_workers", 1)),
    )
    return h5_dir


def _load_model(cfg: Dict, ckpt_path: Path, device: torch.device) -> HydroAIBasinPhaseH:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config") or ckpt.get("cfg") or {}
    model = HydroAIBasinPhaseH(
        dims={"dyn": len(cfg["dyn_vars"]), "stat": len(cfg["stat_vars"])},
        hidden_dim=int(saved_cfg.get("hidden_dim", cfg.get("hidden_dim", 128))),
        dropout=float(saved_cfg.get("dropout", cfg.get("dropout", 0.4))),
        precompute_inputs=bool(saved_cfg.get("precompute_inputs", cfg.get("precompute_inputs", True))),
        precompute_time_chunk=int(saved_cfg.get("precompute_time_chunk", cfg.get("precompute_time_chunk", 0))),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if obs.size < 2:
        return float("nan")
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((sim - obs) ** 2) / denom)


def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if obs.size < 2:
        return float("nan")
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(np.std(sim) / (np.std(obs) + 1e-12))
    beta = float(np.mean(sim) / (np.mean(obs) + 1e-12))
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def evaluate_split(
    cfg: Dict,
    split: str,
    model: HydroAIBasinPhaseH,
    device: torch.device,
    h5_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    basins = read_basin_ids(cfg["basins_file"])
    dataset = BlockPerBasinH5Dataset(
        h5_path=h5_dir,
        scalers_path=cfg["scalers_path"],
        seq_len=int(cfg["seq_len"]),
        basins=basins,
        target_block_size=int(cfg["target_block_size"]),
        target_block_stride=int(cfg["target_block_stride"]),
        drop_last_block=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_basin_batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_block_basin_batch,
    )

    by_basin: dict[str, dict[str, list[Any]]] = {}
    errors: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {split}", dynamic_ncols=True):
            basin_meta = batch["basin_meta"]
            try:
                outputs = model(
                    None,
                    None,
                    basin_meta=basin_meta,
                    max_lag=int(cfg["max_lag"]),
                    generator_chunk_size=int(cfg.get("generator_chunk_size", 8192)),
                    is_basin=True,
                )
            except Exception as exc:
                for meta in basin_meta:
                    errors.append({"basin_id": str(meta.get("basin_id", "")), "reason": f"{type(exc).__name__}: {exc}"})
                continue

            for q_pred, meta in zip(outputs["q_pred"], basin_meta):
                basin_id = str(meta["basin_id"])
                q_true = meta["q_true"].detach().cpu().numpy().reshape(-1)
                q_valid = meta["q_valid"].detach().cpu().numpy().reshape(-1).astype(bool)
                pred = q_pred.detach().cpu().numpy().reshape(-1)
                dates = int64ns_to_datetime(np.asarray(meta["target_dates"], dtype=np.int64))
                store = by_basin.setdefault(basin_id, {"date": [], "q_true": [], "q_pred": []})
                store["date"].extend([str(pd.Timestamp(d).date()) for d in dates[q_valid]])
                store["q_true"].extend(q_true[q_valid].astype(float).tolist())
                store["q_pred"].extend(pred[q_valid].astype(float).tolist())

    rows: list[dict[str, Any]] = []
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for basin_id, values in sorted(by_basin.items()):
        obs = np.asarray(values["q_true"], dtype=np.float64)
        sim = np.asarray(values["q_pred"], dtype=np.float64)
        rows.append(
            {
                "basin_id": basin_id,
                "n": int(obs.size),
                "nse": _nse(obs, sim),
                "kge": _kge(obs, sim),
                "bias": float(np.nanmean(sim - obs)) if obs.size else float("nan"),
                "rmse": float(np.sqrt(np.nanmean((sim - obs) ** 2))) if obs.size else float("nan"),
            }
        )
        pd.DataFrame(values).to_csv(split_dir / f"{basin_id}.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / f"{split}_metrics.csv", index=False)
    pd.DataFrame({"basin_id": sorted(by_basin.keys())}).to_csv(out_dir / f"basins_{split}_available.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "evaluation_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["basin_id", "reason"]).to_csv(out_dir / "evaluation_errors.csv", index=False)
    return metrics


def evaluate(cfg: Dict) -> None:
    cfg = update_config(cfg)
    run_dir = Path(cfg["run_dir"])
    ckpt = resolve_eval_checkpoint(run_dir, str(cfg.get("eval_model", "best")), cfg.get("eval_ckpt"))
    out_dir = resolve_eval_output_dir(run_dir, str(cfg.get("eval_model", "best")))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda") else "cpu")
    model = _load_model(cfg, ckpt, device)
    splits = ["train", "validation"] if cfg.get("eval_split") == "all" else [str(cfg.get("eval_split", "validation"))]

    all_metrics: list[pd.DataFrame] = []
    for split in splits:
        h5_dir = _ensure_split_h5(cfg, split)
        metrics = evaluate_split(cfg, split, model, device, h5_dir, out_dir)
        metrics.insert(0, "split", split)
        all_metrics.append(metrics)
    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        summary.to_csv(out_dir / "metrics_summary.csv", index=False)
        print(summary.groupby("split")[["nse", "kge", "rmse"]].mean(numeric_only=True))


if __name__ == "__main__":
    evaluate(get_args())
