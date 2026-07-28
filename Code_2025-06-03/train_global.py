from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from balanced_block_sampler import BalancedBlockDistributedSampler
from common import (
    build_criterion,
    build_or_update_h5_if_needed,
    cleanup_distributed,
    configure_run_artifact_dirs,
    ddp_barrier,
    find_latest_checkpoint,
    is_main_process,
    load_checkpoint,
    maybe_empty_cuda_cache,
    resolve_device,
    save_checkpoint,
    seed_everything,
    setup_distributed,
)
from config import get_args, update_config
from dataset_global import BlockPerBasinH5Dataset, collate_block_basin_batch
from model_phaseh import HydroAIBasinPhaseH
from utils import read_basin_ids, save_json


def route_weights(cfg: Dict) -> Dict[str, float]:
    scale = float(cfg.get("w_route", 1.0))
    return {
        "w_route_mass": scale * float(cfg.get("w_route_mass", 1e-3)),
        "w_route_sigma_monotonic": scale * float(cfg.get("w_route_sigma_monotonic", 1e-2)),
        "w_route_hillslope_ratio": scale * float(cfg.get("w_route_hillslope_ratio", 1e-2)),
        "w_route_velocity_smooth": scale * float(cfg.get("w_route_velocity_smooth", 1e-3)),
        "w_route_sigma_width": scale * float(cfg.get("w_route_sigma_width", 1e-4)),
        "w_route_dispersion_coupling": scale * float(cfg.get("w_route_dispersion_coupling", 1e-4)),
        "w_route_effective_lag": scale * float(cfg.get("w_route_effective_lag", 0.0)),
    }


def compute_routing_regularization(model: HydroAIBasinPhaseH, outputs: Dict, basin_meta: List[Dict], device: torch.device, weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = torch.zeros((), device=device)
    route_aux_list = outputs.get("route_aux_for_loss", outputs.get("route_aux", []))
    runoff_m3s_list = outputs.get("runoff_m3s_for_loss", outputs.get("runoff_m3s", []))
    if not route_aux_list or not runoff_m3s_list:
        return zero, {"loss_route_total": 0.0}

    raw_model = model.module if hasattr(model, "module") else model
    total_route = zero
    accum = {k.replace("w_", ""): 0.0 for k in weights}
    count = 0
    for runoff_m3s, aux, meta in zip(runoff_m3s_list, route_aux_list, basin_meta):
        reg = raw_model.routing.regularization(
            dist_map_m=meta["dist_m"].to(device, non_blocking=True),
            aux=aux,
            runoff_m3s=runoff_m3s,
        )
        basin_route = zero
        for key, weight in weights.items():
            reg_key = key.replace("w_", "")
            basin_route = basin_route + float(weight) * reg[reg_key]
            accum[reg_key] += float(reg[reg_key].detach().cpu())
        total_route = total_route + basin_route
        count += 1

    total_route = total_route / max(count, 1)
    metrics = {"loss_route_total": float(total_route.detach().cpu())}
    for key, value in accum.items():
        metrics[f"loss_{key}"] = value / max(count, 1)
    return total_route, metrics


def _mean_aux_value(route_aux_list, keys: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for aux in route_aux_list or []:
        if not isinstance(aux, dict):
            continue
        for key in keys:
            if key not in aux:
                continue
            value = aux[key]
            if torch.is_tensor(value):
                value = value.detach().float().mean().item()
            values.append(float(value))
            break
    return sum(values) / len(values) if values else None


def _progress_postfix(loss_value: float, metrics: Dict[str, float], route_metrics: Dict[str, float], outputs: Dict) -> Dict[str, str]:
    route_aux = outputs.get("route_aux", [])
    lag_value = _mean_aux_value(
        route_aux,
        ("lag", "lag_mean", "lag_days", "total_lag_days", "channel_lag_days", "lag_len"),
    )
    v_mean_value = _mean_aux_value(
        route_aux,
        ("v_mean", "velocity_mean", "mean_velocity", "velocity_mps"),
    )
    postfix = {
        "lag": "-" if lag_value is None else f"{lag_value:.3g}",
        "loss": f"{loss_value:.3g}",
        "mb": f"{float(metrics.get('loss_mb', metrics.get('mb', 0.0))):.3g}",
        "q": f"{float(metrics.get('loss_q', metrics.get('q_loss', 0.0))):.3g}",
        "peak": f"{float(metrics.get('loss_peak', 0.0)):.3g}",
        "route": f"{float(route_metrics.get('loss_route_total', route_metrics.get('route_loss', 0.0))):.3g}",
        "v_mean": "-" if v_mean_value is None else f"{v_mean_value:.3g}",
    }
    return postfix


def setup_run_dir(cfg: Dict) -> Dict:
    cfg["run_dir"].mkdir(parents=True, exist_ok=True)
    cfg["out_dir"] = cfg["run_dir"]
    cfg["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    print(cfg["run_dir"])
    return cfg


def resolve_resume_path(cfg: Dict) -> Path | None:
    if cfg.get("resume_ckpt") is not None:
        path = Path(cfg["resume_ckpt"])
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return path
    if cfg.get("resume_latest"):
        return find_latest_checkpoint(cfg["ckpt_dir"])
    return None


def build_loader(cfg: Dict, rank: int, world_size: int, ddp_enabled: bool) -> Tuple[BlockPerBasinH5Dataset, DataLoader, object | None]:
    print(cfg["basins_file"])
    basins = read_basin_ids(cfg["basins_file"])
    print(len(basins))
    dataset = BlockPerBasinH5Dataset(
        h5_path=cfg["h5_dir"],
        scalers_path=cfg["scalers_path"],
        seq_len=int(cfg["seq_len"]),
        basins=basins,
        target_block_size=int(cfg["target_block_size"]),
        target_block_stride=int(cfg["target_block_stride"]),
        drop_last_block=bool(cfg["target_block_drop_last"]),
    )

    sampler = None
    shuffle = bool(cfg["target_block_shuffle"])
    if ddp_enabled and bool(cfg.get("balanced_sampler", True)):
        sampler = BalancedBlockDistributedSampler(
            dataset,
            batch_size=int(cfg["basin_batch_size"]),
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=int(cfg.get("balanced_sampler_seed", 0)),
            bucket_size=int(cfg.get("balanced_bucket_size", 16)),
            drop_last=bool(cfg.get("balanced_sampler_drop_last", False)),
        )
        shuffle = False
    elif ddp_enabled:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["basin_batch_size"]),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg.get("pin_memory", False)),
        persistent_workers=bool(cfg.get("persistent_workers", False)) and int(cfg["num_workers"]) > 0,
        collate_fn=collate_block_basin_batch,
    )
    return dataset, loader, sampler


def build_model(cfg: Dict, device: torch.device) -> HydroAIBasinPhaseH:
    model = HydroAIBasinPhaseH(
        dims={"dyn": len(cfg["dyn_vars"]), "stat": len(cfg["stat_vars"])},
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        precompute_inputs=bool(cfg["precompute_inputs"]),
        precompute_time_chunk=int(cfg["precompute_time_chunk"]),
        precompute_max_positions=int(cfg.get("precompute_max_positions", 1000000)),
    )
    return model.to(device)


def build_optimizer(cfg: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    raw_model = model.module if hasattr(model, "module") else model
    return torch.optim.AdamW(
        [
            {"params": raw_model.generator.parameters(), "lr": float(cfg["lr_gen"])},
            {"params": raw_model.routing.parameters(), "lr": float(cfg["lr_vel"])},
        ],
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )


def scheduled_lrs_for_epoch(cfg: Dict, current_epoch: int) -> tuple[float, float]:
    total_epochs = int(cfg["epochs"])
    base_lr_gen = float(cfg["lr_gen"])
    base_lr_vel = float(cfg["lr_vel"])

    if total_epochs <= 50:
        if current_epoch >= 21:
            return 1e-4, 1e-5
        if current_epoch >= 11:
            return 5e-4, 5e-5
        return base_lr_gen, base_lr_vel

    m1 = int(math.floor(total_epochs * 0.40)) + 1
    m2 = int(math.floor(total_epochs * 0.70)) + 1
    m3 = int(math.floor(total_epochs * 0.90)) + 1
    if current_epoch >= m3:
        scale = 0.05
    elif current_epoch >= m2:
        scale = 0.1
    elif current_epoch >= m1:
        scale = 0.5
    else:
        scale = 1.0
    return base_lr_gen * scale, base_lr_vel * scale


def _safe_float(value, default=""):
    if value is None:
        return default
    try:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            value = value.detach().float().mean().cpu().item()
        return float(value)
    except Exception:
        return default


def _first_metric(metrics: Dict, keys: tuple[str, ...], default=""):
    for key in keys:
        if key in metrics:
            return _safe_float(metrics.get(key), default=default)
    return default


def _aux_mean(aux, keys: tuple[str, ...]):
    if not isinstance(aux, dict):
        return ""
    for key in keys:
        if key in aux:
            return _safe_float(aux[key], default="")
    return ""


def _path_with_rank(path: Path, rank: int) -> Path:
    suffix = path.suffix
    if suffix:
        return path.with_name(f"{path.stem}.rank{rank}{suffix}")
    return path.with_name(f"{path.name}.rank{rank}.csv")


class LossComponentLogger:
    fields = [
        "rank",
        "epoch",
        "step_in_epoch",
        "global_step",
        "lr_gen",
        "lr_vel",
        "loss_total_step",
        "loss_data_step",
        "loss_total_from_criterion",
        "loss_q",
        "loss_q_mfm",
        "loss_peak_raw",
        "loss_peak_contrib",
        "loss_mb",
        "loss_route_total",
        "loss_route_mass",
        "loss_route_sigma_monotonic",
        "loss_route_hillslope_ratio",
        "loss_route_velocity_smooth",
        "loss_route_sigma_width",
        "loss_route_dispersion_coupling",
        "loss_route_effective_lag",
        "w_q",
        "w_peak",
        "w_balance",
        "w_route",
        "lag_mean",
        "v_mean",
        "basin_ids",
        "sample_ids",
        "block_starts",
        "block_ends",
        "valid_target_counts",
    ]

    def __init__(self, cfg: Dict, rank: int) -> None:
        self.interval = max(1, int(cfg.get("loss_log_interval", 1)))
        self.flush_interval = max(1, int(cfg.get("loss_log_flush_interval", 100)))
        self.rank = int(rank)
        self.rows_since_flush = 0
        self.file = None
        self.writer = None
        if not bool(cfg.get("write_loss_log", False)):
            self.path = None
            return
        raw_path = str(cfg.get("loss_log_file", "") or "")
        self.path = _path_with_rank(Path(raw_path), self.rank) if raw_path else Path(cfg["out_dir"]) / f"loss_components_rank{self.rank}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()
        self.file.flush()

    def enabled(self, step_in_epoch: int) -> bool:
        return self.writer is not None and step_in_epoch % self.interval == 0

    def close(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None

    @staticmethod
    def _join_meta_values(basin_meta: List[Dict], key: str) -> str:
        values = []
        for meta in basin_meta:
            value = meta.get(key, "")
            if torch.is_tensor(value):
                if value.numel() == 0:
                    value = ""
                else:
                    value = value.detach().reshape(-1)[0].cpu().item()
            values.append(str(value))
        return "|".join(values)

    @staticmethod
    def _mean_route_aux(outputs: Dict, keys: tuple[str, ...]):
        return _mean_aux_value(outputs.get("route_aux", []), keys)

    def log(
        self,
        *,
        cfg: Dict,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        loss_value: float,
        data_loss,
        metrics: Dict,
        route_loss,
        route_metrics: Dict,
        outputs: Dict,
        basin_meta: List[Dict],
    ) -> None:
        if not self.enabled(step_in_epoch):
            return
        peak_raw = float(metrics.get("loss_peak", 0.0))
        row = {
            "rank": self.rank,
            "epoch": int(epoch),
            "step_in_epoch": int(step_in_epoch),
            "global_step": int(global_step),
            "lr_gen": float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else "",
            "lr_vel": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else "",
            "loss_total_step": float(loss_value),
            "loss_data_step": _safe_float(data_loss, default=""),
            "loss_total_from_criterion": _first_metric(metrics, ("loss_total",), default=""),
            "loss_q": _first_metric(metrics, ("loss_q", "q_loss"), default=""),
            "loss_q_mfm": _first_metric(metrics, ("loss_q_mfm",), default=""),
            "loss_peak_raw": peak_raw,
            "loss_peak_contrib": float(cfg.get("w_peak", 0.0)) * peak_raw,
            "loss_mb": _first_metric(metrics, ("loss_mb", "mb"), default=""),
            "loss_route_total": _first_metric(route_metrics, ("loss_route_total", "route_loss"), default=_safe_float(route_loss, default="")),
            "loss_route_mass": _first_metric(route_metrics, ("loss_route_mass",), default=""),
            "loss_route_sigma_monotonic": _first_metric(route_metrics, ("loss_route_sigma_monotonic",), default=""),
            "loss_route_hillslope_ratio": _first_metric(route_metrics, ("loss_route_hillslope_ratio",), default=""),
            "loss_route_velocity_smooth": _first_metric(route_metrics, ("loss_route_velocity_smooth",), default=""),
            "loss_route_sigma_width": _first_metric(route_metrics, ("loss_route_sigma_width",), default=""),
            "loss_route_dispersion_coupling": _first_metric(route_metrics, ("loss_route_dispersion_coupling",), default=""),
            "loss_route_effective_lag": _first_metric(route_metrics, ("loss_route_effective_lag",), default=""),
            "w_q": float(cfg.get("w_q", 1.0)),
            "w_peak": float(cfg.get("w_peak", 0.0)),
            "w_balance": float(cfg.get("w_balance", 0.0)),
            "w_route": float(cfg.get("w_route", 1.0)),
            "lag_mean": self._mean_route_aux(outputs, ("lag", "lag_mean", "lag_days", "total_lag_days", "channel_lag_days", "lag_len")),
            "v_mean": self._mean_route_aux(outputs, ("v_mean", "velocity_mean", "mean_velocity", "velocity_mps")),
            "basin_ids": self._join_meta_values(basin_meta, "basin_id"),
            "sample_ids": self._join_meta_values(basin_meta, "sample_id"),
            "block_starts": self._join_meta_values(basin_meta, "block_start"),
            "block_ends": self._join_meta_values(basin_meta, "block_end"),
            "valid_target_counts": self._join_meta_values(basin_meta, "valid_target_count"),
        }
        self.writer.writerow(row)
        self.rows_since_flush += 1
        if self.rows_since_flush >= self.flush_interval:
            self.file.flush()
            self.rows_since_flush = 0


def write_last_block_diagnostics(dataset, out_csv_path: str | Path) -> None:
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = getattr(dataset, "samples", [])
    last_by_basin: Dict[str, Dict] = {}
    for sample in samples:
        basin_id = str(sample.get("basin_id", ""))
        if not basin_id:
            continue
        previous = last_by_basin.get(basin_id)
        key = (int(sample.get("block_start", -1)), int(sample.get("block_end", -1)))
        previous_key = (
            int(previous.get("block_start", -1)),
            int(previous.get("block_end", -1)),
        ) if previous else (-1, -1)
        if previous is None or key > previous_key:
            last_by_basin[basin_id] = sample

    fields = [
        "basin_id",
        "sample_id",
        "block_start",
        "block_end",
        "last_time_len",
        "valid_target_len",
        "timeindex_start",
        "timeindex_end",
        "h5_path",
    ]
    rows: List[Dict] = []
    for basin_id, sample in sorted(last_by_basin.items()):
        block_start = int(sample.get("block_start", 0))
        block_end = int(sample.get("block_end", 0))
        h5_path = Path(sample.get("h5_file", ""))
        timeindex_start = ""
        timeindex_end = ""
        valid_target_len = sample.get("valid_target_count", "")
        try:
            with h5py.File(h5_path, "r") as f:
                target_idx_block = f["target_idx"][block_start:block_end]
                if len(target_idx_block) > 0:
                    timeindex_start = int(target_idx_block[0])
                    timeindex_end = int(target_idx_block[-1])
                if valid_target_len == "" or valid_target_len is None:
                    target_valid = f["target_valid"][block_start:block_end]
                    valid_target_len = int(np.asarray(target_valid).astype(bool).sum())
        except Exception as exc:
            print(f"[last-block diagnostics] warning: failed to inspect {h5_path}: {exc}")
            valid_target_len = "" if valid_target_len is None else valid_target_len

        rows.append(
            {
                "basin_id": basin_id,
                "sample_id": sample.get("sample_id", f"{basin_id}:{block_start}:{block_end}"),
                "block_start": block_start,
                "block_end": block_end,
                "last_time_len": block_end - block_start,
                "valid_target_len": valid_target_len,
                "timeindex_start": timeindex_start,
                "timeindex_end": timeindex_end,
                "h5_path": str(h5_path),
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print(f"[last-block diagnostics] warning: no records written to {out_path}")
        return

    last_lens = np.asarray([int(row["last_time_len"]) for row in rows], dtype=np.float64)
    valid_lens = np.asarray([float(row["valid_target_len"]) for row in rows if row["valid_target_len"] != ""], dtype=np.float64)
    print(f"[last-block diagnostics] wrote {out_path}")
    print(
        "[last-block diagnostics] "
        f"basins={len(rows)} last_time_len min/median/max="
        f"{int(np.min(last_lens))}/{float(np.median(last_lens)):.1f}/{int(np.max(last_lens))}"
    )
    if valid_lens.size > 0:
        below_365 = int(np.sum(valid_lens < 365))
        print(
            "[last-block diagnostics] "
            f"valid_target_len min/median/max={int(np.min(valid_lens))}/{float(np.median(valid_lens)):.1f}/{int(np.max(valid_lens))}"
        )
        print(f"[last-block diagnostics] basins with valid_target_len < 365: {below_365}")


class HighLossLogger:
    fields = [
        "rank",
        "epoch",
        "step_in_epoch",
        "global_step",
        "loss_total_step",
        "loss_data_step",
        "loss_q_step",
        "loss_mb_step",
        "loss_route_step",
        "basin_id",
        "sample_id",
        "block_start",
        "block_end",
        "block_len",
        "valid_target_count",
        "timeindex_start",
        "timeindex_end",
        "q_std_loss",
        "obs_mean",
        "obs_max",
        "obs_p90",
        "sim_mean",
        "sim_max",
        "sim_min",
        "rmse",
        "norm_rmse",
        "bias",
        "rel_bias",
        "lag_mean",
        "v_mean",
    ]

    def __init__(self, cfg: Dict, rank: int) -> None:
        self.threshold = float(cfg.get("debug_loss_threshold", 0.0))
        self.max_records = int(cfg.get("debug_loss_max_records", 2000))
        self.print_limit = int(cfg.get("debug_loss_print_limit", 50))
        self.rank = int(rank)
        self.records_written = 0
        self.printed = 0
        self.warned_missing_q_pred = False
        self.warned_error = False
        raw_path = str(cfg.get("debug_loss_log_file", "") or "")
        self.path = _path_with_rank(Path(raw_path), self.rank) if raw_path else Path(cfg["out_dir"]) / f"high_loss_rank{self.rank}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def enabled(self) -> bool:
        return self.threshold > 0 and self.records_written < self.max_records

    def _append_row(self, row: Dict) -> None:
        write_header = (not self.path.exists()) or self.path.stat().st_size == 0
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.records_written += 1

    def _q_pred_list(self, outputs: Dict, basin_count: int):
        q_pred = outputs.get("q_pred", outputs.get("q_pred_list"))
        if q_pred is None:
            if not self.warned_missing_q_pred:
                print("[HIGH_LOSS] warning: outputs do not contain q_pred; prediction stats will be empty.")
                self.warned_missing_q_pred = True
            return [None] * basin_count
        if isinstance(q_pred, (list, tuple)):
            return list(q_pred)
        if torch.is_tensor(q_pred) and basin_count == 1:
            return [q_pred]
        return [None] * basin_count

    def maybe_log(
        self,
        *,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        loss_value: float,
        data_loss,
        metrics: Dict,
        route_loss,
        route_metrics: Dict,
        outputs: Dict,
        basin_meta: List[Dict],
    ) -> None:
        if loss_value <= self.threshold or not self.enabled():
            return
        try:
            with torch.no_grad():
                q_preds = self._q_pred_list(outputs, len(basin_meta))
                route_aux_list = outputs.get("route_aux", [])
                for basin_index, meta in enumerate(basin_meta):
                    if self.records_written >= self.max_records:
                        break
                    q_pred = q_preds[basin_index] if basin_index < len(q_preds) else None
                    aux = route_aux_list[basin_index] if isinstance(route_aux_list, (list, tuple)) and basin_index < len(route_aux_list) else None
                    row = self._build_row(
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        loss_value=loss_value,
                        data_loss=data_loss,
                        metrics=metrics,
                        route_loss=route_loss,
                        route_metrics=route_metrics,
                        meta=meta,
                        q_pred=q_pred,
                        aux=aux,
                    )
                    self._append_row(row)
                    if self.printed < self.print_limit:
                        print(
                            "[HIGH_LOSS] "
                            f"rank={self.rank} epoch={epoch} step={step_in_epoch} loss={loss_value:.3f} "
                            f"basin={row['basin_id']} block={row['block_start']}:{row['block_end']} "
                            f"valid={row['valid_target_count']} q_std_loss={row['q_std_loss']} rmse={row['rmse']}",
                            flush=True,
                        )
                        self.printed += 1
        except Exception as exc:
            if not self.warned_error:
                print(f"[HIGH_LOSS] warning: diagnostics failed and training will continue: {exc}")
                self.warned_error = True

    def _build_row(self, *, epoch, step_in_epoch, global_step, loss_value, data_loss, metrics, route_loss, route_metrics, meta, q_pred, aux) -> Dict:
        block_start = int(meta.get("block_start", 0))
        block_end = int(meta.get("block_end", 0))
        q_std_value = self._meta_scalar(meta, "q_std_loss")
        timeindex_start, timeindex_end = self._timeindex_bounds(meta)
        stats = self._flow_stats(meta, q_pred, q_std_value)
        return {
            "rank": self.rank,
            "epoch": int(epoch),
            "step_in_epoch": int(step_in_epoch),
            "global_step": int(global_step),
            "loss_total_step": float(loss_value),
            "loss_data_step": _safe_float(data_loss, default=""),
            "loss_q_step": _first_metric(metrics, ("loss_q", "q_loss"), default=""),
            "loss_mb_step": _first_metric(metrics, ("loss_mb", "mb"), default=""),
            "loss_route_step": _first_metric(route_metrics, ("loss_route_total", "route_loss"), default=_safe_float(route_loss, default="")),
            "basin_id": meta.get("basin_id", ""),
            "sample_id": meta.get("sample_id", f"{meta.get('basin_id', '')}:{block_start}:{block_end}"),
            "block_start": block_start,
            "block_end": block_end,
            "block_len": block_end - block_start,
            **stats,
            "valid_target_count": int(meta.get("valid_target_count", stats.get("valid_target_count", 0))),
            "timeindex_start": timeindex_start,
            "timeindex_end": timeindex_end,
            "q_std_loss": q_std_value,
            "lag_mean": _aux_mean(aux, ("lag", "lag_mean", "lag_days", "total_lag_days", "channel_lag_days", "lag_len")),
            "v_mean": _aux_mean(aux, ("v_mean", "velocity_mean", "mean_velocity", "velocity_mps")),
        }

    @staticmethod
    def _meta_scalar(meta: Dict, key: str):
        if key not in meta:
            return ""
        value = meta[key]
        if torch.is_tensor(value):
            if value.numel() == 0:
                return ""
            return float(value.detach().reshape(-1)[0].cpu())
        return _safe_float(value, default="")

    @staticmethod
    def _timeindex_bounds(meta: Dict) -> tuple[object, object]:
        # target_idx is local to the sliced sequence; target_timeindex preserves the original H5 time index.
        idx = meta.get("target_timeindex", meta.get("target_idx"))
        if torch.is_tensor(idx):
            if idx.numel() == 0:
                return "", ""
            flat = idx.detach().reshape(-1).cpu()
            return int(flat[0]), int(flat[-1])
        try:
            arr = np.asarray(idx).reshape(-1)
            if arr.size == 0:
                return "", ""
            return int(arr[0]), int(arr[-1])
        except Exception:
            return "", ""

    @staticmethod
    def _flow_stats(meta: Dict, q_pred, q_std_value) -> Dict:
        keys = [
            "valid_target_count",
            "obs_mean",
            "obs_max",
            "obs_p90",
            "sim_mean",
            "sim_max",
            "sim_min",
            "rmse",
            "norm_rmse",
            "bias",
            "rel_bias",
        ]
        empty = {key: "" for key in keys}
        if q_pred is None or "q_true" not in meta or "q_valid" not in meta:
            return empty
        q_valid = meta["q_valid"].detach().reshape(-1).bool().cpu()
        q_true = meta["q_true"].detach().reshape(-1).float().cpu()
        q_pred_flat = q_pred.detach().reshape(-1).float().cpu()
        n = min(q_valid.numel(), q_true.numel(), q_pred_flat.numel())
        if n <= 0:
            return empty
        q_valid = q_valid[:n]
        q_true = q_true[:n]
        q_pred_flat = q_pred_flat[:n]
        valid_count = int(q_valid.sum().item())
        empty["valid_target_count"] = valid_count
        if valid_count == 0:
            return empty
        q_true_valid = q_true[q_valid]
        q_pred_valid = q_pred_flat[q_valid]
        diff = q_pred_valid - q_true_valid
        rmse = torch.sqrt(torch.mean(diff ** 2))
        bias = torch.mean(diff)
        obs_mean = torch.mean(q_true_valid)
        try:
            obs_p90 = torch.quantile(q_true_valid.float(), 0.9)
        except Exception:
            obs_p90 = None
        q_std = q_std_value if isinstance(q_std_value, (int, float)) else 0.0
        norm_rmse = rmse / (float(q_std) + 0.1)
        return {
            "valid_target_count": valid_count,
            "obs_mean": float(obs_mean.cpu()),
            "obs_max": float(torch.max(q_true_valid).cpu()),
            "obs_p90": "" if obs_p90 is None else float(obs_p90.cpu()),
            "sim_mean": float(torch.mean(q_pred_valid).cpu()),
            "sim_max": float(torch.max(q_pred_valid).cpu()),
            "sim_min": float(torch.min(q_pred_valid).cpu()),
            "rmse": float(rmse.cpu()),
            "norm_rmse": float(norm_rmse.cpu()),
            "bias": float(bias.cpu()),
            "rel_bias": float((bias / (torch.abs(obs_mean) + 1e-6)).cpu()),
        }


def reduce_epoch_stats(component_sums: Dict[str, float], step_count: int, device: torch.device) -> tuple[Dict[str, float], int]:
    keys = ["loss", "q", "peak_raw", "peak_contrib", "mb", "route"]
    values = [float(component_sums.get(key, 0.0)) for key in keys] + [float(step_count)]
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = {key: float(tensor[i].item()) for i, key in enumerate(keys)}
        return reduced, int(tensor[-1].item())
    return {key: float(component_sums.get(key, 0.0)) for key in keys}, step_count


def checkpoint_monitor_loss(cfg: Dict, epoch_metrics: Dict[str, float]) -> float:
    return (
        float(cfg.get("w_q", 1.0)) * float(epoch_metrics.get("q", 0.0))
        + float(cfg.get("w_balance", 0.0)) * float(epoch_metrics.get("mb", 0.0))
        + float(epoch_metrics.get("route", 0.0))
    )


def print_peak_loss_config(cfg: Dict) -> None:
    print("[PeakFlowLoss]", flush=True)
    print("mode=bounded_huber", flush=True)
    print(f"quantile={float(cfg.get('peak_quantile', 0.9))}", flush=True)
    print(f"min_valid_count={int(cfg.get('peak_min_valid_count', 100))}", flush=True)
    print(f"min_peak_count={int(cfg.get('peak_min_peak_count', 10))}", flush=True)
    print(f"huber_beta={float(cfg.get('peak_huber_beta', 1.0))}", flush=True)
    print(f"peak_weight={float(cfg.get('peak_weight', 1.0))}", flush=True)
    print(f"outer_w_peak={float(cfg.get('w_peak', 0.0))}", flush=True)


def shutdown_dataloader(loader) -> None:
    if loader is None:
        return

    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

    try:
        loader._iterator = None
    except Exception:
        pass


def train_one_epoch(
    *,
    cfg: Dict,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    scaler,
    loader: DataLoader,
    device: torch.device,
    global_step: int,
    high_loss_logger: HighLossLogger | None = None,
    loss_component_logger: LossComponentLogger | None = None,
) -> tuple[Dict[str, float], int]:
    model.train()
    route_w = route_weights(cfg)
    component_sums = {"loss": 0.0, "q": 0.0, "peak_raw": 0.0, "peak_contrib": 0.0, "mb": 0.0, "route": 0.0}
    step_count = 0
    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    progress = tqdm(loader, desc=f"epoch {epoch}", disable=not is_main_process(), dynamic_ncols=True)

    for step_in_epoch, batch in enumerate(progress, start=1):
        basin_meta = batch["basin_meta"]
        q_mean_global = batch["q_mean_global"].to(device, non_blocking=True)
        q_std_global = batch["q_std_global"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                None,
                None,
                basin_meta=basin_meta,
                max_lag=int(cfg["max_lag"]),
                generator_chunk_size=int(cfg.get("generator_chunk_size", 8192)),
                clear_cache=bool(cfg.get("clear_cache", False)),
                use_checkpoint=bool(cfg.get("use_checkpoint", False)),
                is_basin=True,
            )
            data_loss, metrics = criterion(outputs, basin_meta, q_mean_global, q_std_global)
            route_loss, route_metrics = compute_routing_regularization(model, outputs, basin_meta, device, route_w)
            loss = data_loss + route_loss

        if not torch.isfinite(loss.detach()):
            print(f"[warn] skip non-finite loss at epoch={epoch}, step={step_in_epoch}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss).backward()
            if float(cfg.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if float(cfg.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

        global_step += 1
        step_count += 1
        loss_value = float(loss.detach().cpu())
        if high_loss_logger is not None:
            high_loss_logger.maybe_log(
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                loss_value=loss_value,
                data_loss=data_loss,
                metrics=metrics,
                route_loss=route_loss,
                route_metrics=route_metrics,
                outputs=outputs,
                basin_meta=basin_meta,
            )
        if loss_component_logger is not None:
            loss_component_logger.log(
                cfg=cfg,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                optimizer=optimizer,
                loss_value=loss_value,
                data_loss=data_loss,
                metrics=metrics,
                route_loss=route_loss,
                route_metrics=route_metrics,
                outputs=outputs,
                basin_meta=basin_meta,
            )
        component_sums["loss"] += loss_value
        component_sums["q"] += float(metrics.get("loss_q", 0.0))
        peak_raw = float(metrics.get("loss_peak", 0.0))
        component_sums["peak_raw"] += peak_raw
        component_sums["peak_contrib"] += float(cfg.get("w_peak", 0.0)) * peak_raw
        component_sums["mb"] += float(metrics.get("loss_mb", 0.0))
        component_sums["route"] += float(route_metrics.get("loss_route_total", 0.0))
        if is_main_process() and step_in_epoch % int(cfg.get("log_interval", 10)) == 0:
            progress.set_postfix(_progress_postfix(loss_value, metrics, route_metrics, outputs))
        maybe_empty_cuda_cache(global_step, enabled=bool(cfg.get("clear_cache", False)), interval=int(cfg.get("empty_cache_interval", 20)))
        del outputs, data_loss, route_loss, loss, metrics, route_metrics

    reduced_sums, reduced_steps = reduce_epoch_stats(component_sums, step_count, device)
    denom = max(reduced_steps, 1)
    epoch_metrics = {key: value / denom for key, value in reduced_sums.items()}
    return epoch_metrics, global_step


def train(cfg: Dict) -> None:
    cfg = setup_run_dir(update_config(cfg))
    configure_run_artifact_dirs(cfg["run_dir"])
    seed_everything(int(cfg["seed"]))

    rank, world_size, ddp_enabled, local_rank = setup_distributed(cfg.get("local_rank", None))
    device = resolve_device(local_rank, requested=str(cfg.get("device", "cuda")))
    print(
        f"DDP rank={rank}, local_rank={local_rank}, world_size={world_size}, "
        f"device={device}, current_cuda_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    dataset = None
    loader = None
    sampler = None
    loss_component_logger = None
    normal_completed = False

    try:
        if is_main_process():
            save_json(cfg, str(cfg["run_dir"] / "config_used.json"))
            build_or_update_h5_if_needed(cfg)
        if ddp_enabled:
            ddp_barrier(local_rank)

        dataset, loader, sampler = build_loader(cfg, rank, world_size, ddp_enabled)
        if is_main_process() and bool(cfg.get("write_last_block_diagnostics", False)):
            diag_path = Path(cfg.get("last_block_diagnostics_file") or cfg["out_dir"] / "last_block_diagnostics.csv")
            write_last_block_diagnostics(dataset, diag_path)
        if ddp_enabled:
            ddp_barrier(local_rank)

        model = build_model(cfg, device)
        optimizer = build_optimizer(cfg, model)
        high_loss_logger = HighLossLogger(cfg, rank) if float(cfg.get("debug_loss_threshold", 0.0)) > 0 else None
        loss_component_logger = LossComponentLogger(cfg, rank)
        if loss_component_logger.path is not None:
            print(f"[loss-log] writing component losses to {loss_component_logger.path}", flush=True)
        # scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("use_amp", False)) and device.type == "cuda")
        scaler = torch.amp.GradScaler( "cuda",enabled=bool(cfg.get("use_amp", False)) and device.type == "cuda",)
        criterion = build_criterion(cfg)
        if is_main_process() and str(cfg.get("Loss", "")).upper() == "MFM":
            print_peak_loss_config(cfg)

        resume_path = resolve_resume_path(cfg)
        start_epoch = 1
        global_step = 0
        best_loss = float("inf")
        best_epoch = 0
        if resume_path is not None:
            ckpt = load_checkpoint(resume_path, model=model, optimizer=optimizer, scaler=scaler, map_location="cpu")
            completed_epoch = int(ckpt.get("completed_epoch", ckpt.get("epoch", 0)))
            start_epoch = completed_epoch + 1
            global_step = int(ckpt.get("global_step", 0))
            previous_cfg = ckpt.get("config", {}) or {}
            previous_metric = previous_cfg.get("best_checkpoint_metric")
            if previous_metric == "q_balance_route":
                best_loss = float(ckpt.get("best_loss", float("inf")))
                best_epoch = int(ckpt.get("best_epoch", ckpt.get("completed_epoch", 0)))
            else:
                best_loss = float("inf")
                best_epoch = 0
            if is_main_process():
                if previous_metric != "q_balance_route":
                    print(
                        "[resume] previous checkpoint used a different best-model metric; "
                        "best monitor state was reset.",
                        flush=True,
                    )
                print(
                    f"[resume] loaded {resume_path}; start_epoch={start_epoch}, "
                    f"global_step={global_step}, best_epoch={best_epoch}, "
                    f"best_monitor_loss={best_loss:.6f}"
                )

        if ddp_enabled and device.type == "cuda":
            model = DDP(model, device_ids=[int(local_rank)], output_device=int(local_rank), find_unused_parameters=False)
        elif ddp_enabled:
            model = DDP(model, find_unused_parameters=False)

        if is_main_process():
            print(f"Training basins/blocks: dataset_blocks={len(dataset)}, world_size={world_size}, device={device}")
            if sampler is not None and hasattr(sampler, "raw_counts"):
                print(f"[Sampler] raw_counts_per_rank={sampler.raw_counts}")
                print(f"[Sampler] raw_loads_per_rank={sampler.raw_loads}")
                print(f"[Sampler] yielded_samples_per_rank={len(sampler)}")
                print(f"[Sampler] drop_last={sampler.drop_last}")
            print("Training is intentionally kept simple; run validation.py or run_eval.sh separately after checkpoints are saved.")

        for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            lr_gen_now, lr_vel_now = scheduled_lrs_for_epoch(cfg, epoch)
            optimizer.param_groups[0]["lr"] = lr_gen_now
            optimizer.param_groups[1]["lr"] = lr_vel_now
            if is_main_process():
                print(
                    f"[lr update] epoch={epoch} "
                    f"lr_gen={optimizer.param_groups[0]['lr']:.2e} "
                    f"lr_vel={optimizer.param_groups[1]['lr']:.2e}"
                )
            epoch_metrics, global_step = train_one_epoch(
                cfg=cfg,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                loader=loader,
                device=device,
                global_step=global_step,
                high_loss_logger=high_loss_logger,
                loss_component_logger=loss_component_logger,
            )
            epoch_loss = float(epoch_metrics.get("loss", 0.0))
            monitor_loss = checkpoint_monitor_loss(cfg, epoch_metrics)
            if is_main_process():
                print(
                    f"Epoch {epoch:03d} finished: total_loss={epoch_loss:.6f} "
                    f"monitor_loss={monitor_loss:.6f} "
                    f"q={epoch_metrics.get('q', 0.0):.6f} "
                    f"peak_raw={epoch_metrics.get('peak_raw', 0.0):.6f} "
                    f"peak_contrib={epoch_metrics.get('peak_contrib', 0.0):.6f} "
                    f"mb={epoch_metrics.get('mb', 0.0):.6f} "
                    f"route={epoch_metrics.get('route', 0.0):.6f}"
                )
                is_new_best = monitor_loss < best_loss
                if is_new_best:
                    best_loss = monitor_loss
                    best_epoch = epoch
                ckpt_path = cfg["ckpt_dir"] / f"ckpt_epoch{epoch:03d}.pth"
                latest_path = cfg["ckpt_dir"] / "latest.pth"
                best_path = cfg["out_dir"] / "best_model.pth"
                print(f"Saving checkpoint: {ckpt_path}")
                save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    completed_epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    config=cfg,
                )
                print(f"Saved checkpoint: {ckpt_path}")
                shutil.copy2(ckpt_path, latest_path)
                print(f"Updated latest checkpoint: {latest_path}")
                if is_new_best:
                    save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        completed_epoch=epoch,
                        global_step=global_step,
                        best_loss=best_loss,
                        best_epoch=best_epoch,
                        config=cfg,
                    )
                    print(f"Updated best model: {best_path}")
                print(
                    f"Best epoch: {best_epoch}, "
                    f"best_monitor_loss={best_loss:.6f}, monitor=q_balance_route"
                )
            if ddp_enabled:
                ddp_barrier(local_rank)
        normal_completed = True
    finally:
        is_rank0 = rank == 0
        if loss_component_logger is not None:
            loss_component_logger.close()
        shutdown_dataloader(loader)
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device if device.type == "cuda" else None)
            except Exception as exc:
                print(f"[cleanup] CUDA synchronize warning: {exc}", flush=True)
        if ddp_enabled and dist.is_available() and dist.is_initialized():
            try:
                ddp_barrier(local_rank)
            except Exception as exc:
                print(f"[cleanup] final barrier warning on rank {rank}: {exc}", flush=True)
        cleanup_distributed()
        try:
            del loader
            del dataset
            del sampler
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if normal_completed and is_rank0:
            print("Training finished.", flush=True)


def main() -> None:
    train(get_args())


if __name__ == "__main__":
    main()
