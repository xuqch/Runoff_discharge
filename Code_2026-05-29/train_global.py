from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from common import build_criterion, configure_run_artifact_dirs, maybe_empty_cuda_cache
from config import get_args, update_config


if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from balanced_block_sampler import BalancedBlockDistributedSampler
    from dataset_global import PerBasinBlockH5Dataset
    from model_hydroai_basin import HydroAIBasinModel


class Trainer:
    def __init__(self, cfg: Dict) -> None:
        from utils import set_seed

        self.cfg = self._setup_run_dir(update_config(cfg))
        set_seed(self.cfg["seed"])
        self.artifact_dirs = configure_run_artifact_dirs(self.cfg["run_dir"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.ddp_enabled = self.world_size > 1

        self.device: "torch.device | None" = None
        self.rank = 0
        self.is_rank0 = True

        self.generator_chunk_size = int(self.cfg.get("generator_chunk_size", 4096))
        self.empty_cache_interval = int(self.cfg.get("empty_cache_interval", 20))
        self.clear_cache = bool(self.cfg.get("clear_cache", False))
        self.use_checkpoint = bool(self.cfg.get("use_checkpoint", False))
        self.log_interval = max(1, int(self.cfg.get("log_interval", 10)))
        self.profile_step_timing = bool(self.cfg.get("profile_step_timing", False))
        self.profile_step_timing_print_every = max(1, int(self.cfg.get("profile_step_timing_print_every", 10)))
        self.debug_nonfinite = bool(self.cfg.get("debug_nonfinite", False))
        self.route_weights = self._route_weights()

        self.dataset: "PerBasinBlockH5Dataset | None" = None
        self.sampler = None
        self.loader: "DataLoader | None" = None
        self.raw_model: "HydroAIBasinModel | None" = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.use_amp = False
        self.profile_step_timing_csv: Path | None = None
        self.resume_path: Path | None = None
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

    def _h5_ready_marker(self) -> Path:
        return Path(self.cfg["h5_path"]) / ".ready"

    def _h5_failed_marker(self) -> Path:
        return Path(self.cfg["h5_path"]) / ".failed"

    def _resolve_resume_ckpt(self) -> Path | None:
        resume_ckpt = self.cfg.get("resume_ckpt")
        if resume_ckpt is not None:
            resume_path = Path(resume_ckpt)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            return resume_path

        if not bool(self.cfg.get("resume_latest", False)):
            return None

        ckpt_dir = Path(self.cfg["ckpt_dir"])
        if not ckpt_dir.exists():
            return None

        candidates = sorted(ckpt_dir.glob("ckpt_epoch*.pth"))
        if not candidates:
            return None
        return candidates[-1]

    @staticmethod
    def _load_local_checkpoint(ckpt_path: Path) -> Dict:
        import torch

        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def _restore_checkpoint_state(self) -> None:
        assert self.raw_model is not None
        assert self.optimizer is not None
        assert self.scaler is not None

        resume_path = self._resolve_resume_ckpt()
        self.resume_path = resume_path
        if resume_path is None:
            self.start_epoch = 0
            self.global_step = 0
            self.best_loss = float("inf")
            return

        ckpt = self._load_local_checkpoint(resume_path)
        self.raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        scaler_state = ckpt.get("scaler")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        completed_epoch = int(ckpt.get("epoch", 0))
        self.start_epoch = completed_epoch
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_loss = float(ckpt.get("best_loss", ckpt.get("epoch_loss_mean", float("inf"))))

        if self.is_rank0:
            print(
                f"[resume] loaded checkpoint {resume_path}, "
                f"start_epoch={completed_epoch + 1}, global_step={self.global_step}"
            )

    def _nan_report_path(self) -> Path:
        return self._resolve_rank_csv_path(self.cfg["out_dir"] / "nan_batches.jsonl", rank=self.rank, world_size=self.world_size)

    @staticmethod
    def _tensor_nonfinite_count(tensor) -> int:
        import torch

        if tensor is None:
            return 0
        return int((~torch.isfinite(tensor.detach())).sum().item())

    def _write_nan_report(
        self,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        basin_meta: List[Dict],
        outputs: Dict | None,
        metrics: Dict | None,
        route_metrics: Dict | None,
        loss=None,
        loss_data=None,
        loss_route=None,
    ) -> None:
        report = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "step_in_epoch": int(step_in_epoch),
            "basins": [],
            "metrics": {},
            "route_metrics": {},
            "loss": None if loss is None else float(loss.detach().cpu()),
            "loss_data": None if loss_data is None else float(loss_data.detach().cpu()),
            "loss_route": None if loss_route is None else float(loss_route.detach().cpu()),
        }

        for meta_idx, meta in enumerate(basin_meta):
            q_true = meta["q_true"].detach().cpu()
            q_valid = meta["q_valid"].detach().cpu().bool()
            q_true_valid = q_true[q_valid]
            basin_report = {
                "basin_id": str(meta.get("basin_id", "")),
                "block_start": int(meta.get("block_start", 0)),
                "block_end": int(meta.get("block_end", 0)),
                "prefix_len": int(meta.get("prefix_len", 0)),
                "q_true_valid_count": int(q_valid.sum().item()),
                "q_true_valid_nonfinite_count": int((~q_true_valid.isfinite()).sum().item()) if q_true_valid.numel() > 0 else 0,
            }
            if outputs is not None and meta_idx < len(outputs.get("q_pred", [])):
                basin_report["q_pred_nonfinite_count"] = self._tensor_nonfinite_count(outputs["q_pred"][meta_idx])
            if outputs is not None and meta_idx < len(outputs.get("runoff_m3s", [])):
                basin_report["runoff_m3s_nonfinite_count"] = self._tensor_nonfinite_count(outputs["runoff_m3s"][meta_idx])
            report["basins"].append(basin_report)

        if metrics is not None:
            for key, value in metrics.items():
                report["metrics"][key] = float(value)
        if route_metrics is not None:
            for key, value in route_metrics.items():
                report["route_metrics"][key] = float(value)

        path = self._nan_report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

    def _outputs_have_nonfinite(self, outputs: Dict, basin_meta: List[Dict]) -> bool:
        import torch

        for tensors in (outputs.get("q_pred", []), outputs.get("runoff_m3s", []), outputs.get("runoff", [])):
            for tensor in tensors:
                if tensor is not None and not torch.isfinite(tensor.detach()).all():
                    return True
        for aux in outputs.get("route_aux_for_loss", outputs.get("route_aux", [])):
            for value in aux.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value.detach()).all():
                    return True

        q_pred_list = outputs.get("q_pred", [])
        for meta_idx, meta in enumerate(basin_meta):
            q_true = meta["q_true"].detach().reshape(-1)
            q_valid = meta["q_valid"].detach().reshape(-1).bool()

            if q_valid.numel() != q_true.numel():
                return True
            if bool(q_valid.any()):
                q_true_valid = q_true[q_valid]
                if not torch.isfinite(q_true_valid).all():
                    return True

            if meta_idx < len(q_pred_list):
                q_pred = q_pred_list[meta_idx]
                q_pred_flat = q_pred.detach().reshape(-1)
                if q_pred_flat.numel() != q_true.numel():
                    return True

                q_valid_on_pred = q_valid.to(q_pred_flat.device)
                q_true_flat = q_true.to(q_pred_flat.device)
                finite_pair_mask = (
                    q_valid_on_pred
                    & torch.isfinite(q_true_flat)
                    & torch.isfinite(q_pred_flat)
                )
                if int(finite_pair_mask.sum().item()) <= 0:
                    return True
        return False

    @staticmethod
    def _supervision_summary(outputs: Dict, basin_meta: List[Dict]) -> str:
        import torch

        parts: List[str] = []
        for meta_idx, meta in enumerate(basin_meta[:3]):
            basin_id = str(meta.get("basin_id", meta.get("sample_id", "")))
            q_valid = meta["q_valid"].detach().reshape(-1).bool()
            q_true = meta["q_true"].detach().reshape(-1)
            q_valid_count = int(q_valid.sum().item())
            finite_pairs = 0
            q_pred_nonfinite = -1
            if meta_idx < len(outputs.get("q_pred", [])):
                q_pred = outputs["q_pred"][meta_idx].detach().reshape(-1)
                q_pred_nonfinite = int((~torch.isfinite(q_pred)).sum().item())
                finite_mask = q_valid.to(q_pred.device) & torch.isfinite(q_true.to(q_pred.device)) & torch.isfinite(q_pred)
                finite_pairs = int(finite_mask.sum().item())
            parts.append(
                f"{basin_id}(q_valid={q_valid_count}, finite_pairs={finite_pairs}, q_pred_nonfinite={q_pred_nonfinite})"
            )
        return "; ".join(parts)

    def _handle_nonfinite_step(
        self,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        basin_meta: List[Dict],
        outputs: Dict,
        metrics: Dict,
        route_metrics: Dict,
        loss,
        loss_data,
        loss_route,
    ) -> bool:
        import torch

        nonfinite = not torch.isfinite(loss.detach()).all()
        nonfinite = nonfinite or not torch.isfinite(loss_data.detach()).all()
        nonfinite = nonfinite or not torch.isfinite(loss_route.detach()).all()

        if not nonfinite and self.debug_nonfinite:
            nonfinite = self._outputs_have_nonfinite(outputs, basin_meta)

        for key, value in metrics.items():
            if not np.isfinite(float(value)):
                nonfinite = True
                break
        if not nonfinite:
            for key, value in route_metrics.items():
                if not np.isfinite(float(value)):
                    nonfinite = True
                    break
        if not nonfinite:
            return False

        self._write_nan_report(
            epoch=epoch,
            global_step=global_step,
            step_in_epoch=step_in_epoch,
            basin_meta=basin_meta,
            outputs=outputs,
            metrics=metrics,
            route_metrics=route_metrics,
            loss=loss,
            loss_data=loss_data,
            loss_route=loss_route,
        )
        basin_ids = [str(meta.get("basin_id", "")) for meta in basin_meta]
        print(
            f"[warn] Non-finite training values detected at epoch={epoch}, step={step_in_epoch}, "
            f"global_step={global_step}, basins={basin_ids}. "
            f"Details written to {self._nan_report_path()} - skipping this batch."
        )
        return True

    @staticmethod
    def _sync_time(device: "torch.device") -> float:
        import torch

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        return time.perf_counter()

    @staticmethod
    def _dist_barrier(device: "torch.device", local_rank: int) -> None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        if device.type == "cuda":
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

    def _sync_skip_step(self, local_skip: bool) -> bool:
        import torch
        import torch.distributed as dist

        if not self.ddp_enabled or not dist.is_available() or not dist.is_initialized():
            return local_skip
        assert self.device is not None
        flag = torch.tensor([1 if local_skip else 0], device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @staticmethod
    def _resolve_rank_csv_path(path_like: str | Path, rank: int, world_size: int) -> Path:
        path = Path(path_like)
        if world_size > 1:
            suffix = path.suffix or ".csv"
            path = path.with_name(f"{path.stem}_rank{rank}{suffix}")
        return path

    @staticmethod
    def _append_step_timing_row(path: Path, row: Dict[str, object]) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "rank", "local_rank", "world_size", "epoch", "global_step", "step_in_epoch",
            "basin_ids", "block_spans", "prefix_lens",
            "model_forward_s", "forward_s", "loss_s", "forward_plus_loss_s",
            "backward_s", "grad_unscale_clip_s", "optimizer_s", "step_total_s",
            "backward_share", "forward_share",
        ]
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _setup_run_dir(self, cfg: Dict) -> Dict:
        repo_runs_dir = Path(__file__).resolve().parent / "runs"
        if cfg.get("run_dir") is None:
            now = datetime.now()
            run_name = f"run_phaseh_{now.month:02d}{now.day:02d}_{now.hour + 8:02d}{now.minute:02d}_seed{cfg['seed']}"
            cfg["run_dir"] = repo_runs_dir / run_name
        cfg["run_dir"].mkdir(parents=True, exist_ok=True)
        cfg["out_dir"] = cfg["run_dir"]
        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        cfg["ckpt_dir"] = cfg["out_dir"] / "checkpoints"
        cfg["ckpt_dir"].mkdir(parents=True, exist_ok=True)
        scalers_path = Path(cfg["scalers_path"])
        cfg["scalers_path"] = scalers_path if scalers_path.is_absolute() else repo_runs_dir / scalers_path
        h5_path = Path(cfg["h5_path"])
        cfg["h5_path"] = h5_path if h5_path.is_absolute() else repo_runs_dir / h5_path
        return cfg

    @staticmethod
    def _mean_route_stat(outputs: Dict, key: str) -> float:
        values = [float(aux[key].mean().detach().cpu()) for aux in outputs.get("route_aux", []) if key in aux]
        return sum(values) / len(values) if values else float("nan")

    @staticmethod
    def _mean_route_scalar(outputs: Dict, key: str) -> float:
        values = [float(aux[key].detach().cpu()) for aux in outputs.get("route_aux", []) if key in aux]
        return sum(values) / len(values) if values else float("nan")

    def _route_weights(self) -> Dict[str, float]:
        return {
            "w_route_mass": float(self.cfg.get("w_route_mass", 1e-3)),
            "w_route_sigma_monotonic": float(self.cfg.get("w_route_sigma_monotonic", 1e-2)),
            "w_route_hillslope_ratio": float(self.cfg.get("w_route_hillslope_ratio", 1e-2)),
            "w_route_velocity_smooth": float(self.cfg.get("w_route_velocity_smooth", 1e-3)),
            "w_route_sigma_width": float(self.cfg.get("w_route_sigma_width", 1e-4)),
            "w_route_dispersion_coupling": float(self.cfg.get("w_route_dispersion_coupling", 1e-4)),
            "w_route_effective_lag": float(self.cfg.get("w_route_effective_lag", 0.0)),
        }

    def _compute_routing_regularization(
        self,
        model: "HydroAIBasinModel",
        outputs: Dict,
        basin_meta: List[Dict],
        device: "torch.device",
        weights: Dict[str, float],
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        import torch

        zero = torch.zeros((), device=device)
        route_aux_list = outputs.get("route_aux_for_loss", outputs.get("route_aux", []))
        runoff_m3s_list = outputs.get("runoff_m3s_for_loss", outputs.get("runoff_m3s", []))
        if not route_aux_list or not runoff_m3s_list:
            return zero, {"loss_route_total": 0.0}
        total_route = zero
        accum = {k.replace("w_", ""): 0.0 for k in weights}
        count = 0
        for runoff_m3s, aux, meta in zip(runoff_m3s_list, route_aux_list, basin_meta):
            finite_runoff = torch.isfinite(runoff_m3s)
            if not torch.any(finite_runoff):
                continue
            runoff_for_reg = torch.where(finite_runoff, runoff_m3s, torch.zeros_like(runoff_m3s))
            reg = model.routing.regularization(
                dist_map_m=meta["dist_m"].to(device, non_blocking=True),
                aux=aux,
                runoff_m3s=runoff_for_reg,
            )
            basin_route = zero
            for key, weight in weights.items():
                reg_key = key.replace("w_", "")
                reg_value = reg[reg_key]
                if not torch.isfinite(reg_value.detach()).all():
                    continue
                basin_route = basin_route + weight * reg_value
                accum[reg_key] += float(reg_value.detach().cpu())
            total_route = total_route + basin_route
            count += 1
        total_route = total_route / max(count, 1)
        metrics = {"loss_route_total": float(total_route.detach().cpu())}
        for reg_key, value in accum.items():
            metrics[f"loss_{reg_key}"] = value / max(count, 1)
        return total_route, metrics

    def _build_train_h5_if_needed(self) -> None:
        from build_h5 import create_h5_files
        from fit_scalers import fit_scalers_on_nc_dir
        from utils import resolve_input_path

        h5_path = Path(self.cfg["h5_path"])
        ready_marker = self._h5_ready_marker()
        failed_marker = self._h5_failed_marker()

        if not self.cfg["scalers_path"].exists():
            fit_scalers_on_nc_dir(
                data_dir=self.cfg["data_dir"],
                out_path=self.cfg["scalers_path"],
                q_file=self.cfg["q_file"],
                dyn_vars=self.cfg["dyn_vars"],
                stat_vars=self.cfg["stat_vars"],
                mask_var=self.cfg["mask_var"],
                qobs_var=self.cfg["qobs_var"],
                time_name=self.cfg["time_name"],
            )
        if self.cfg["rebuild_h5"] and self.cfg["h5_path"].exists():
            if self.cfg["h5_path"].is_dir():
                shutil.rmtree(self.cfg["h5_path"])
            else:
                self.cfg["h5_path"].unlink()

        # 05-22 behavior: if the H5 cache directory already exists, trust it.
        # Do not require manifest.csv, do not delete the cache, and do not run incremental
        # missing-cache checks. The dataset loads <h5_path>/<basin_id>.h5 according to q_file.
        if h5_path.exists():
            if not h5_path.is_dir():
                raise NotADirectoryError(f"Expected per-basin H5 directory, got file: {h5_path}")
            failed_marker.unlink(missing_ok=True)
            ready_marker.write_text("ready\n", encoding="utf-8")
            build_q_file = resolve_input_path(self.cfg["q_file"], search_roots=[self.cfg["data_dir"]])
            if self.is_rank0:
                h5_count = sum(1 for _ in h5_path.glob("*.h5"))
                print(f"Using existing H5 cache directory: {h5_path} ({h5_count} .h5 files)")
                print(f"Dataset will be loaded by basin_id order from q_file: {build_q_file}")
            return

        ready_marker.unlink(missing_ok=True)
        failed_marker.unlink(missing_ok=True)
        build_q_file = resolve_input_path(self.cfg["q_file"], search_roots=[self.cfg["data_dir"]])
        if self.is_rank0:
            print(f"H5 cache directory not found; building from basin table: {build_q_file}")

        try:
            create_h5_files(
                nc_dir=self.cfg["nc_dir"],
                q_file=build_q_file,
                scalers_path=self.cfg["scalers_path"],
                out_file=self.cfg["h5_path"],
                dyn_vars=self.cfg["dyn_vars"],
                stat_vars=self.cfg["stat_vars"],
                seq_length=int(self.cfg["seq_len"]),
                mask_var=self.cfg["mask_var"],
                dist_var=self.cfg["dist_var"],
                time_name=self.cfg["time_name"],
                qobs_var=self.cfg["qobs_var"],
                precip_var=self.cfg["precip_var"],
                pet_var=self.cfg.get("pet_var"),
                pet_method=self.cfg.get("pet_method", "hamon"),
                target_start_date=self.cfg["train_start_date"],
                target_end_date=self.cfg["train_end_date"],
                fraction_var=self.cfg["fraction_var"],
                water_year_start_month=int(self.cfg.get("budyko_year_start_month", 10)),
                h5_build_workers=int(self.cfg.get("h5_build_workers", 1)),
            )
        except Exception as exc:
            failed_marker.parent.mkdir(parents=True, exist_ok=True)
            failed_marker.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            raise

        if not h5_path.exists() or not h5_path.is_dir():
            raise RuntimeError(f"H5 build finished but cache directory is missing: {h5_path}")
        ready_marker.write_text("ready\n", encoding="utf-8")

    def _wait_for_train_h5_ready(self, timeout_seconds: int = 86400) -> None:
        h5_path = self.cfg["h5_path"]
        ready_marker = self._h5_ready_marker()
        failed_marker = self._h5_failed_marker()
        start_time = time.time()
        last_status_at = 0.0
        while True:
            if failed_marker.exists():
                detail = failed_marker.read_text(encoding="utf-8").strip()
                raise RuntimeError(f"H5 build failed on rank0 for {h5_path}: {detail}")
            if ready_marker.exists() and h5_path.exists() and h5_path.is_dir():
                return
            elapsed = time.time() - start_time
            if elapsed - last_status_at >= 300:
                print(
                    f"[rank {self.global_rank}] waiting for H5 build under {h5_path} "
                    f"(elapsed {elapsed / 60:.1f} min)"
                )
                last_status_at = elapsed
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for per-basin H5 directory: {h5_path}. "
                    f"Expected markers: ready={ready_marker}, failed={failed_marker}"
                )
            time.sleep(5)

    def _compute_primary_loss(
        self,
        criterion,
        outputs: Dict,
        basin_meta: List[Dict],
        q_mean_global: "torch.Tensor",
        q_std_global: "torch.Tensor",
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        import torch

        from loss import HydroLoss, HydroMFMLoss

        loss_q = criterion.q_loss(outputs["q_pred"], basin_meta, q_mean_global=q_mean_global, q_std_global=q_std_global)
        loss_mb = criterion._balance_term(outputs, basin_meta, loss_q.device)
        if isinstance(criterion, HydroLoss):
            total = criterion.w_q * loss_q + criterion.w_balance * loss_mb
            return total, criterion._package_metrics(total, loss_q, loss_mb)
        if isinstance(criterion, HydroMFMLoss):
            loss_peak = criterion.peak_loss(outputs["q_pred"], basin_meta) if criterion.w_peak > 0.0 else torch.zeros((), device=loss_q.device)
            total = criterion.w_q * loss_q + criterion.w_balance * loss_mb + criterion.w_peak * loss_peak
            return total, criterion._package_metrics(
                total,
                loss_q,
                loss_mb,
                extra={"loss_q_mfm": float(loss_q.detach().cpu()), "loss_peak": float(loss_peak.detach().cpu())},
            )
        raise TypeError(f"Unsupported criterion type: {type(criterion)!r}")

    @staticmethod
    def _has_any_grad(model: "torch.nn.Module") -> bool:
        for param in model.parameters():
            if param.grad is not None:
                return True
        return False

    def _prepare_h5(self) -> None:
        if self.ddp_enabled:
            if self.global_rank == 0:
                self._build_train_h5_if_needed()
            else:
                self._wait_for_train_h5_ready()
        else:
            self._build_train_h5_if_needed()

    def _setup_device(self) -> None:
        import torch
        import torch.distributed as dist

        if self.ddp_enabled:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if self.local_rank >= device_count:
                    raise RuntimeError(
                        f"DDP local_rank={self.local_rank} but only {device_count} CUDA device(s) are visible. "
                        f"Check CUDA_VISIBLE_DEVICES and NPROC_PER_NODE."
                    )
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cpu")
            self.rank = dist.get_rank()
        else:
            self.device = torch.device(
                self.cfg["device"] if torch.cuda.is_available() and str(self.cfg["device"]).startswith("cuda") else "cpu"
            )
            self.rank = 0

        self.is_rank0 = self.rank == 0
        if self.is_rank0:
            print("Process temporary directory:", self.artifact_dirs["tmp"])
            print("Using device:", self.device)

    def _setup_data(self) -> None:
        from torch.utils.data import DataLoader
        from balanced_block_sampler import BalancedBlockDistributedSampler
        from dataset_global import PerBasinBlockH5Dataset, collate_block_basin_batch
        from utils import read_basin_table, resolve_input_path

        assert self.device is not None
        pin_memory = bool(self.cfg.get("pin_memory", False)) and self.device.type == "cuda"
        persistent_workers = bool(self.cfg.get("persistent_workers", False)) and int(self.cfg["num_workers"]) > 0
        self.profile_step_timing_csv = self._resolve_rank_csv_path(
            self.cfg.get("profile_step_timing_csv", "timing_step_phaseh.csv"),
            rank=self.rank,
            world_size=self.world_size,
        )

        q_file_path = resolve_input_path(self.cfg["q_file"], search_roots=[self.cfg["data_dir"]])
        q_basin_ids = read_basin_table(q_file_path)["basin_id"].astype(str).tolist()
        if self.is_rank0:
            print(f"Loading H5 dataset by q_file basin order: {q_file_path} ({len(q_basin_ids)} basins)")

        self.dataset = PerBasinBlockH5Dataset(
            h5_path=self.cfg["h5_path"],
            scalers_path=self.cfg["scalers_path"],
            seq_len=int(self.cfg["seq_len"]),
            basins=q_basin_ids,
            drop_invalid_targets=bool(self.cfg.get("drop_invalid_targets", False)),
            target_block_size=int(self.cfg.get("target_block_size", 512)),
            target_block_stride=int(self.cfg.get("target_block_stride", self.cfg.get("target_block_size", 512))),
            drop_last_block=bool(self.cfg.get("target_block_drop_last", False)),
        )
        # Multi-basin batches are supported today through basin_meta lists.
        # A future optimization can bucket same-prefix_len basins to batch the generator more aggressively
        # while still keeping routing and bookkeeping per basin.
        self.sampler = None
        if self.ddp_enabled:
            if bool(self.cfg.get("balanced_sampler", True)):
                self.sampler = BalancedBlockDistributedSampler(
                    dataset=self.dataset,
                    batch_size=int(self.cfg["basin_batch_size"]),
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=bool(self.cfg.get("target_block_shuffle", True)),
                    seed=int(self.cfg.get("balanced_sampler_seed", 0)),
                    bucket_size=int(self.cfg.get("balanced_bucket_size", 16)),
                    drop_last=bool(self.cfg.get("balanced_sampler_drop_last", True)),
                )
            else:
                from torch.utils.data.distributed import DistributedSampler

                self.sampler = DistributedSampler(self.dataset, shuffle=bool(self.cfg.get("target_block_shuffle", True)))

        self.loader = DataLoader(
            self.dataset,
            batch_size=int(self.cfg["basin_batch_size"]),
            shuffle=False if self.sampler is not None else bool(self.cfg.get("target_block_shuffle", True)),
            sampler=self.sampler,
            num_workers=int(self.cfg["num_workers"]),
            collate_fn=collate_block_basin_batch,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def _setup_model(self) -> None:
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP
        from model_hydroai_basin import HydroAIBasinModel
        from utils import save_json

        assert self.device is not None
        self.raw_model = HydroAIBasinModel(
            dims={"dyn": len(self.cfg["dyn_vars"]), "stat": len(self.cfg["stat_vars"])},
            hidden_dim=int(self.cfg["hidden_dim"]),
            dropout=float(self.cfg["dropout"]),
            precompute_inputs=bool(self.cfg.get("precompute_inputs", True)),
            precompute_time_chunk=int(self.cfg.get("precompute_time_chunk", 0)),
            precompute_gate_positions_limit=int(self.cfg.get("precompute_gate_positions_limit", 1200000)),
        ).to(self.device)
        if bool(self.cfg.get("compile_generator", False)) and hasattr(torch, "compile"):
            self.raw_model.generator = torch.compile(self.raw_model.generator)
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.raw_model.generator.parameters()), "lr": float(self.cfg["lr_gen"])},
                {"params": list(self.raw_model.routing.parameters()), "lr": float(self.cfg["lr_vel"])},
            ]
        )
        self.criterion = build_criterion(self.cfg)
        self.use_amp = bool(self.cfg["use_amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._restore_checkpoint_state()
        self.model = (
            DDP(
                self.raw_model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=bool(self.cfg.get("ddp_find_unused_parameters", True)),
            )
            if self.ddp_enabled
            else self.raw_model
        )
        if self.is_rank0:
            save_json(
                {k: str(v) if isinstance(v, Path) else v for k, v in self.cfg.items()},
                str(self.cfg["out_dir"] / "config_used.json"),
            )

    def _train_one_epoch(self, epoch: int, global_step: int) -> Tuple[float, int, int, Dict[str, int]]:
        import torch

        assert self.device is not None
        assert self.loader is not None
        assert self.model is not None
        assert self.raw_model is not None
        assert self.optimizer is not None
        assert self.criterion is not None
        assert self.scaler is not None

        current_epoch = epoch + 1
        self.model.train()
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_skip_nonfinite = 0
        epoch_skip_no_grad = 0
        epoch_skip_synced = 0
        no_grad_examples_logged = 0
        show_progress = self.is_rank0
        pbar = tqdm(self.loader, total=len(self.loader), desc=f"Epoch {current_epoch}/{self.cfg['epochs']}", disable=not show_progress)

        for step_in_epoch, batch in enumerate(pbar, start=1):
            global_step += 1
            self.optimizer.zero_grad(set_to_none=True)
            step_t0 = self._sync_time(self.device) if self.profile_step_timing else 0.0
            basin_meta = batch["basin_meta"]
            q_mean_global = batch["q_mean_global"].to(self.device, non_blocking=True)
            q_std_global = batch["q_std_global"].to(self.device, non_blocking=True)
            try:
                forward_t0 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                model_forward_t0 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(
                        x_dyn_flat=None,
                        x_stat_flat=None,
                        basin_meta=basin_meta,
                        max_lag=int(self.cfg["max_lag"]),
                        generator_chunk_size=self.generator_chunk_size,
                        clear_cache=self.clear_cache,
                        use_checkpoint=self.use_checkpoint,
                        is_basin=True,
                        retain_runoff_grad_for_loss=True,
                        retain_route_grad_for_loss=True,
                    )
                    model_forward_t1 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                    loss_data, metrics = self._compute_primary_loss(self.criterion, outputs, basin_meta, q_mean_global, q_std_global)
                    loss_route, route_metrics = self._compute_routing_regularization(self.raw_model, outputs, basin_meta, self.device, self.route_weights)
                    loss = loss_data + loss_route
                local_skip_nonfinite = self._handle_nonfinite_step(
                    epoch=current_epoch,
                    global_step=global_step,
                    step_in_epoch=step_in_epoch,
                    basin_meta=basin_meta,
                    outputs=outputs,
                    metrics=metrics,
                    route_metrics=route_metrics,
                    loss=loss,
                    loss_data=loss_data,
                    loss_route=loss_route,
                )
                local_skip_no_grad = False
                if not bool(loss.requires_grad):
                    local_skip_no_grad = True
                    self._write_nan_report(
                        epoch=current_epoch,
                        global_step=global_step,
                        step_in_epoch=step_in_epoch,
                        basin_meta=basin_meta,
                        outputs=outputs,
                        metrics=metrics,
                        route_metrics=route_metrics,
                        loss=loss,
                        loss_data=loss_data,
                        loss_route=loss_route,
                    )
                    if no_grad_examples_logged < 3:
                        basin_ids = [str(meta.get("basin_id", "")) for meta in basin_meta]
                        supervision_text = self._supervision_summary(outputs, basin_meta)
                        print(
                            f"[warn] Loss tensor has no gradient at epoch={current_epoch}, step={step_in_epoch}, "
                            f"global_step={global_step}, basins={basin_ids}. "
                            f"Supervision={supervision_text}. Skipping this batch."
                        )
                        no_grad_examples_logged += 1
                skip_step = local_skip_nonfinite or local_skip_no_grad
                skip_step = self._sync_skip_step(skip_step)
                if skip_step:
                    if local_skip_nonfinite:
                        epoch_skip_nonfinite += 1
                    if local_skip_no_grad:
                        epoch_skip_no_grad += 1
                    if not local_skip_nonfinite and not local_skip_no_grad:
                        epoch_skip_synced += 1
                    if self.device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    del outputs, loss, loss_data, loss_route, metrics, route_metrics
                    del basin_meta, q_mean_global, q_std_global
                    maybe_empty_cuda_cache(global_step, enabled=(self.device.type == "cuda"), interval=self.empty_cache_interval)
                    continue
                forward_t1 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                backward_t0 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                did_backward = False
                if bool(loss.requires_grad):
                    self.scaler.scale(loss).backward()
                    did_backward = self._has_any_grad(self.model)
                backward_t1 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                grad_clip_t0 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                if did_backward:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["grad_clip"]))
                grad_clip_t1 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                optimizer_t0 = self._sync_time(self.device) if self.profile_step_timing else 0.0
                if did_backward:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                elif self.use_amp:
                    self.scaler.update()
                optimizer_t1 = self._sync_time(self.device) if self.profile_step_timing else 0.0
            except torch.OutOfMemoryError as exc:
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                basin_ids = [str(meta.get("basin_id", "")) for meta in basin_meta]
                raise RuntimeError(
                    "CUDA OOM during training step. Try reducing --target_block_size, "
                    "--target_block_stride, or --generator_chunk_size. "
                    f"Current settings: target_block_size={self.cfg.get('target_block_size')}, "
                    f"target_block_stride={self.cfg.get('target_block_stride')}, "
                    f"generator_chunk_size={self.generator_chunk_size}, "
                    f"basin_batch_size={self.cfg.get('basin_batch_size')}, basins={basin_ids}"
                ) from exc
            step_t1 = self._sync_time(self.device) if self.profile_step_timing else 0.0

            epoch_loss += float(loss.detach().cpu())
            epoch_steps += 1

            if show_progress and (global_step == 1 or global_step % self.log_interval == 0):
                postfix = {
                    "loss": float(loss.detach().cpu()),
                    "mb": float(metrics["loss_mb"]),
                    "route": float(route_metrics.get("loss_route_total", 0.0)),
                    "v_mean": self._mean_route_stat(outputs, "velocity_mps"),
                    "lag": self._mean_route_scalar(outputs, "lag_len"),
                }
                if self.cfg["Loss"] == "NSEstd":
                    postfix["q"] = float(metrics["loss_q"])
                else:
                    postfix["mfm"] = float(1.0 - metrics["loss_q_mfm"])
                    postfix["peak"] = float(metrics.get("loss_peak", 0.0))
                pbar.set_postfix(**postfix)

            if self.profile_step_timing and self.profile_step_timing_csv is not None:
                model_forward_s = max(0.0, model_forward_t1 - model_forward_t0)
                forward_plus_loss_s = max(0.0, forward_t1 - forward_t0)
                loss_s = max(0.0, forward_plus_loss_s - model_forward_s)
                backward_s = max(0.0, backward_t1 - backward_t0)
                grad_unscale_clip_s = max(0.0, grad_clip_t1 - grad_clip_t0)
                optimizer_s = max(0.0, optimizer_t1 - optimizer_t0)
                step_total_s = max(0.0, step_t1 - step_t0)
                row = {
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "world_size": self.world_size,
                    "epoch": current_epoch,
                    "global_step": global_step,
                    "step_in_epoch": step_in_epoch,
                    "basin_ids": ";".join(str(meta.get("basin_id", "")) for meta in basin_meta),
                    "block_spans": ";".join(f"{int(meta.get('block_start', 0))}:{int(meta.get('block_end', 0))}" for meta in basin_meta),
                    "prefix_lens": ";".join(str(int(meta.get('prefix_len', 0))) for meta in basin_meta),
                    "model_forward_s": model_forward_s,
                    "forward_s": model_forward_s,
                    "loss_s": loss_s,
                    "forward_plus_loss_s": forward_plus_loss_s,
                    "backward_s": backward_s,
                    "grad_unscale_clip_s": grad_unscale_clip_s,
                    "optimizer_s": optimizer_s,
                    "step_total_s": step_total_s,
                    "backward_share": (backward_s / step_total_s) if step_total_s > 0 else 0.0,
                    "forward_share": (forward_plus_loss_s / step_total_s) if step_total_s > 0 else 0.0,
                }
                self._append_step_timing_row(self.profile_step_timing_csv, row)

            del outputs, loss, loss_data, loss_route, metrics, route_metrics
            del basin_meta, q_mean_global, q_std_global
            maybe_empty_cuda_cache(global_step, enabled=(self.device.type == "cuda"), interval=self.empty_cache_interval)

        epoch_stats = {
            "skip_nonfinite": int(epoch_skip_nonfinite),
            "skip_no_grad": int(epoch_skip_no_grad),
            "skip_synced": int(epoch_skip_synced),
        }
        return epoch_loss, epoch_steps, global_step, epoch_stats

    def _reduce_epoch_stats(self, epoch_loss: float, epoch_steps: int, epoch_stats: Dict[str, int]) -> Dict[str, float]:
        import torch
        import torch.distributed as dist

        assert self.device is not None
        loss_tensor = torch.tensor(
            [
                epoch_loss,
                epoch_steps,
                int(epoch_stats.get("skip_nonfinite", 0)),
                int(epoch_stats.get("skip_no_grad", 0)),
                int(epoch_stats.get("skip_synced", 0)),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if self.ddp_enabled:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_steps = int(loss_tensor[1].item())
        loss_mean = float("nan")
        if total_steps > 0:
            loss_mean = float(loss_tensor[0].item()) / total_steps
        return {
            "loss_mean": loss_mean,
            "step_sum": total_steps,
            "skip_nonfinite": int(loss_tensor[2].item()),
            "skip_no_grad": int(loss_tensor[3].item()),
            "skip_synced": int(loss_tensor[4].item()),
        }

    def _save_checkpoint(self, current_epoch: int, epoch_loss_mean: float, epoch_steps: int, best_loss: float, global_step: int) -> float:
        import torch

        assert self.raw_model is not None
        assert self.optimizer is not None
        if self.is_rank0:
            improved = bool(epoch_steps > 0 and np.isfinite(epoch_loss_mean) and epoch_loss_mean < best_loss)
            if improved:
                best_loss = epoch_loss_mean
            ckpt = {
                "epoch": current_epoch,
                "model": self.raw_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg,
                "epoch_loss_mean": epoch_loss_mean,
                "epoch_steps": int(epoch_steps),
                "global_step": global_step,
                "best_loss": best_loss,
            }
            if self.scaler is not None:
                ckpt["scaler"] = self.scaler.state_dict()
            torch.save(ckpt, self.cfg["ckpt_dir"] / f"ckpt_epoch{current_epoch:03d}.pth")
            if improved:
                torch.save(ckpt, self.cfg["out_dir"] / "best_model.pth")
            loss_text = f"{epoch_loss_mean:.6f}" if np.isfinite(epoch_loss_mean) else "nan"
            best_text = f"{best_loss:.6f}" if np.isfinite(best_loss) else "nan"
            print(f"[epoch {current_epoch}] loss={loss_text} best={best_text} steps={int(epoch_steps)}")
        return best_loss

    def _cleanup(self) -> None:
        import torch.distributed as dist

        if self.ddp_enabled and dist.is_initialized() and self.device is not None:
            self._dist_barrier(self.device, self.local_rank)
            dist.destroy_process_group()

    def run(self) -> None:
        self._prepare_h5()
        self._setup_device()
        self._setup_data()
        self._setup_model()

        total_epochs = int(self.cfg["epochs"])
        if self.start_epoch >= total_epochs:
            if self.is_rank0:
                print(
                    f"Checkpoint already reached epoch {self.start_epoch}, "
                    f"which is >= requested epochs {total_epochs}. Training is already complete."
                )
            self._cleanup()
            return

        best_loss = self.best_loss
        global_step = self.global_step
        for epoch in range(self.start_epoch, total_epochs):
            current_epoch = epoch + 1
            epoch_loss, epoch_steps, global_step, epoch_stats = self._train_one_epoch(epoch, global_step)
            reduced_stats = self._reduce_epoch_stats(epoch_loss, epoch_steps, epoch_stats)
            epoch_loss_mean = float(reduced_stats["loss_mean"])
            total_epoch_steps = int(reduced_stats["step_sum"])
            if self.is_rank0:
                print(
                    f"[epoch {current_epoch}] valid_steps={total_epoch_steps} "
                    f"skip_nonfinite={int(reduced_stats['skip_nonfinite'])} "
                    f"skip_no_grad={int(reduced_stats['skip_no_grad'])} "
                    f"skip_synced={int(reduced_stats['skip_synced'])}"
                )
            if total_epoch_steps <= 0:
                raise RuntimeError(
                    f"Epoch {current_epoch} completed with zero valid optimization steps. "
                    f"skip_nonfinite={int(reduced_stats['skip_nonfinite'])}, "
                    f"skip_no_grad={int(reduced_stats['skip_no_grad'])}, "
                    f"skip_synced={int(reduced_stats['skip_synced'])}. "
                    "This usually means every batch was skipped because losses became non-finite, "
                    "or because the loss tensor no longer required gradients."
                )
            best_loss = self._save_checkpoint(current_epoch, epoch_loss_mean, total_epoch_steps, best_loss, global_step)

        self._cleanup()


def train(cfg: Dict) -> None:
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    train(get_args())
