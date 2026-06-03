from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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
        "route": f"{float(route_metrics.get('loss_route_total', route_metrics.get('route_loss', 0.0))):.3g}",
        "v_mean": "-" if v_mean_value is None else f"{v_mean_value:.3g}",
    }
    return postfix


def setup_run_dir(cfg: Dict) -> Dict:
    cfg["run_dir"].mkdir(parents=True, exist_ok=True)
    cfg["out_dir"] = cfg["run_dir"]
    cfg["ckpt_dir"].mkdir(parents=True, exist_ok=True)
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
    basins = read_basin_ids(cfg["basins_file"])
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
            drop_last=bool(cfg.get("balanced_sampler_drop_last", True)),
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


def reduce_epoch_stats(loss_sum: float, step_count: int, device: torch.device) -> tuple[float, int]:
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([loss_sum, float(step_count)], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor[0].item()), int(tensor[1].item())
    return loss_sum, step_count


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
) -> tuple[float, int]:
    model.train()
    route_w = route_weights(cfg)
    loss_sum = 0.0
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
        loss_sum += loss_value
        if is_main_process() and step_in_epoch % int(cfg.get("log_interval", 10)) == 0:
            progress.set_postfix(_progress_postfix(loss_value, metrics, route_metrics, outputs))
        maybe_empty_cuda_cache(global_step, enabled=bool(cfg.get("clear_cache", False)), interval=int(cfg.get("empty_cache_interval", 20)))
        del outputs, data_loss, route_loss, loss, metrics, route_metrics

    reduced_loss_sum, reduced_steps = reduce_epoch_stats(loss_sum, step_count, device)
    epoch_loss = reduced_loss_sum / max(reduced_steps, 1)
    return epoch_loss, global_step


def train(cfg: Dict) -> None:
    cfg = setup_run_dir(update_config(cfg))
    configure_run_artifact_dirs(cfg["run_dir"])
    seed_everything(int(cfg["seed"]))

    local_rank = int(cfg.get("local_rank", 0))
    rank, world_size, ddp_enabled = setup_distributed(local_rank)
    device = resolve_device(local_rank, requested=str(cfg.get("device", "cuda")))

    try:
        if is_main_process():
            save_json(cfg, str(cfg["run_dir"] / "config_used.json"))
            build_or_update_h5_if_needed(cfg)
        if ddp_enabled:
            dist.barrier()

        dataset, loader, sampler = build_loader(cfg, rank, world_size, ddp_enabled)
        model = build_model(cfg, device)
        optimizer = build_optimizer(cfg, model)
        # scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("use_amp", False)) and device.type == "cuda")
        scaler = torch.amp.GradScaler( "cuda",enabled=bool(cfg.get("use_amp", False)) and device.type == "cuda",)
        criterion = build_criterion(cfg)

        resume_path = resolve_resume_path(cfg)
        start_epoch = 1
        global_step = 0
        best_loss = float("inf")
        if resume_path is not None:
            ckpt = load_checkpoint(resume_path, model=model, optimizer=optimizer, scaler=scaler, map_location="cpu")
            completed_epoch = int(ckpt.get("completed_epoch", ckpt.get("epoch", 0)))
            start_epoch = completed_epoch + 1
            global_step = int(ckpt.get("global_step", 0))
            best_loss = float(ckpt.get("best_loss", float("inf")))
            if is_main_process():
                print(f"[resume] loaded {resume_path}; start_epoch={start_epoch}, global_step={global_step}, best_loss={best_loss:.6f}")

        if ddp_enabled:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)

        if is_main_process():
            print(f"Training basins/blocks: dataset_blocks={len(dataset)}, world_size={world_size}, device={device}")
            print("Training is intentionally kept simple; run validation.py or run_eval.sh separately after checkpoints are saved.")

        for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            epoch_loss, global_step = train_one_epoch(
                cfg=cfg,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                loader=loader,
                device=device,
                global_step=global_step,
            )
            if is_main_process():
                print(f"Epoch {epoch:03d}: loss={epoch_loss:.6f}")
                ckpt_path = cfg["ckpt_dir"] / f"ckpt_epoch{epoch:03d}.pth"
                save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    completed_epoch=epoch,
                    global_step=global_step,
                    best_loss=best_loss,
                    config=cfg,
                )
                shutil.copy2(ckpt_path, cfg["ckpt_dir"] / "latest.pth")
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_checkpoint(
                        cfg["run_dir"] / "best_model.pth",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        completed_epoch=epoch,
                        global_step=global_step,
                        best_loss=best_loss,
                        config=cfg,
                    )
            if ddp_enabled:
                dist.barrier()
    finally:
        cleanup_distributed()


def main() -> None:
    train(get_args())


if __name__ == "__main__":
    main()
