from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from common import find_checkpoint


def reshape_windows(x_dyn: np.ndarray, seq_len: int) -> np.ndarray:
    if x_dyn.ndim != 3:
        raise ValueError(f"x_dyn must have shape (P, T, D), got {x_dyn.shape}")
    _, t_count, _ = x_dyn.shape
    if t_count < seq_len:
        raise ValueError(f"time length {t_count} < seq_len {seq_len}")
    windows = np.lib.stride_tricks.sliding_window_view(x_dyn, window_shape=seq_len, axis=1)
    windows = np.transpose(windows, (0, 1, 3, 2))
    return np.ascontiguousarray(windows)


def gpu_memory_status(device) -> str:
    try:
        import torch
    except ImportError:
        return "cuda_mem=n/a"
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda_mem=n/a"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(idx) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
    return f"cuda_mem_alloc={allocated:.1f}MB reserved={reserved:.1f}MB"


def setup_inference_dir(cfg: Dict, name: str) -> Path:
    run_dir = Path(cfg["run_dir"])
    eval_dir = run_dir / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inference directory: {eval_dir}")
    cfg["inference_dir"] = eval_dir
    return eval_dir


def resolve_inference_tmp_dir(cfg: Dict, default_name: str) -> Path:
    custom = cfg.get("inference_tmp_dir")
    if custom:
        tmp_dir = Path(custom)
    else:
        tmp_dir = Path(cfg["run_dir"]) / default_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def load_model(
    ckpt_path: str | os.PathLike[str],
    dyn_dim: int,
    stat_dim: int,
    device,
):
    import torch

    from model import HydroAIBasin

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    legacy_config_key = "cfg"
    cfg = dict(ckpt.get("config") or ckpt.get(legacy_config_key) or {})
    dyn_vars = cfg.get("dynamic_vars") or cfg.get("dyn_vars")
    stat_vars = cfg.get("static_vars") or cfg.get("stat_vars")
    model_dyn_dim = len(dyn_vars) if dyn_vars else int(dyn_dim)
    model_stat_dim = len(stat_vars) if stat_vars else int(stat_dim)
    routing_method = cfg.get("routing_method")
    if routing_method not in (None, "muskingum"):
        raise ValueError(f"Unsupported routing_method in checkpoint config: {routing_method!r}")
    model = HydroAIBasin(
        dims={"dyn": model_dyn_dim, "stat": model_stat_dim},
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.4)),
        precompute_inputs=bool(cfg.get("precompute_inputs", True)),
        precompute_time_chunk=int(cfg.get("precompute_time_chunk", 0)),
    )
    model.use_checkpoint_for_inference = bool(cfg.get("use_checkpoint", False))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def build_valid_mask(
    ds: xr.Dataset,
    dyn_vars: Sequence[str],
    stat_vars: Sequence[str],
    time_name: str,
) -> np.ndarray:
    sample_dyn = ds[dyn_vars[0]]
    if sample_dyn.dims[0] != time_name:
        raise ValueError(f"{dyn_vars[0]} first dim must be '{time_name}', got {sample_dyn.dims}")

    dyn_valid = np.ones(sample_dyn.shape[1:], dtype=bool)
    for name in dyn_vars:
        arr = ds[name].values
        dyn_valid &= np.all(np.isfinite(arr), axis=0)

    stat_valid = np.ones(ds[stat_vars[0]].shape, dtype=bool)
    for name in stat_vars:
        stat_valid &= np.isfinite(ds[name].values)

    return dyn_valid & stat_valid


def load_static_features(
    ds: xr.Dataset,
    stat_vars: Sequence[str],
    valid_linear_idx: np.ndarray,
) -> np.ndarray:
    stat_list = []
    for name in stat_vars:
        stat_list.append(ds[name].values.reshape(-1)[valid_linear_idx])
    return np.stack(stat_list, axis=-1).astype(np.float32)


def iter_year_slices(
    time_values: np.ndarray,
    seq_len: int,
    infer_start_year: Optional[int] = None,
    infer_end_year: Optional[int] = None,
) -> Iterator[Tuple[int, int, int]]:
    target_time = np.asarray(time_values)[seq_len - 1:]
    years = xr.DataArray(target_time).dt.year.values.astype(int)
    unique_years = np.unique(years)
    if infer_start_year is not None and infer_end_year is not None and infer_start_year > infer_end_year:
        raise ValueError(
            f"infer_start_year must be <= infer_end_year, got {infer_start_year} > {infer_end_year}"
        )
    for year in unique_years:
        if infer_start_year is not None and year < infer_start_year:
            continue
        if infer_end_year is not None and year > infer_end_year:
            continue
        idx = np.where(years == year)[0]
        if idx.size == 0:
            continue
        target_start = int(idx[0])
        target_end = int(idx[-1]) + 1
        raw_start = target_start
        raw_end = target_end + seq_len - 1
        yield int(year), raw_start, raw_end


def load_year_dynamic_data(
    tile_nc: str | os.PathLike[str],
    dyn_vars: Sequence[str],
    time_name: str,
    valid_linear_idx: np.ndarray,
    raw_start: int,
    raw_end: int,
) -> np.ndarray:
    with xr.open_dataset(tile_nc) as ds:
        year_subset = ds[list(dyn_vars)].isel({time_name: slice(raw_start, raw_end)}).load()

    dyn_list = []
    for name in dyn_vars:
        arr = year_subset[name].values
        arr_flat = arr.reshape(arr.shape[0], -1)[:, valid_linear_idx]
        dyn_list.append(arr_flat.astype(np.float32, copy=False))

    del year_subset
    return np.stack(dyn_list, axis=-1).transpose(1, 0, 2).astype(np.float32, copy=False)


def predict_grid_batch(
    model,
    dyn_scaler,
    x_dyn_year_batch: np.ndarray,
    x_stat_batch: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    import torch

    batch_size, _, dyn_dim = x_dyn_year_batch.shape
    x_dyn_std = dyn_scaler.transform(np.asarray(x_dyn_year_batch, dtype=np.float32)).astype(np.float32, copy=False)
    windows = reshape_windows(x_dyn_std, seq_len=seq_len)
    n_windows = windows.shape[1]
    x_dyn_flat = windows.reshape(batch_size * n_windows, seq_len, dyn_dim).astype(np.float32, copy=False)
    x_stat_flat = np.repeat(np.asarray(x_stat_batch, dtype=np.float32), repeats=n_windows, axis=0).astype(np.float32, copy=False)

    try:
        with torch.no_grad():
            runoff = model(
                x_dyn_flat=torch.from_numpy(x_dyn_flat),
                x_stat_flat=torch.from_numpy(x_stat_flat),
                basin_meta=None,
                is_basin=False,
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"Inference OOM with grid_batch_size={batch_size}. Reduce --grid_batch_size and retry."
            ) from exc
        raise

    return runoff.detach().cpu().numpy().reshape(batch_size, n_windows).astype(np.float32, copy=False)


def predict_one_grid(
    model,
    dyn_scaler,
    x_dyn_year_grid: np.ndarray,
    x_stat_grid: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    pred = predict_grid_batch(
        model=model,
        dyn_scaler=dyn_scaler,
        x_dyn_year_batch=np.asarray(x_dyn_year_grid, dtype=np.float32)[None, :, :],
        x_stat_batch=np.asarray(x_stat_grid, dtype=np.float32)[None, :],
        seq_len=seq_len,
    )
    return pred[0]
