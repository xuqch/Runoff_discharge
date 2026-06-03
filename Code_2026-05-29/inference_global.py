from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import xarray as xr

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from config import get_args, update_run_config
from inference_unit import (
    find_checkpoint,
    gpu_memory_status,
    load_model,
    load_static_features,
    predict_grid_batch,
    setup_inference_dir,
)


def _year_tile_paths(area_dir: Path, year: int) -> tuple[Path, Path]:
    current_nc = area_dir / f"Global_{year}_Local_Assemble.nc"
    previous_nc = area_dir / f"Global_{year - 1}_Local_Assemble.nc"
    if not current_nc.exists():
        raise FileNotFoundError(f"Current year tile file not found: {current_nc}")
    if not previous_nc.exists():
        raise FileNotFoundError(
            f"Previous year tile file not found for year={year}: {previous_nc}. "
            f"The current implementation requires Global_{year - 1}_Local_Assemble.nc "
            f"to build complete windows for Global_{year}_Local_Assemble.nc."
        )
    return current_nc, previous_nc


def _available_years(area_dir: Path) -> list[int]:
    years: list[int] = []
    for path in sorted(area_dir.glob("Global_*_Local_Assemble.nc")):
        stem = path.stem
        if not stem.startswith("Global_") or not stem.endswith("_Local_Assemble"):
            continue
        year_text = stem[len("Global_") : -len("_Local_Assemble")]
        try:
            years.append(int(year_text))
        except ValueError:
            continue
    return sorted(set(years))


def _build_context_mask(
    ds_current: xr.Dataset,
    ds_previous: xr.Dataset,
    *,
    dyn_vars: Sequence[str],
    stat_vars: Sequence[str],
    time_name: str,
    seq_len: int,
) -> tuple[np.ndarray, int]:
    sample_dyn = ds_current[dyn_vars[0]]
    if sample_dyn.dims[0] != time_name:
        raise ValueError(f"{dyn_vars[0]} first dim must be '{time_name}', got {sample_dyn.dims}")

    prev_total = int(ds_previous.sizes[time_name])
    prev_tail_count = min(max(seq_len - 1, 0), prev_total)
    if prev_tail_count < max(seq_len - 1, 0):
        raise ValueError(
            f"Previous year data is too short for seq_len={seq_len}: "
            f"required {seq_len - 1} timesteps, got {prev_tail_count}"
        )

    dyn_valid = np.ones(sample_dyn.shape[1:], dtype=bool)
    for name in dyn_vars:
        curr_arr = ds_current[name].values
        dyn_valid &= np.all(np.isfinite(curr_arr), axis=0)
        if prev_tail_count > 0:
            prev_arr = ds_previous[name].isel({time_name: slice(prev_total - prev_tail_count, prev_total)}).values
            dyn_valid &= np.all(np.isfinite(prev_arr), axis=0)

    stat_valid = np.ones(ds_current[stat_vars[0]].shape, dtype=bool)
    for name in stat_vars:
        stat_valid &= np.isfinite(ds_current[name].values)

    return dyn_valid & stat_valid, prev_tail_count


def _load_year_dynamic_data_with_context(
    *,
    current_nc: str | Path,
    previous_nc: str | Path,
    dyn_vars: Sequence[str],
    time_name: str,
    valid_linear_idx: np.ndarray,
    prev_tail_count: int,
) -> np.ndarray:
    prev_tail_arrays: list[np.ndarray] = []
    if prev_tail_count > 0:
        with xr.open_dataset(previous_nc) as ds_prev:
            prev_subset = ds_prev[list(dyn_vars)].isel({time_name: slice(-prev_tail_count, None)}).load()
        for name in dyn_vars:
            arr = prev_subset[name].values
            arr_flat = arr.reshape(arr.shape[0], -1)[:, valid_linear_idx]
            prev_tail_arrays.append(arr_flat.astype(np.float32, copy=False))
        del prev_subset

    with xr.open_dataset(current_nc) as ds_current:
        current_subset = ds_current[list(dyn_vars)].load()

    dyn_list = []
    for var_idx, name in enumerate(dyn_vars):
        curr_arr = current_subset[name].values
        curr_flat = curr_arr.reshape(curr_arr.shape[0], -1)[:, valid_linear_idx].astype(np.float32, copy=False)
        if prev_tail_count > 0:
            merged = np.concatenate([prev_tail_arrays[var_idx], curr_flat], axis=0)
        else:
            merged = curr_flat
        dyn_list.append(merged)

    del current_subset
    return np.stack(dyn_list, axis=-1).transpose(1, 0, 2).astype(np.float32, copy=False)


def infer(
    area_dir: str | Path,
    ckpt_path: str | Path,
    scalers_path: str | Path,
    out_dir: str | Path,
    dyn_vars: Sequence[str],
    stat_vars: Sequence[str],
    seq_len: Optional[int] = None,
    infer_start_year: Optional[int] = None,
    infer_end_year: Optional[int] = None,
    device: Optional[str] = None,
    grid_batch_size: int = 1,
) -> None:
    import torch

    area_dir = Path(area_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    from utils import load_scalers
    scalers = load_scalers(str(scalers_path))
    dyn_scaler = scalers["dyn"]

    print("[Prepare] Loading model")
    model, cfg = load_model(
        ckpt_path=ckpt_path,
        dyn_dim=len(dyn_vars),
        stat_dim=len(stat_vars),
        device=dev,
    )
    seq_len_used = int(seq_len or cfg.get("seq_len", 270))
    years = _available_years(area_dir)
    if infer_start_year is not None:
        years = [year for year in years if year >= int(infer_start_year)]
    if infer_end_year is not None:
        years = [year for year in years if year <= int(infer_end_year)]
    if not years:
        raise ValueError(
            f"No yearly input files found in requested range: infer_start_year={infer_start_year}, "
            f"infer_end_year={infer_end_year}"
        )

    grid_batch_size = max(1, int(grid_batch_size))
    print(f"[Prepare] Available yearly tiles: {years}")

    for year in years:
        current_nc, previous_nc = _year_tile_paths(area_dir, int(year))
        time_name = "time"
        with xr.open_dataset(current_nc) as ds_current, xr.open_dataset(previous_nc) as ds_previous:
            if not dyn_vars:
                raise ValueError("dyn_vars must not be empty.")
            if not stat_vars:
                raise ValueError("stat_vars must not be empty.")
            if time_name not in ds_current.coords and time_name not in ds_current.dims:
                raise KeyError("Dataset must contain 'time' coordinate/dimension.")

            print(f"[Prepare] Building valid mask for year={year}")
            mask_2d, prev_tail_count = _build_context_mask(
                ds_current=ds_current,
                ds_previous=ds_previous,
                dyn_vars=dyn_vars,
                stat_vars=stat_vars,
                time_name=time_name,
                seq_len=seq_len_used,
            )
            valid_flat = mask_2d.reshape(-1)
            valid_linear_idx = np.where(valid_flat)[0]
            if valid_linear_idx.size == 0:
                raise ValueError(f"No valid pixels found for year={year}.")

            print(f"[Prepare] Loading static features for year={year}")
            x_stat = load_static_features(ds=ds_current, stat_vars=stat_vars, valid_linear_idx=valid_linear_idx)
            target_time = ds_current[time_name].values
            total_time = prev_tail_count + int(ds_current.sizes[time_name])
            if total_time < seq_len_used:
                raise ValueError(
                    f"Context time length {total_time} < seq_len {seq_len_used} for year={year} "
                    f"using previous file {previous_nc.name} and current file {current_nc.name}"
                )

            lat_name = "lat" if "lat" in ds_current.coords else ("y" if "y" in ds_current.coords else None)
            lon_name = "lon" if "lon" in ds_current.coords else ("x" if "x" in ds_current.coords else None)
            if lat_name is None or lon_name is None:
                raise KeyError("Could not infer spatial coordinates from the tile dataset.")

            lat_values = ds_current[lat_name].values
            lon_values = ds_current[lon_name].values
            tile_name = current_nc.name
            ny, nx = mask_2d.shape

        n_windows = int(target_time.shape[0])
        runoff_flat = np.full((n_windows, ny * nx), np.nan, dtype=np.float32)

        print(
            f"[Prepare] Total valid grids for year={year}: {valid_linear_idx.size} {gpu_memory_status(dev)}"
        )
        print(
            f"[inference] year={year} current_file={current_nc.name} prev_file={previous_nc.name} "
            f"prev_tail={prev_tail_count} target_steps={n_windows} valid_grids={valid_linear_idx.size}"
        )
        print(f"[inference] loading dynamic data for year={year}")
        x_dyn_year = _load_year_dynamic_data_with_context(
            current_nc=current_nc,
            previous_nc=previous_nc,
            dyn_vars=dyn_vars,
            time_name="time",
            valid_linear_idx=valid_linear_idx,
            prev_tail_count=prev_tail_count,
        )
        print(
            f"[inference] year={year} loaded dynamic array shape={x_dyn_year.shape} "
            f"{gpu_memory_status(dev)}"
        )

        grid_iter = tqdm(
            range(0, valid_linear_idx.size, grid_batch_size),
            total=(valid_linear_idx.size + grid_batch_size - 1) // grid_batch_size,
            desc=f"Year {year} grid batches",
            unit="batch",
            leave=False,
        )
        for batch_start in grid_iter:
            batch_end = min(batch_start + grid_batch_size, valid_linear_idx.size)
            preds = predict_grid_batch(
                model=model,
                dyn_scaler=dyn_scaler,
                x_dyn_year_batch=x_dyn_year[batch_start:batch_end],
                x_stat_batch=x_stat[batch_start:batch_end],
                seq_len=seq_len_used,
            )
            runoff_flat[:, valid_linear_idx[batch_start:batch_end]] = preds.T

        del x_dyn_year
        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()

        runoff_map = runoff_flat.reshape(n_windows, ny, nx)
        out = xr.Dataset(
            data_vars={"runoff": (("time", lat_name, lon_name), runoff_map)},
            coords={
                "time": target_time,
                lat_name: lat_values,
                lon_name: lon_values,
            },
            attrs={
                "source_model": "HydroAIBasin generator-only per-grid yearly single-CUDA tqdm inference",
                "routing_applied": "False",
                "tile_input": tile_name,
                "context_previous_tile": previous_nc.name,
                "output_year": str(year),
                "grid_batch_size": int(grid_batch_size),
            },
        )
        encoding = {
            "runoff": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            }
        }
        out_path = out_dir / f"Area_{year}.nc"
        out.to_netcdf(out_path, encoding=encoding)
        print(f"[inference] saved yearly output: {out_path}")


def interfere(cfg: Dict) -> None:
    cfg = update_run_config(cfg)
    cfg["Area_dir"] = Path(cfg["data_dir"]) / "Area"

    ckpt_path = find_checkpoint(cfg)
    print(f"Validation: {ckpt_path}")

    cfg["inference_dir"] = setup_inference_dir(cfg, "inference_global")
    infer(
        area_dir=cfg["Area_dir"],
        ckpt_path=ckpt_path,
        scalers_path=cfg["scalers_path"],
        out_dir=cfg["inference_dir"],
        dyn_vars=cfg["dyn_vars"],
        stat_vars=cfg["stat_vars"],
        seq_len=cfg["seq_len"],
        infer_start_year=cfg.get("infer_start_year"),
        infer_end_year=cfg.get("infer_end_year"),
        device=cfg["device"],
        grid_batch_size=int(cfg.get("grid_batch_size", 1)),
    )


if __name__ == "__main__":
    interfere(get_args())
