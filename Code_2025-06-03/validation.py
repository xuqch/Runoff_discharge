from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

from build_h5 import _open_basin_dataset, _resolve_nc_path, create_h5_files, find_missing_basin_h5s
from common import int64ns_to_datetime, mfm_1d_numpy, resolve_eval_checkpoint, resolve_eval_output_dir
from config import get_args, update_config
from model_phaseh import HydroAIBasinPhaseH
from utils import extract_rectilinear_grid_metadata, load_scalers, read_basin_ids


def _split_sources(cfg: Dict, split: str) -> tuple[Path, Path]:
    if split == "train":
        return Path(cfg["train_basins_file"]), Path(cfg["h5_dir"])
    if split == "validation":
        return Path(cfg["eval_basins_file"]), Path(cfg.get("eval_h5_dir") or Path(cfg["run_dir"]) / "ealstm_validation")
    raise ValueError(f"Unsupported split: {split}")


def _ensure_split_h5(cfg: Dict, split: str, basins_file: Path, h5_dir: Path) -> Path:
    if split == "train":
        if not h5_dir.exists():
            raise FileNotFoundError(f"[H5] train h5_dir does not exist: {h5_dir}")
        missing = find_missing_basin_h5s(basins_file, h5_dir)
        if missing:
            report_path = h5_dir / "missing_train_basin_h5s.csv"
            pd.DataFrame(missing, columns=["basin_id", "expected_h5_path"]).to_csv(report_path, index=False)
            print(f"[H5] train h5_dir exists: {h5_dir}")
            print(f"[H5] Missing train basin H5 count: {len(missing)}")
            print(f"[H5] Full missing train list written to: {report_path}")
        return h5_dir

    if split == "validation":
        if h5_dir.exists():
            missing = find_missing_basin_h5s(basins_file, h5_dir)
            if missing:
                report_path = h5_dir / "missing_validation_basin_h5s.csv"
                pd.DataFrame(missing, columns=["basin_id", "expected_h5_path"]).to_csv(report_path, index=False)
                print(f"[H5] validation h5_dir exists: {h5_dir}")
                print(f"[H5] Missing validation basin H5 count: {len(missing)}")
                print(f"[H5] Full missing validation list written to: {report_path}")
                print("[H5] No basin H5 files were built because h5_dir already exists.")
            return h5_dir

        print(f"[H5] validation h5_dir does not exist, building all validation basin H5 files: {h5_dir}")
        h5_dir.mkdir(parents=True, exist_ok=True)

        create_h5_files(
            nc_dir=cfg["nc_dir"],
            basins_file=basins_file,
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

    raise ValueError(f"Unsupported split: {split}")


def _load_model(cfg: Dict, ckpt_path: Path, device: torch.device) -> HydroAIBasinPhaseH:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config") or ckpt.get("cfg") or {}
    model = HydroAIBasinPhaseH(
        dims={"dyn": len(cfg["dyn_vars"]), "stat": len(cfg["stat_vars"])},
        hidden_dim=int(saved_cfg.get("hidden_dim", cfg.get("hidden_dim", 128))),
        dropout=float(saved_cfg.get("dropout", cfg.get("dropout", 0.4))),
        precompute_inputs=bool(saved_cfg.get("precompute_inputs", cfg.get("precompute_inputs", False))),
        precompute_time_chunk=int(saved_cfg.get("precompute_time_chunk", cfg.get("precompute_time_chunk", 0))),
        precompute_max_positions=int(saved_cfg.get("precompute_max_positions", cfg.get("precompute_max_positions", 1000000))),
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


def _basin_h5_files(h5_dir: Path, basins_file: Path) -> tuple[list[Path], list[dict[str, str]]]:
    files: list[Path] = []
    missing: list[dict[str, str]] = []
    seen: set[str] = set()

    for basin in read_basin_ids(basins_file):
        basin_id = str(basin).strip()
        if not basin_id or basin_id in seen:
            continue
        seen.add(basin_id)
        h5_file = h5_dir / f"{basin_id}.h5"
        if h5_file.exists():
            files.append(h5_file)
        else:
            missing.append({"basin_id": basin_id, "reason": f"h5_not_found: {h5_file}"})
    return files, missing


def _attr_to_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_full_basin_meta(h5_file: Path, scalers_path: Path) -> Dict[str, Any]:
    q_scaler = load_scalers(str(scalers_path))["q"]

    with h5py.File(h5_file, "r") as f:
        basin_id = _attr_to_str(f.attrs.get("basin_id"), h5_file.stem)
        x_dyn_base = f["x_dyn_base"][:].astype(np.float32)
        x_stat_base = f["x_stat_base"][:].astype(np.float32)
        target_idx = f["target_idx"][:].astype(np.int64)
        q_true = f["target_data"][:].astype(np.float32)
        q_valid = f["target_valid"][:].astype(np.float32)
        q_std_loss = f["q_stds"][:].astype(np.float32)
        target_dates = f["target_dates"][:].astype(np.int64)
        target_years = f["target_years"][:].astype(np.int32) if "target_years" in f else None

        precip_base = f["precip_base"][:].astype(np.float32)
        pet_base = f["pet_base"][:].astype(np.float32) if "pet_base" in f else None
        dist_m = f["dist_m"][:].astype(np.float32)
        area_m2 = f["area_m2"][:].astype(np.float32)
        fraction = f["fraction"][:].astype(np.float32)

    precip_mm = precip_base[:, target_idx]
    pet_mm = pet_base[:, target_idx] if pet_base is not None else None
    meta: Dict[str, Any] = {
        "basin_id": basin_id,
        "sample_id": basin_id,
        "h5_file": str(h5_file),
        "p_count": int(x_dyn_base.shape[0]),
        "n_count": int(target_idx.shape[0]),
        "block_start": 0,
        "block_end": int(target_idx.shape[0]),
        "seq_start": 0,
        "seq_end": int(x_dyn_base.shape[1]),
        "prefix_len": int(x_dyn_base.shape[1]),
        "valid_target_count": int(np.asarray(q_valid).astype(bool).sum()),
        "x_dyn_base": torch.from_numpy(x_dyn_base),
        "x_stat_base": torch.from_numpy(x_stat_base),
        "target_idx": torch.from_numpy(target_idx.astype(np.int64)),
        "target_timeindex": torch.from_numpy(target_idx.astype(np.int64)),
        "q_true": torch.from_numpy(q_true),
        "q_valid": torch.from_numpy(q_valid.astype(np.float32)),
        "q_std_loss": torch.from_numpy(q_std_loss),
        "q_mean_global": torch.tensor(float(np.asarray(q_scaler.mean).reshape(-1)[0]), dtype=torch.float32),
        "q_std_global": torch.tensor(float(np.asarray(q_scaler.std).reshape(-1)[0]), dtype=torch.float32),
        "target_dates": target_dates,
        "precip_mm": torch.from_numpy(precip_mm),
        "dist_m": torch.from_numpy(dist_m),
        "area_m2": torch.from_numpy(area_m2),
        "fraction": torch.from_numpy(fraction),
    }
    if target_years is not None:
        meta["target_years"] = torch.from_numpy(target_years)
    if pet_mm is not None:
        meta["pet_mm"] = torch.from_numpy(pet_mm)
    return meta


def _load_spatial_grid(h5_file: Path, cfg: Dict[str, Any], basin_id: str) -> Dict[str, np.ndarray]:
    """Load the original grid layout without modifying the existing H5 cache."""
    with h5py.File(h5_file, "r") as f:
        if all(name in f for name in ("lat", "lon", "valid_mask_2d")):
            return {
                "lat": f["lat"][:].astype(np.float32),
                "lon": f["lon"][:].astype(np.float32),
                "valid_mask_2d": f["valid_mask_2d"][:].astype(bool),
            }

    nc_path = _resolve_nc_path(Path(cfg["nc_dir"]), basin_id)
    if nc_path is None:
        raise FileNotFoundError(
            f"{basin_id}: H5 has no spatial metadata and no source basin NetCDF was found under {cfg['nc_dir']}"
        )
    with _open_basin_dataset(nc_path) as ds:
        lat, lon, valid_mask_2d = extract_rectilinear_grid_metadata(ds, mask_var=str(cfg["mask_var"]))
    return {"lat": lat, "lon": lon, "valid_mask_2d": valid_mask_2d}


def _grid_field_from_points(values: np.ndarray, valid_mask_2d: np.ndarray, name: str) -> np.ndarray:
    """Map a point or point/time array back to its original grid, preserving NaNs outside the basin."""
    values = np.asarray(values)
    mask = np.asarray(valid_mask_2d, dtype=bool)
    point_count = int(mask.sum())
    if values.ndim == 1:
        if values.size != point_count:
            raise ValueError(f"{name} has {values.size} points but grid has {point_count} valid cells")
        grid = np.full(mask.shape, np.nan, dtype=np.float32)
        grid.reshape(-1)[mask.reshape(-1)] = values.astype(np.float32, copy=False)
        return grid
    if values.ndim == 2:
        if values.shape[0] != point_count:
            raise ValueError(f"{name} has {values.shape[0]} points but grid has {point_count} valid cells")
        grid = np.full((values.shape[1], *mask.shape), np.nan, dtype=np.float32)
        grid.reshape(values.shape[1], -1)[:, mask.reshape(-1)] = values.T.astype(np.float32, copy=False)
        return grid
    raise ValueError(f"{name} must be point or point/time data, got shape={values.shape}")


def _output_first(outputs: Dict[str, Any], key: str):
    if key not in outputs:
        return None
    value = outputs[key]
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value[0]


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_point_time(arr: np.ndarray, n_time: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D point/time array, got shape={arr.shape}")
    if arr.shape[1] == n_time:
        return arr
    if arr.shape[0] == n_time:
        return arr.T
    raise ValueError(f"cannot align {name} shape={arr.shape} with n_time={n_time}")


def _make_basin_outputs(
    basin_id: str,
    meta: Dict[str, Any],
    outputs: Dict[str, Any],
    spatial_grid: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, xr.Dataset, Dict[str, float]]:
    q_pred_raw = _output_first(outputs, "q_pred")
    runoff_raw = _output_first(outputs, "runoff")
    if q_pred_raw is None:
        raise ValueError("model outputs do not contain q_pred")
    if runoff_raw is None:
        raise ValueError("model outputs do not contain runoff")

    q_pred = _to_numpy(q_pred_raw).reshape(-1).astype(np.float64)
    q_pred[q_pred < 0] = 0.0
    q_true = _to_numpy(meta["q_true"]).reshape(-1).astype(np.float64)
    q_valid = _to_numpy(meta["q_valid"]).reshape(-1).astype(bool)
    n_time = min(q_pred.size, q_true.size, q_valid.size)
    q_pred = q_pred[:n_time]
    q_true = q_true[:n_time]
    q_valid = q_valid[:n_time]
    q_obs = q_true.copy()
    q_obs[~q_valid] = np.nan

    dates = int64ns_to_datetime(np.asarray(meta["target_dates"], dtype=np.int64)[:n_time])
    time_strings = [str(pd.Timestamp(d).date()) for d in dates]
    runoff = _as_point_time(_to_numpy(runoff_raw).astype(np.float32), n_time, "runoff")
    precip_mm = _as_point_time(_to_numpy(meta["precip_mm"]).astype(np.float32), n_time, "precip_mm")
    area_m2 = _to_numpy(meta["area_m2"]).reshape(-1).astype(np.float64)
    fraction = np.clip(_to_numpy(meta["fraction"]).reshape(-1).astype(np.float64), 0.0, None)
    point_count = runoff.shape[0]
    valid_mask_2d = spatial_grid["valid_mask_2d"]
    lat = spatial_grid["lat"]
    lon = spatial_grid["lon"]
    if valid_mask_2d.shape != (lat.size, lon.size):
        raise ValueError(f"spatial grid shape {valid_mask_2d.shape} != (lat, lon)=({lat.size}, {lon.size})")

    runoff_sum = np.nansum(runoff, axis=0)
    precip_sum_mmday = np.nansum(precip_mm * fraction[: precip_mm.shape[0], None], axis=0)
    basin_area_sum_m2 = float(np.nansum(area_m2[: fraction.size] * fraction))

    csv_data: Dict[str, Any] = {
        "basin_id": basin_id,
        "time": time_strings,
        "qsim": q_pred,
        "qobs": q_obs,
        "valid": q_valid.astype(np.int8),
        "runoff_sum": runoff_sum,
        "precip_sum_mmday": precip_sum_mmday,
    }
    ds_data: Dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "runoff": (("time", "lat", "lon"), _grid_field_from_points(runoff, valid_mask_2d, "runoff")),
        "runoff_sum": (("time",), runoff_sum),
        "precip_mm": (("time", "lat", "lon"), _grid_field_from_points(precip_mm, valid_mask_2d, "precip_mm")),
        "precip_sum_mmday": (("time",), precip_sum_mmday),
        "area_m2": (("lat", "lon"), _grid_field_from_points(area_m2[:point_count], valid_mask_2d, "area_m2")),
        "fraction": (("lat", "lon"), _grid_field_from_points(fraction[:point_count], valid_mask_2d, "fraction")),
        "qsim": (("time",), q_pred.astype(np.float32)),
        "qobs": (("time",), q_obs.astype(np.float32)),
        "q_valid": (("time",), q_valid.astype(np.int8)),
    }

    runoff_mm_raw = _output_first(outputs, "runoff_mm")
    if runoff_mm_raw is not None:
        runoff_mm = _as_point_time(_to_numpy(runoff_mm_raw).astype(np.float32), n_time, "runoff_mm")
        runoff_sum_mmday = np.nansum(runoff_mm, axis=0)
        csv_data["runoff_sum_mmday"] = runoff_sum_mmday
        ds_data["runoff_mm"] = (("time", "lat", "lon"), _grid_field_from_points(runoff_mm, valid_mask_2d, "runoff_mm"))
        ds_data["runoff_sum_mmday"] = (("time",), runoff_sum_mmday)

    runoff_m3s_raw = _output_first(outputs, "runoff_m3s")
    if runoff_m3s_raw is not None:
        runoff_m3s = _as_point_time(_to_numpy(runoff_m3s_raw).astype(np.float32), n_time, "runoff_m3s")
        runoff_sum_m3s = np.nansum(runoff_m3s, axis=0)
        csv_data["runoff_sum_m3s"] = runoff_sum_m3s
        ds_data["runoff_m3s"] = (("time", "lat", "lon"), _grid_field_from_points(runoff_m3s, valid_mask_2d, "runoff_m3s"))
        ds_data["runoff_sum_m3s"] = (("time",), runoff_sum_m3s)

    basin_df = pd.DataFrame(csv_data)
    mfm = mfm_1d_numpy(
        q_obs,
        q_pred,
        p=float(cfg.get("mfm_p", 1.0)),
        bins_suse=int(cfg.get("mfm_bins_suse", 10)),
        bins_phi=int(cfg.get("mfm_bins_phi", 10)),
        phase_penalty_scaling=float(cfg.get("mfm_phase_penalty_scaling", 4.0)),
        phase=bool(cfg.get("mfm_phase", True)),
    )
    ds_data["mfm"] = ((), np.float32(mfm))
    basin_ds = xr.Dataset(
        data_vars=ds_data,
        coords={"lat": lat, "lon": lon, "time": dates},
        attrs={
            "basin_id": basin_id,
            "h5_file": str(meta.get("h5_file", "")),
            "spatial_layout": "original rectilinear basin grid; outside-basin cells are NaN",
        },
    )
    basin_ds["runoff"].attrs = {
        "long_name": "Unweighted grid-cell runoff depth",
        "description": "Original model runoff for each grid cell before multiplication by fraction.",
        "units": "mm day-1",
    }
    if "runoff_mm" in basin_ds:
        basin_ds["runoff_mm"].attrs = {
            "long_name": "Fraction-weighted grid-cell runoff depth",
            "description": "Grid-cell runoff depth after multiplication by the basin fraction.",
            "units": "mm day-1",
        }
    if "runoff_m3s" in basin_ds:
        basin_ds["runoff_m3s"].attrs = {
            "long_name": "Fraction-weighted grid-cell runoff discharge",
            "description": "Grid-cell runoff converted to discharge and multiplied by the basin fraction.",
            "units": "m3 s-1",
        }
    basin_ds["precip_mm"].attrs = {"long_name": "Grid-cell precipitation depth", "units": "mm day-1"}
    basin_ds["area_m2"].attrs = {"long_name": "Grid-cell area", "units": "m2"}
    basin_ds["fraction"].attrs = {"long_name": "Grid-cell basin fraction", "units": "1"}
    basin_ds["qsim"].attrs = {"long_name": "Simulated outlet discharge", "units": "m3 s-1"}
    basin_ds["qobs"].attrs = {"long_name": "Observed outlet discharge", "units": "m3 s-1"}
    basin_ds["q_valid"].attrs = {"long_name": "Observed discharge validity flag", "flag_values": np.array([0, 1], dtype=np.int8)}
    basin_ds["runoff_sum"].attrs = {"long_name": "Sum of unweighted grid-cell runoff", "units": "mm day-1"}
    basin_ds["precip_sum_mmday"].attrs = {"long_name": "Fraction-weighted sum of grid-cell precipitation", "units": "mm day-1"}
    if "runoff_sum_mmday" in basin_ds:
        basin_ds["runoff_sum_mmday"].attrs = {"long_name": "Sum of fraction-weighted grid-cell runoff depth", "units": "mm day-1"}
    if "runoff_sum_m3s" in basin_ds:
        basin_ds["runoff_sum_m3s"].attrs = {"long_name": "Sum of fraction-weighted grid-cell runoff discharge", "units": "m3 s-1"}
    basin_ds["mfm"].attrs = {
        "long_name": "Model Fidelity Metric",
        "description": "Outlet-discharge MFM calculated from qsim and valid qobs for this basin.",
        "units": "1",
    }
    valid_count = int(np.isfinite(q_obs).sum())
    metrics = {
        "basin_id": basin_id,
        "n": valid_count,
        "nse": _nse(q_obs, q_pred),
        "kge": _kge(q_obs, q_pred),
        "mfm": float(mfm),
        "bias": float(np.nanmean(q_pred - q_obs)) if valid_count else float("nan"),
        "rmse": float(np.sqrt(np.nanmean((q_pred - q_obs) ** 2))) if valid_count else float("nan"),
        "p_count": int(point_count),
        "model_area_sum_m2": basin_area_sum_m2,
    }
    return basin_df, basin_ds, metrics


def evaluate_split(
    cfg: Dict,
    split: str,
    model: HydroAIBasinPhaseH,
    device: torch.device,
    h5_dir: Path,
    basins_file: Path,
    out_dir: Path,
) -> pd.DataFrame:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    basin_files, missing_rows = _basin_h5_files(h5_dir, basins_file)
    metrics_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, str]] = list(missing_rows)

    print(f"[evaluate {split}] basins_file={basins_file}")
    print(f"[evaluate {split}] h5_dir={h5_dir}")
    print(f"[evaluate {split}] output_dir={split_dir}")
    print(f"[evaluate {split}] basin_h5_files={len(basin_files)} missing={len(missing_rows)}")

    model.eval()
    with torch.no_grad():
        for step, h5_file in enumerate(tqdm(basin_files, desc=f"evaluate {split}", dynamic_ncols=True), start=1):
            basin_id = h5_file.stem
            try:
                meta = _load_full_basin_meta(h5_file, Path(cfg["scalers_path"]))
                outputs = model(
                    None,
                    None,
                    basin_meta=[meta],
                    max_lag=int(cfg["max_lag"]),
                    generator_chunk_size=int(cfg.get("generator_chunk_size", 8192)),
                    clear_cache=bool(cfg.get("clear_cache", False)),
                    is_basin=True,
                )
                spatial_grid = _load_spatial_grid(h5_file, cfg, basin_id)
                basin_df, basin_ds, metrics = _make_basin_outputs(basin_id, meta, outputs, spatial_grid, cfg)
                basin_df.to_csv(split_dir / f"{basin_id}.csv", index=False)
                basin_ds.to_netcdf(split_dir / f"{basin_id}.nc")
                metrics_rows.append(metrics)
            except Exception as exc:
                error_rows.append({"basin_id": basin_id, "reason": f"{type(exc).__name__}: {exc}"})

            empty_cache_interval = int(cfg.get("empty_cache_interval", 0))
            if device.type == "cuda" and empty_cache_interval > 0 and step % empty_cache_interval == 0:
                torch.cuda.empty_cache()

    metric_columns = ["basin_id", "n", "nse", "kge", "mfm", "bias", "rmse", "p_count", "model_area_sum_m2"]
    metrics = pd.DataFrame(metrics_rows, columns=metric_columns)
    metrics.to_csv(out_dir / f"{split}_metrics.csv", index=False)
    pd.DataFrame({"basin_id": [p.stem for p in basin_files]}).to_csv(out_dir / f"basins_{split}_available.csv", index=False)
    error_df = pd.DataFrame(error_rows, columns=["basin_id", "reason"])
    error_df.to_csv(out_dir / f"{split}_evaluation_errors.csv", index=False)
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
    all_errors: list[pd.DataFrame] = []
    split_sources: Dict[str, Any] = {"eval_split": str(cfg.get("eval_split", "validation"))}
    for split in splits:
        basins_file, h5_dir = _split_sources(cfg, split)
        print(f"[eval split] split={split}")
        print(f"[eval split] basins_file={basins_file}")
        print(f"[eval split] h5_dir={h5_dir}")

        h5_dir = _ensure_split_h5(cfg, split, basins_file, h5_dir)
        split_sources[split] = {"basins_file": str(basins_file), "h5_dir": str(h5_dir)}
        metrics = evaluate_split(cfg, split, model, device, h5_dir, basins_file, out_dir)
        metrics.insert(0, "split", split)
        metrics.insert(1, "basins_file", str(basins_file))
        metrics.insert(2, "h5_dir", str(h5_dir))
        all_metrics.append(metrics)
        split_error_path = out_dir / f"{split}_evaluation_errors.csv"
        if split_error_path.exists():
            errors = pd.read_csv(split_error_path)
            errors.insert(0, "split", split)
            all_errors.append(errors)
    with open(out_dir / "eval_split_sources.json", "w", encoding="utf-8") as f:
        json.dump(split_sources, f, ensure_ascii=False, indent=2)
    if all_metrics:
        summary = pd.concat(all_metrics, ignore_index=True)
        summary.to_csv(out_dir / "metrics_summary.csv", index=False)
        print(summary.groupby("split")[["nse", "kge", "mfm", "rmse"]].mean(numeric_only=True))
    if all_errors:
        pd.concat(all_errors, ignore_index=True).to_csv(out_dir / "evaluation_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["split", "basin_id", "reason"]).to_csv(out_dir / "evaluation_errors.csv", index=False)


if __name__ == "__main__":
    evaluate(get_args())
