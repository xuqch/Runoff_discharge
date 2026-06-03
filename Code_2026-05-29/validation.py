from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from build_h5 import (
    _collect_expected_basin_jobs,
    _scan_valid_h5_files,
    _normalize_dates,
    _nc_station_id,
    _open_basin_dataset,
    _resolve_basin_target_window,
    _slice_period,
    _trim_to_common_time,
    create_h5_files,
)
from common import (
    calc_metrics,
    configure_process_tmp_dir,
    maybe_empty_cuda_cache,
    resolve_eval_checkpoint,
    resolve_eval_output_dir,
)
from config import get_args, update_run_config
from utils import (
    build_latitude_vector_from_ds,
    estimate_pet_hamon_mmday,
    load_dist_map,
    load_nc_vars,
    load_scalers,
    read_basin_table,
    save_json,
)

if TYPE_CHECKING:
    import torch


VALIDATION_AVAILABLE_COLUMNS = [
    "basin_id",
    "station_id",
    "start_date",
    "end_date",
    "intersection_start",
    "intersection_end",
    "num_valid_timesteps",
]
VALIDATION_ERROR_COLUMNS = [
    "basin_id",
    "error_type",
    "error_message",
    "available_start",
    "available_end",
    "validation_start",
    "validation_end",
]


def _normalize_eval_model(cfg: Dict) -> str:
    eval_model = cfg.get("eval_model")
    if eval_model is None:
        raise ValueError("evaluation requires --eval_model best or --eval_model last")
    value = str(eval_model).strip().lower()
    if value not in {"best", "last"}:
        raise ValueError("evaluation requires --eval_model best or --eval_model last")
    return value


def _normalize_eval_splits(cfg: Dict) -> List[str]:
    split = str(cfg.get("eval_split", "validation")).strip().lower()
    if split == "all":
        return ["train", "validation"]
    if split in {"train", "validation"}:
        return [split]
    raise ValueError("eval_split must be one of: train, validation, all")


def _split_output_dir(eval_root: Path, split: str) -> Path:
    path = eval_root / f"eval-{split}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _split_date_range(cfg: Dict, split: str) -> tuple[str, str]:
    if split == "train":
        return str(cfg["train_start_date"]), str(cfg["train_end_date"])
    if split == "validation":
        eval_start = cfg.get("eval_start_date")
        eval_end = cfg.get("eval_end_date")
        if eval_start is None or eval_end is None:
            raise ValueError("validation evaluation requires eval_start_date and eval_end_date")
        return str(eval_start), str(eval_end)
    raise ValueError(f"Unsupported split: {split}")


def _validation_cache_paths(run_dir: Path) -> Dict[str, Path]:
    repo_runs_dir = Path(__file__).resolve().parent / "runs"
    root = repo_runs_dir / "ealstm_validation"
    return {
        "root": root,
        "available_csv": root / "basins_validation_available.csv",
        "errors_csv": root / "basins_validation_errors.csv",
        "manifest": root / "manifest.csv",
        "missing_report": root / "missing_or_skipped_basins.csv",
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns + [c for c in df.columns if c not in columns]]
    df.to_csv(path, index=False)


def _merge_validation_build_errors(errors_csv: Path, missing_report: Path) -> None:
    if not missing_report.exists():
        if not errors_csv.exists():
            _write_csv(errors_csv, [], VALIDATION_ERROR_COLUMNS)
        return

    current = pd.read_csv(errors_csv, dtype={"basin_id": str}) if errors_csv.exists() else pd.DataFrame(columns=VALIDATION_ERROR_COLUMNS)
    build_errors = pd.read_csv(missing_report, dtype={"basin_id": str})
    if build_errors.empty:
        if not errors_csv.exists():
            _write_csv(errors_csv, [], VALIDATION_ERROR_COLUMNS)
        return

    mapped_rows: List[Dict[str, Any]] = []
    for _, row in build_errors.iterrows():
        mapped_rows.append(
            {
                "basin_id": str(row.get("basin_id", "")).strip(),
                "error_type": str(row.get("status", "build")).strip() or "build",
                "error_message": str(row.get("reason", "")).strip(),
                "available_start": np.nan,
                "available_end": np.nan,
                "validation_start": np.nan,
                "validation_end": np.nan,
            }
        )
    merged = pd.concat([current, pd.DataFrame(mapped_rows)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["basin_id", "error_type", "error_message"], keep="last")
    merged.to_csv(errors_csv, index=False)


def _rebuild_manifest_from_existing_h5(h5_dir: Path) -> Path:
    manifest_path = h5_dir / "manifest.csv"
    manifest_rows, invalid_rows = _scan_valid_h5_files(h5_dir)
    if invalid_rows:
        missing_report_path = h5_dir / "missing_or_skipped_basins.csv"
        invalid_df = pd.DataFrame(invalid_rows, columns=["basin_id", "station_id", "status", "reason"])
        invalid_df.to_csv(missing_report_path, index=False)
    if not manifest_rows:
        raise RuntimeError(f"No valid basin H5 files were found under {h5_dir}")
    pd.DataFrame(manifest_rows).sort_values("basin_id").to_csv(manifest_path, index=False)
    return manifest_path


def _incremental_build_h5_dir(
    cfg: Dict,
    h5_dir: Path,
    q_file: str | Path,
    target_start_date: str,
    target_end_date: str,
    *,
    rebuild: bool = False,
    label: str,
) -> Path:
    manifest_path = h5_dir / "manifest.csv"
    missing_q_file = h5_dir / "missing_basins_for_build.csv"

    h5_dir.mkdir(parents=True, exist_ok=True)
    if rebuild and h5_dir.exists():
        for path in h5_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    basin_table = read_basin_table(q_file)
    existing_h5_basins = {path.stem for path in h5_dir.glob("*.h5")}
    missing_mask = ~basin_table["basin_id"].astype(str).isin(existing_h5_basins)
    missing_basin_table = basin_table.loc[missing_mask].copy()

    if missing_basin_table.empty:
        missing_q_file.unlink(missing_ok=True)
        print(f"[eval] {label} H5 cache is complete for {h5_dir}; existing basin H5 files will be reused.")
        return h5_dir

    missing_basin_table.to_csv(missing_q_file, index=False)
    print(
        f"[eval] {label} H5 incremental build: existing={len(existing_h5_basins)}, "
        f"missing={len(missing_basin_table)}, subset_csv={missing_q_file}, "
        f"workers={int(cfg.get('h5_build_workers', 1))}"
    )
    create_h5_files(
        nc_dir=cfg["nc_dir"],
        q_file=missing_q_file,
        scalers_path=cfg["scalers_path"],
        out_file=h5_dir,
        dyn_vars=cfg["dyn_vars"],
        stat_vars=cfg["stat_vars"],
        seq_length=int(cfg["seq_len"]),
        mask_var=cfg["mask_var"],
        dist_var=cfg["dist_var"],
        time_name=cfg["time_name"],
        qobs_var=cfg["qobs_var"],
        precip_var=cfg.get("precip_var"),
        pet_var=cfg.get("pet_var"),
        pet_method=cfg.get("pet_method", "hamon"),
        target_start_date=target_start_date,
        target_end_date=target_end_date,
        fraction_var=cfg.get("fraction_var", "fraction"),
        water_year_start_month=int(cfg.get("budyko_year_start_month", 10)),
        h5_build_workers=int(cfg.get("h5_build_workers", 1)),
    )
    if not h5_dir.exists() or not h5_dir.is_dir():
        raise RuntimeError(f"{label} H5 build finished but cache directory is missing: {h5_dir}")
    return h5_dir


def _load_full_basin_meta(h5_file: Path, scalers_path: Path) -> Dict[str, Any]:
    import torch

    q_scaler = load_scalers(str(scalers_path))["q"]
    with h5py.File(h5_file, "r") as f:
        basin_id = str(f.attrs["basin_id"])
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
        "p_count": int(x_dyn_base.shape[0]),
        "n_count": int(target_idx.shape[0]),
        "block_start": 0,
        "block_end": int(target_idx.shape[0]),
        "seq_start": 0,
        "seq_end": int(x_dyn_base.shape[1]),
        "prefix_len": int(x_dyn_base.shape[1]),
        "x_dyn_base": torch.from_numpy(x_dyn_base),
        "x_stat_base": torch.from_numpy(x_stat_base),
        "target_idx": torch.from_numpy(target_idx),
        "q_true": torch.from_numpy(q_true),
        "q_valid": torch.from_numpy(q_valid),
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


def _make_basin_outputs(
    basin_id: str,
    meta: Dict[str, Any],
    outputs: Dict[str, List["torch.Tensor"]],
) -> tuple[pd.DataFrame, "xr.Dataset", Dict[str, float]]:

    q_pred = outputs["q_pred"][0].detach().cpu().numpy().reshape(-1)
    runoff = outputs["runoff"][0].detach().cpu().numpy()
    runoff_mm = outputs["runoff_mm"][0].detach().cpu().numpy()
    runoff_m3s = outputs["runoff_m3s"][0].detach().cpu().numpy()
    q_true = meta["q_true"].detach().cpu().numpy().reshape(-1)
    q_valid = meta["q_valid"].detach().cpu().numpy().reshape(-1).astype(bool)
    precip_mm = meta["precip_mm"].detach().cpu().numpy()
    area_m2 = meta["area_m2"].detach().cpu().numpy().reshape(-1)
    fraction = meta["fraction"].detach().cpu().numpy().reshape(-1)
    dates = pd.to_datetime(np.asarray(meta["target_dates"]).astype(np.int64), unit="ns")

    n = min(
        len(q_pred),
        len(q_true),
        len(q_valid),
        len(dates),
        runoff.shape[1],
        runoff_mm.shape[1],
        runoff_m3s.shape[1],
        precip_mm.shape[1],
    )
    q_pred = q_pred[:n]
    q_pred[q_pred < 0] = 0.0
    q_true = q_true[:n]
    q_valid = q_valid[:n]
    dates = dates[:n]
    runoff = runoff[:, :n]
    runoff_mm = runoff_mm[:, :n]
    runoff_m3s = runoff_m3s[:, :n]
    precip_mm = precip_mm[:, :n]

    q_obs = q_true.copy()
    q_obs[~q_valid] = np.nan
    runoff_sum = runoff.sum(axis=0)
    runoff_sum_mmday = runoff_mm.sum(axis=0)
    runoff_sum_m3s = runoff_m3s.sum(axis=0)
    precip_sum_mmday = precip_mm.sum(axis=0)
    area_scaled = area_m2 * fraction
    basin_area_sum_m2 = float(area_scaled.sum())

    basin_df = pd.DataFrame(
        {
            "basin_id": basin_id,
            "time": dates,
            "qsim": q_pred,
            "qobs": q_obs,
            "valid": q_valid.astype(np.int8),
            "runoff_sum": runoff_sum,
            "runoff_sum_mmday": runoff_sum_mmday,
            "runoff_sum_m3s": runoff_sum_m3s,
            "precip_sum_mmday": precip_sum_mmday,
        }
    )

    basin_ds = xr.Dataset(
        data_vars={
            "runoff": (("point", "time"), runoff),
            "runoff_mm": (("point", "time"), runoff_mm),
            "runoff_m3s": (("point", "time"), runoff_m3s),
            "runoff_sum": (("time",), runoff_sum),
            "runoff_sum_mmday": (("time",), runoff_sum_mmday),
            "runoff_sum_m3s": (("time",), runoff_sum_m3s),
            "precip_mm": (("point", "time"), precip_mm),
            "precip_sum_mmday": (("time",), precip_sum_mmday),
            "area_m2": (("point",), area_scaled),
            "fraction": (("point",), fraction),
            "qsim": (("time",), q_pred),
            "qobs": (("time",), q_obs),
            "q_valid": (("time",), q_valid.astype(np.int8)),
        },
        coords={"point": np.arange(runoff.shape[0]), "time": dates},
        attrs={"basin_id": basin_id, "model_area_sum_m2": basin_area_sum_m2},
    )
    metrics = calc_metrics(q_obs, q_pred, include_mfm=True)
    metrics["basin_id"] = basin_id
    metrics["n_samples"] = int(n)
    metrics["n_valid"] = int(np.isfinite(q_obs).sum())
    metrics["p_count"] = int(runoff.shape[0])
    metrics["model_area_sum_m2"] = basin_area_sum_m2
    return basin_df, basin_ds, metrics


def _prepare_validation_rows(cfg: Dict, basin_table: pd.DataFrame) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from utils import build_area_m2_vector_from_ds  # local import to keep validation import light

    validation_start, validation_end = _split_date_range(cfg, "validation")
    validation_start_ts = pd.Timestamp(validation_start)
    validation_end_ts = pd.Timestamp(validation_end)
    nc_dir = Path(cfg["nc_dir"])
    nc_files = [str(path) for path in sorted(nc_dir.glob("*.nc"))]
    expected_jobs = _collect_expected_basin_jobs(basin_table=basin_table, nc_files=nc_files)

    available_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for basin_id, station_id, nc_path in tqdm(expected_jobs, desc="prepare validation basins", dynamic_ncols=True):
        basin_row = basin_table.loc[basin_table["basin_id"] == basin_id].iloc[0].copy()
        if nc_path is None:
            error_rows.append(
                {
                    "basin_id": basin_id,
                    "error_type": "nc_missing",
                    "error_message": "basin data file not found",
                    "available_start": np.nan,
                    "available_end": np.nan,
                    "validation_start": validation_start_ts.date(),
                    "validation_end": validation_end_ts.date(),
                }
            )
            continue
        available_start = np.nan
        available_end = np.nan
        try:
            with _open_basin_dataset(Path(nc_path)) as ds:
                x_dyn_raw, _, q_obs, fraction, _ = load_nc_vars(
                    ds,
                    dyn_vars=tuple(cfg["dyn_vars"]),
                    stat_vars=tuple(cfg["stat_vars"]),
                    fraction_var=cfg.get("fraction_var", "fraction"),
                    mask_var=cfg["mask_var"],
                    qobs_var=cfg["qobs_var"],
                    time_name=cfg["time_name"],
                    read_qobs=True,
                )
                assert q_obs is not None
                q_valid = np.isfinite(q_obs)
                q_valid &= q_obs >= 0.0
                q_obs = q_obs.astype(np.float32, copy=False)
                q_obs[~q_valid] = np.nan

                dist_flat = load_dist_map(ds, dist_var=cfg["dist_var"])
                if dist_flat is None:
                    raise KeyError(f"{cfg['dist_var']} not found in basin dataset")
                valid_flat = (~np.isnan(ds[cfg["mask_var"]].values)).reshape(-1)
                _ = dist_flat[valid_flat].astype(np.float32)
                _ = build_area_m2_vector_from_ds(ds, mask_var=cfg["mask_var"]).astype(np.float32)

                dates = _normalize_dates(ds[cfg["time_name"]].values)
                available_start = pd.Timestamp(dates[0]).date()
                available_end = pd.Timestamp(dates[-1]).date()

                if cfg.get("pet_var") is not None and cfg["pet_var"] in ds:
                    pet_mm = ds[cfg["pet_var"]].values.reshape(ds[cfg["pet_var"]].shape[0], -1)[:, valid_flat].T.astype(np.float32)
                else:
                    if "temp" not in ds:
                        raise KeyError(f"pet_var={cfg.get('pet_var')!r} not found and temp is unavailable")
                    lat_deg = build_latitude_vector_from_ds(ds, mask_var=cfg["mask_var"]).astype(np.float32)
                    temp_raw = ds["temp"].values.reshape(ds["temp"].shape[0], -1)[:, valid_flat].T
                    pet_mm = estimate_pet_hamon_mmday(temp_raw, dates.values, lat_deg)
                if fraction.size == 0:
                    raise ValueError("fraction data is empty after masking")

            x_dyn_raw, q_obs, q_valid, dates, pet_mm, _ = _trim_to_common_time(
                basin_id=basin_id,
                x_dyn_raw=x_dyn_raw,
                q_obs=q_obs,
                q_valid=q_valid.astype(np.uint8),
                dates=dates,
                pet_mm=pet_mm,
            )
            basin_target_start, basin_target_end = _resolve_basin_target_window(
                basin_row=basin_row,
                global_start_date=validation_start_ts,
                global_end_date=validation_end_ts,
            )
            x_period, q_period, q_valid_period, dates_period, target_start, target_end = _slice_period(
                x_dyn_raw=x_dyn_raw,
                q_obs=q_obs,
                q_valid=q_valid.astype(np.uint8),
                dates=dates,
                seq_length=int(cfg["seq_len"]),
                target_start_date=basin_target_start,
                target_end_date=basin_target_end,
            )
            target_idx = np.arange(int(cfg["seq_len"]) - 1, len(dates_period), dtype=np.int32)
            target_valid = q_valid_period[target_idx].astype(bool)
            if target_valid.size <= 0:
                raise ValueError("no target timesteps available after validation slicing")
            num_valid = int(target_valid.sum())
            if num_valid <= 0:
                raise ValueError("no finite qobs on validation target timeline")

            row = basin_row.to_dict()
            row.update(
                {
                    "basin_id": basin_id,
                    "station_id": station_id or "",
                    "start_date": str(available_start),
                    "end_date": str(available_end),
                    "intersection_start": str(pd.Timestamp(target_start).date()),
                    "intersection_end": str(pd.Timestamp(target_end).date()),
                    "num_valid_timesteps": num_valid,
                }
            )
            available_rows.append(row)
        except Exception as exc:
            error_rows.append(
                {
                    "basin_id": basin_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "available_start": available_start,
                    "available_end": available_end,
                    "validation_start": validation_start_ts.date(),
                    "validation_end": validation_end_ts.date(),
                }
            )
    return available_rows, error_rows


def _prepare_validation_data(cfg: Dict, run_dir: Path) -> Path:
    cache_paths = _validation_cache_paths(run_dir)
    cache_root = cache_paths["root"]
    available_csv = cache_paths["available_csv"]
    errors_csv = cache_paths["errors_csv"]
    manifest_path = cache_paths["manifest"]
    rebuild = bool(cfg.get("rebuild_validation_data", False))

    cache_root.mkdir(parents=True, exist_ok=True)
    if (
        not rebuild
        and available_csv.exists()
        and any(cache_root.glob("*.h5"))
    ):
        print(f"[eval] reuse validation data cache: {cache_root}")
        print(f"[eval] validation basin csv: {available_csv}")
        if manifest_path.exists():
            print(f"[eval] validation manifest: {manifest_path}")
        if not errors_csv.exists():
            _write_csv(errors_csv, [], VALIDATION_ERROR_COLUMNS)
        return cache_root

    if rebuild and cache_root.exists():
        for path in cache_root.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    basin_table = read_basin_table(cfg["q_file"])
    available_rows, error_rows = _prepare_validation_rows(cfg, basin_table)
    _write_csv(available_csv, available_rows, VALIDATION_AVAILABLE_COLUMNS)
    _write_csv(errors_csv, error_rows, VALIDATION_ERROR_COLUMNS)
    print(f"[eval] validation available basins csv: {available_csv}")
    print(f"[eval] validation errors csv: {errors_csv}")
    print(f"[eval] validation available basins: {len(available_rows)}")
    print(f"[eval] validation skipped/error basins: {len(error_rows)}")

    if not available_rows:
        raise RuntimeError("No basins have a valid intersection with the requested validation period.")

    _incremental_build_h5_dir(
        cfg=cfg,
        h5_dir=cache_root,
        q_file=available_csv,
        target_start_date=str(cfg["eval_start_date"]),
        target_end_date=str(cfg["eval_end_date"]),
        rebuild=False,
        label="validation",
    )

    _merge_validation_build_errors(errors_csv, cache_paths["missing_report"])
    available_df = pd.read_csv(available_csv, dtype={"basin_id": str})
    if cache_paths["manifest"].exists():
        manifest_df = pd.read_csv(cache_paths["manifest"], dtype={"basin_id": str})
        h5_basins = set(manifest_df["basin_id"].astype(str))
    else:
        h5_basins = {path.stem for path in cache_root.glob("*.h5")}
    available_df = available_df[available_df["basin_id"].isin(h5_basins)]
    available_df.to_csv(available_csv, index=False)
    return cache_root


def _ensure_train_eval_h5(cfg: Dict) -> Path:
    train_h5_path = Path(cfg["h5_path"])
    if train_h5_path.exists() and train_h5_path.is_dir():
        return train_h5_path

    _incremental_build_h5_dir(
        cfg=cfg,
        h5_dir=train_h5_path,
        q_file=cfg["q_file"],
        target_start_date=str(cfg["train_start_date"]),
        target_end_date=str(cfg["train_end_date"]),
        rebuild=bool(cfg.get("rebuild_eval_h5", False)),
        label="train-eval",
    )
    return train_h5_path


def _split_h5_dir(cfg: Dict, run_dir: Path, split: str) -> Path:
    if split == "train":
        return _ensure_train_eval_h5(cfg)
    if split == "validation":
        return _prepare_validation_data(cfg, run_dir)
    raise ValueError(f"Unsupported split: {split}")


def _basin_h5_files(h5_dir: Path, basins: List[str] | None = None) -> List[Path]:
    if basins is not None:
        seen_basins = set()
        files: List[Path] = []
        for basin in basins:
            basin_id = str(basin).strip()
            if not basin_id or basin_id in seen_basins:
                continue
            seen_basins.add(basin_id)
            h5_file = h5_dir / f"{basin_id}.h5"
            if h5_file.exists():
                files.append(h5_file)
        return files

    manifest_path = h5_dir / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={"basin_id": str, "h5_file": str})
        return [h5_dir / str(row["h5_file"]) for _, row in manifest.sort_values("basin_id").iterrows() if (h5_dir / str(row["h5_file"])).exists()]
    return sorted(h5_dir.glob("*.h5"))


def _append_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate_split(
    cfg: Dict,
    split: str,
    model,
    device: "torch.device",
    h5_dir: Path,
    split_dir: Path,
) -> None:
    import torch

    if split == "train":
        ordered_basins = read_basin_table(cfg["q_file"])["basin_id"].astype(str).tolist()
    elif split == "validation":
        available_csv = _validation_cache_paths(Path(cfg["run_dir"]))["available_csv"]
        ordered_basins = pd.read_csv(available_csv, dtype={"basin_id": str})["basin_id"].astype(str).tolist() if available_csv.exists() else None
    else:
        ordered_basins = None
    basin_files = _basin_h5_files(h5_dir, ordered_basins)
    metrics_path = split_dir / "metrics.csv"
    errors_path = split_dir / "evaluation_errors.csv"
    for path in (metrics_path, errors_path):
        path.unlink(missing_ok=True)

    print(f"[eval] split={split}")
    print(f"[eval] split data path: {h5_dir}")
    print(f"[eval] split output dir: {split_dir}")
    print(f"[eval] basin count: {len(basin_files)}")
    print(f"[eval] metrics path: {metrics_path}")
    print(f"[eval] per-basin csv path: {split_dir}/<basin_id>.csv")
    print(f"[eval] per-basin nc path: {split_dir}/<basin_id>.nc")

    metrics_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(basin_files, total=len(basin_files), desc=f"Evaluate {split}", dynamic_ncols=True)
        for step, h5_file in enumerate(pbar, start=1):
            basin_id = h5_file.stem
            pbar.set_postfix(basin=basin_id, metrics=len(metrics_rows), errors=len(error_rows))
            try:
                meta = _load_full_basin_meta(h5_file, Path(cfg["scalers_path"]))
                outputs = model(
                    x_dyn_flat=None,
                    x_stat_flat=None,
                    basin_meta=[meta],
                    max_lag=int(cfg["max_lag"]),
                    generator_chunk_size=int(cfg.get("generator_chunk_size", 8192)),
                    clear_cache=bool(cfg.get("clear_cache", False)),
                    is_basin=True,
                )
                basin_df, basin_ds, metrics = _make_basin_outputs(basin_id, meta, outputs)
                basin_df.to_csv(split_dir / f"{basin_id}.csv", index=False)
                basin_ds.to_netcdf(split_dir / f"{basin_id}.nc")
                metrics_rows.append(metrics)
                maybe_empty_cuda_cache(step, enabled=(device.type == "cuda"), interval=int(cfg.get("empty_cache_interval", 20)))
            except Exception as exc:
                error_rows.append(
                    {
                        "basin_id": basin_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=["basin_id", "NSE", "KGE", "RMSE", "MAE", "MFM", "Bias", "Corr", "n_samples", "n_valid", "p_count", "model_area_sum_m2"])
    metrics_df.to_csv(metrics_path, index=False)
    error_df = pd.DataFrame(error_rows)
    if error_df.empty:
        error_df = pd.DataFrame(columns=["basin_id", "error_type", "error_message"])
    error_df.to_csv(errors_path, index=False)
    summary = {
        "split": split,
        "h5_dir": str(h5_dir),
        "n_basins_total": len(basin_files),
        "n_basins_success": int(len(metrics_df)),
        "n_basins_error": int(len(error_rows)),
        "metrics_path": str(metrics_path),
        "per_basin_csv_glob": str(split_dir / "*.csv"),
        "per_basin_nc_glob": str(split_dir / "*.nc"),
        "errors_path": str(errors_path),
    }
    save_json(summary, str(split_dir / "eval_summary.json"))
    print(f"[eval] split={split} finished, success={len(metrics_df)}, errors={len(error_rows)}")


def evaluate(cfg: Dict) -> None:
    import torch

    cfg = update_run_config(cfg)
    eval_model = _normalize_eval_model(cfg)
    eval_splits = _normalize_eval_splits(cfg)

    run_dir = Path(cfg["run_dir"])
    tmp_dir = configure_process_tmp_dir(run_dir)
    device = torch.device(cfg["device"] if torch.cuda.is_available() and str(cfg["device"]).startswith("cuda") else "cpu")

    ckpt_path = resolve_eval_checkpoint(run_dir, eval_model=eval_model, eval_ckpt=cfg.get("eval_ckpt"))
    eval_root = resolve_eval_output_dir(run_dir, eval_model)
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model selector: {eval_model}")
    print(f"[eval] checkpoint path: {ckpt_path}")
    print(f"[eval] eval split: {cfg.get('eval_split')}")
    print(f"[eval] eval root: {eval_root}")
    print(f"[eval] tmp dir: {tmp_dir}")
    print(f"[eval] rebuild validation data: {bool(cfg.get('rebuild_validation_data', False))}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = dict(ckpt.get("cfg", {}))
    train_cfg.update(cfg)
    cfg = update_run_config(train_cfg)

    from model_hydroai_basin import HydroAIBasinModel

    model = HydroAIBasinModel(
        dims={"dyn": len(cfg["dyn_vars"]), "stat": len(cfg["stat_vars"])},
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg.get("dropout", 0.4)),
        precompute_inputs=bool(cfg.get("precompute_inputs", True)),
        precompute_time_chunk=int(cfg.get("precompute_time_chunk", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    for split in eval_splits:
        split_dir = _split_output_dir(eval_root, split)
        h5_dir = _split_h5_dir(cfg, run_dir, split)
        evaluate_split(cfg, split, model, device, h5_dir, split_dir)

    save_json({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}, str(eval_root / "eval_config_used.json"))
    print(f"[eval] finished. Results saved under: {eval_root}")


if __name__ == "__main__":
    evaluate(get_args())
