from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from utils import (
    StandardScaler,
    build_area_m2_vector_from_ds,
    build_latitude_vector_from_ds,
    compute_water_year,
    estimate_pet_hamon_mmday,
    load_dist_map,
    load_nc_vars,
    load_scalers,
    read_basin_table,
)


@dataclass(frozen=True)
class H5BuildJob:
    basin_id: str
    station_id: str | None
    nc_path: str | None
    output_h5_path: str
    basin_row: dict[str, object]
    dyn_scaler: dict[str, object]
    q_scaler: dict[str, object]
    dyn_vars: tuple[str, ...]
    stat_vars: tuple[str, ...]
    seq_length: int
    mask_var: str
    dist_var: str
    time_name: str
    qobs_var: str
    precip_var: str
    pet_var: str | None
    pet_method: str
    target_start_date: str | None
    target_end_date: str | None
    fraction_var: str
    water_year_start_month: int
    compression: str | None


@dataclass(frozen=True)
class H5BuildResult:
    basin_id: str
    station_id: str | None
    status: str
    output_path: str | None = None
    message: str = ""
    n_time: int = 0
    n_valid_qobs: int = 0


def _nc_station_id(nc_path: str | Path) -> str:
    stem = Path(nc_path).stem
    return stem[:-15] if stem.endswith("_Local_Assemble") else stem


def _normalize_dates(values: object) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _clean_optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    ts = pd.Timestamp(text)
    if pd.isna(ts):
        return None
    return ts.normalize()


def _clean_optional_year(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _resolve_basin_target_window(
    basin_row: pd.Series | dict[str, object],
    global_start_date: str | pd.Timestamp | None,
    global_end_date: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    basin_start = _clean_optional_timestamp(basin_row.get("sdate"))
    basin_end = _clean_optional_timestamp(basin_row.get("edate"))

    if basin_start is None:
        basin_start_year = _clean_optional_year(basin_row.get("syear"))
        if basin_start_year is not None:
            basin_start = pd.Timestamp(f"{basin_start_year}-01-01")
    if basin_end is None:
        basin_end_year = _clean_optional_year(basin_row.get("eyear"))
        if basin_end_year is not None:
            basin_end = pd.Timestamp(f"{basin_end_year}-12-31")

    global_start = _clean_optional_timestamp(global_start_date)
    global_end = _clean_optional_timestamp(global_end_date)

    target_start = basin_start if global_start is None else global_start
    if basin_start is not None and global_start is not None:
        target_start = max(global_start, basin_start)

    target_end = basin_end if global_end is None else global_end
    if basin_end is not None and global_end is not None:
        target_end = min(global_end, basin_end)

    return target_start, target_end


def _slice_period(
    x_dyn_raw: np.ndarray,
    q_obs: np.ndarray,
    q_valid: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_length: int,
    target_start_date: str | pd.Timestamp | None,
    target_end_date: str | pd.Timestamp | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    full_dates = pd.DatetimeIndex(dates)
    target_start = pd.Timestamp(target_start_date) if target_start_date is not None else pd.Timestamp(full_dates[seq_length - 1])
    target_end = pd.Timestamp(target_end_date) if target_end_date is not None else pd.Timestamp(full_dates[-1])

    if target_end < target_start:
        raise ValueError(f"target_end_date {target_end} < target_start_date {target_start}")

    warmup_start = target_start - pd.DateOffset(days=seq_length - 1)
    period_mask = (full_dates >= warmup_start) & (full_dates <= target_end)
    if period_mask.sum() < seq_length:
        raise ValueError(
            f"Not enough records after warmup slicing: selected {int(period_mask.sum())}, need at least {seq_length}. "
            f"warmup_start={warmup_start}, target_end={target_end}"
        )

    x_sel = x_dyn_raw[:, period_mask, :]
    q_sel = q_obs[period_mask]
    q_valid_sel = q_valid[period_mask]
    dates_sel = full_dates[period_mask]
    if pd.Timestamp(dates_sel[seq_length - 1]) != target_start:
        raise ValueError(
            f"Sliced period is not aligned like ealstm reshape. "
            f"Expected first target date {target_start}, got {pd.Timestamp(dates_sel[seq_length - 1])}"
        )
    if pd.Timestamp(dates_sel[-1]) != target_end:
        raise ValueError(
            f"Sliced period does not end at requested target_end_date. "
            f"Expected {target_end}, got {pd.Timestamp(dates_sel[-1])}"
        )
    return x_sel, q_sel, q_valid_sel, dates_sel, target_start, target_end


def _compute_q_std_loss(target_raw: np.ndarray, target_valid: np.ndarray, fallback: float = 1.0) -> float:
    valid = target_valid.reshape(-1).astype(bool)
    values = target_raw.reshape(-1)[valid]
    if values.size < 2:
        return float(fallback)
    q_std = float(np.nanstd(values.astype(np.float64)))
    if (not np.isfinite(q_std)) or q_std <= 0.0:
        return float(fallback)
    return q_std


def _manifest_row_from_h5(h5_path: Path) -> dict[str, object]:
    with h5py.File(h5_path, "r") as f:
        basin_id = str(f.attrs["basin_id"])
        return {
            "basin_id": basin_id,
            "h5_file": h5_path.name,
            "n_samples": int(f.attrs["n_samples"]),
            "p_count": int(f.attrs["p_count"]),
            "t_count": int(f.attrs["t_count"]),
            "target_start_date": str(f.attrs["target_start_date"]),
            "target_end_date": str(f.attrs["target_end_date"]),
        }


def _open_basin_dataset(nc_path: Path) -> xr.Dataset:
    open_errors: list[str] = []
    for engine in ("h5netcdf", None):
        try:
            if engine is None:
                return xr.open_dataset(nc_path)
            return xr.open_dataset(nc_path, engine=engine)
        except Exception as exc:
            engine_name = "default" if engine is None else engine
            open_errors.append(f"{engine_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Failed to open basin nc {nc_path}. Tried engines: {' | '.join(open_errors)}")


def _collect_expected_basin_jobs(
    basin_table: pd.DataFrame,
    nc_files: Sequence[str],
) -> list[tuple[str, str | None, Path | None]]:
    nc_by_station: dict[str, Path] = {}
    nc_by_basin: dict[str, Path] = {}
    for nc_path_str in nc_files:
        nc_path = Path(nc_path_str)
        station_id = _nc_station_id(nc_path)
        nc_by_station.setdefault(station_id, nc_path)
        nc_by_basin.setdefault(station_id, nc_path)

    expected_jobs: list[tuple[str, str | None, Path | None]] = []
    seen_basins: set[str] = set()
    for _, row in basin_table.iterrows():
        basin_id = str(row["basin_id"]).strip()
        if not basin_id or basin_id in seen_basins:
            continue
        seen_basins.add(basin_id)
        station_id = str(row.get("station_id", "")).strip() or None
        nc_path = nc_by_station.get(station_id) if station_id else None
        if nc_path is None:
            nc_path = nc_by_basin.get(basin_id)
        expected_jobs.append((basin_id, station_id, nc_path))
    return expected_jobs


def _load_existing_missing_report(report_path: Path) -> dict[str, dict[str, str]]:
    if not report_path.exists():
        return {}
    try:
        df = pd.read_csv(report_path)
    except Exception:
        return {}
    if "basin_id" not in df.columns:
        return {}

    existing: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        basin_id = str(row.get("basin_id", "")).strip()
        if not basin_id:
            continue
        existing[basin_id] = {
            "station_id": str(row.get("station_id", "")).strip(),
            "status": str(row.get("status", "")).strip(),
            "reason": str(row.get("reason", "")).strip(),
        }
    return existing


def _write_single_basin_h5(
    out_path: Path,
    basin: str,
    x_period: np.ndarray,
    x_stat_std: np.ndarray,
    target_idx: np.ndarray,
    target_raw: np.ndarray,
    target_valid: np.ndarray,
    target_dates: np.ndarray,
    target_years: np.ndarray,
    q_std_array: np.ndarray,
    q_mean: float,
    q_std: float,
    q_std_loss: float,
    precip_period: np.ndarray,
    pet_period: np.ndarray,
    dist_valid: np.ndarray,
    area_m2: np.ndarray,
    fraction: np.ndarray,
    dyn_vars: Sequence[str],
    stat_vars: Sequence[str],
    seq_length: int,
    fraction_var: str,
    pet_var: Optional[str],
    pet_method: str,
    water_year_start_month: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    dates_period: pd.DatetimeIndex,
    compression: str | None,
) -> None:
    with h5py.File(out_path, "w") as f:
        f.attrs["basin_id"] = basin
        f.attrs["seq_length"] = int(seq_length)
        f.attrs["dyn_vars"] = np.array(list(dyn_vars), dtype="S64")
        f.attrs["stat_vars"] = np.array(list(stat_vars), dtype="S64")
        f.attrs["q_mean"] = float(q_mean)
        f.attrs["q_std"] = float(q_std)
        f.attrs["fraction_var"] = str(fraction_var)
        f.attrs["pet_var"] = "" if pet_var is None else str(pet_var)
        f.attrs["pet_method"] = str(pet_method)
        f.attrs["water_year_start_month"] = int(water_year_start_month)
        f.attrs["h5_layout"] = "compact_per_basin_block_v1"
        f.attrs["p_count"] = int(x_period.shape[0])
        f.attrs["t_count"] = int(x_period.shape[1])
        f.attrs["n_samples"] = int(target_raw.shape[0])
        f.attrs["period_start"] = str(pd.Timestamp(dates_period[0]).date())
        f.attrs["period_end"] = str(pd.Timestamp(dates_period[-1]).date())
        f.attrs["target_start_date"] = str(target_start.date())
        f.attrs["target_end_date"] = str(target_end.date())
        f.attrs["q_std_loss"] = float(q_std_loss)
        f.attrs["windows_precomputed"] = False

        p_count, t_count, dyn_dim = x_period.shape
        target_chunk_n = max(1, min(2048, target_raw.shape[0]))

        x_dyn_chunks = (min(p_count, 64), min(t_count, 1024), dyn_dim)
        base_chunks = (min(p_count, 64), min(t_count, 1024))
        f.create_dataset("x_dyn_base", data=x_period, chunks=x_dyn_chunks, compression=compression, dtype=np.float32)
        f.create_dataset("x_stat_base", data=x_stat_std, compression=compression, dtype=np.float32)
        f.create_dataset("target_idx", data=target_idx, chunks=(target_chunk_n,), compression=compression, dtype=np.int32)
        f.create_dataset("target_data", data=target_raw, chunks=(target_chunk_n, 1), compression=compression, dtype=np.float32)
        f.create_dataset("target_valid", data=target_valid, chunks=(target_chunk_n, 1), compression=compression, dtype=np.uint8)
        f.create_dataset("target_dates", data=target_dates, compression=compression, dtype=np.int64)
        f.create_dataset("target_years", data=target_years, compression=compression, dtype=np.int32)
        f.create_dataset("q_stds", data=q_std_array, chunks=(target_chunk_n, 1), compression=compression, dtype=np.float32)
        f.create_dataset("precip_base", data=precip_period, chunks=base_chunks, compression=compression, dtype=np.float32)
        f.create_dataset("pet_base", data=pet_period, chunks=base_chunks, compression=compression, dtype=np.float32)
        f.create_dataset("dist_m", data=dist_valid, compression=compression, dtype=np.float32)
        f.create_dataset("area_m2", data=area_m2, compression=compression, dtype=np.float32)
        f.create_dataset("fraction", data=fraction, compression=compression, dtype=np.float32)


def _scaler_to_payload(scaler: StandardScaler) -> dict[str, object]:
    return {"mean": np.asarray(scaler.mean).tolist(), "std": np.asarray(scaler.std).tolist(), "eps": float(scaler.eps)}


def _scaler_from_payload(payload: dict[str, object]) -> StandardScaler:
    return StandardScaler(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        eps=float(payload.get("eps", 1e-6)),
    )


def _required_h5_datasets() -> tuple[str, ...]:
    return (
        "x_dyn_base",
        "x_stat_base",
        "target_idx",
        "target_data",
        "target_valid",
        "target_dates",
        "target_years",
        "q_stds",
        "precip_base",
        "pet_base",
        "dist_m",
        "area_m2",
        "fraction",
    )


def _check_basin_h5_integrity(h5_path: Path) -> tuple[bool, str]:
    if not h5_path.exists():
        return False, "file_missing"
    try:
        with h5py.File(h5_path, "r") as f:
            basin_id = str(f.attrs.get("basin_id", "")).strip()
            if not basin_id:
                return False, "missing_attr:basin_id"
            for key in _required_h5_datasets():
                if key not in f:
                    return False, f"missing_dataset:{key}"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    return True, ""


def _remove_file_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _trim_to_common_time(
    basin_id: str,
    x_dyn_raw: np.ndarray,
    q_obs: np.ndarray,
    q_valid: np.ndarray,
    dates: pd.DatetimeIndex,
    pet_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, list[str]]:
    lengths = {
        "x_dyn": int(x_dyn_raw.shape[1]),
        "q_obs": int(len(q_obs)),
        "q_valid": int(len(q_valid)),
        "dates": int(len(dates)),
        "pet": int(pet_mm.shape[1]),
    }
    common_t = min(lengths.values())
    warnings: list[str] = []
    if len(set(lengths.values())) > 1:
        warnings.append(f"{basin_id}: length mismatch trimmed to common shortest length {common_t} from {lengths}")
    x_dyn_raw = x_dyn_raw[:, :common_t, :]
    q_obs = q_obs[:common_t]
    q_valid = q_valid[:common_t]
    dates = dates[:common_t]
    pet_mm = pet_mm[:, :common_t]
    return x_dyn_raw, q_obs, q_valid, dates, pet_mm, warnings


def _build_one_basin_h5(job: H5BuildJob) -> H5BuildResult:
    basin = job.basin_id
    station_id = job.station_id
    output_path = Path(job.output_h5_path)

    valid_existing, reason = _check_basin_h5_integrity(output_path)
    if valid_existing:
        return H5BuildResult(
            basin_id=basin,
            station_id=station_id,
            status="exists",
            output_path=str(output_path),
            message="",
        )
    if output_path.exists():
        _remove_file_if_exists(output_path)

    if job.nc_path is None:
        return H5BuildResult(
            basin_id=basin,
            station_id=station_id,
            status="skipped",
            output_path=None,
            message="nc_file_not_found",
        )

    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    _remove_file_if_exists(tmp_path)

    try:
        dyn_scaler = _scaler_from_payload(job.dyn_scaler)
        q_scaler = _scaler_from_payload(job.q_scaler)
        with _open_basin_dataset(Path(job.nc_path)) as ds:
            x_dyn_raw, x_stat_raw, q_obs, fraction, _ = load_nc_vars(
                ds,
                dyn_vars=job.dyn_vars,
                stat_vars=job.stat_vars,
                fraction_var=job.fraction_var,
                mask_var=job.mask_var,
                qobs_var=job.qobs_var,
                time_name=job.time_name,
                read_qobs=True,
            )
            assert q_obs is not None
            q_valid = np.isfinite(q_obs)
            q_valid &= q_obs >= 0.0
            q_obs = q_obs.astype(np.float32, copy=False)
            q_obs[~q_valid] = np.nan

            dist_flat = load_dist_map(ds, dist_var=job.dist_var)
            if dist_flat is None:
                raise KeyError(f"{job.dist_var} not found in {job.nc_path}")

            valid_flat = (~np.isnan(ds[job.mask_var].values)).reshape(-1)
            dist_valid = dist_flat[valid_flat].astype(np.float32)
            area_m2 = build_area_m2_vector_from_ds(ds, mask_var=job.mask_var).astype(np.float32)
            lat_deg = build_latitude_vector_from_ds(ds, mask_var=job.mask_var).astype(np.float32)
            dates = _normalize_dates(ds[job.time_name].values)
            if job.pet_var is not None and job.pet_var in ds:
                pet_mm = ds[job.pet_var].values.reshape(ds[job.pet_var].shape[0], -1)[:, valid_flat].T.astype(np.float32)
            else:
                if "temp" not in ds:
                    raise KeyError(
                        f"{basin}: pet_var={job.pet_var!r} not found and temp is unavailable for pet_method={job.pet_method!r}"
                    )
                if str(job.pet_method).lower() != "hamon":
                    raise ValueError(f"Unsupported pet_method={job.pet_method!r}; currently only 'hamon' is supported.")
                temp_mm = ds["temp"].values.reshape(ds["temp"].shape[0], -1)[:, valid_flat].T
                pet_mm = estimate_pet_hamon_mmday(temp_mm, dates.values, lat_deg)

        x_dyn_raw, q_obs, q_valid, dates, pet_mm, warnings = _trim_to_common_time(
            basin_id=basin,
            x_dyn_raw=x_dyn_raw,
            q_obs=q_obs,
            q_valid=q_valid.astype(np.uint8),
            dates=dates,
            pet_mm=pet_mm,
        )
        n_valid_qobs = int(np.asarray(q_valid).sum())
        if x_dyn_raw.shape[1] < job.seq_length:
            return H5BuildResult(
                basin_id=basin,
                station_id=station_id,
                status="skipped",
                output_path=None,
                message=f"time_length_lt_seq_length:{x_dyn_raw.shape[1]}<{job.seq_length}",
                n_time=int(x_dyn_raw.shape[1]),
                n_valid_qobs=n_valid_qobs,
            )

        x_dyn_std = dyn_scaler.transform(x_dyn_raw).astype(np.float32)
        precip_idx = list(job.dyn_vars).index(job.precip_var)
        precip_mm = x_dyn_raw[:, :, precip_idx].astype(np.float32)
        x_stat_std = x_stat_raw.astype(np.float32)
        basin_target_start, basin_target_end = _resolve_basin_target_window(
            basin_row=job.basin_row,
            global_start_date=job.target_start_date,
            global_end_date=job.target_end_date,
        )
        try:
            x_period, q_period, q_valid_period, dates_period, target_start, target_end = _slice_period(
                x_dyn_raw=x_dyn_std,
                q_obs=q_obs,
                q_valid=q_valid.astype(np.uint8),
                dates=dates,
                seq_length=job.seq_length,
                target_start_date=basin_target_start,
                target_end_date=basin_target_end,
            )
        except ValueError as exc:
            return H5BuildResult(
                basin_id=basin,
                station_id=station_id,
                status="skipped",
                output_path=None,
                message=str(exc),
                n_time=int(x_dyn_raw.shape[1]),
                n_valid_qobs=n_valid_qobs,
            )

        period_mask = (dates >= dates_period[0]) & (dates <= dates_period[-1])
        precip_period = precip_mm[:, period_mask]
        pet_period = pet_mm[:, period_mask].astype(np.float32)
        target_idx = np.arange(job.seq_length - 1, len(dates_period), dtype=np.int32)
        target_raw = q_period[target_idx].reshape(-1, 1).astype(np.float32)
        target_valid = q_valid_period[target_idx].reshape(-1, 1).astype(np.uint8)
        target_dates = dates_period[target_idx].values.astype("datetime64[ns]").astype(np.int64)
        target_years = compute_water_year(dates_period[target_idx].values, start_month=job.water_year_start_month)
        q_std_loss = _compute_q_std_loss(target_raw, target_valid)
        q_std_array = np.full((target_raw.shape[0], 1), q_std_loss, dtype=np.float32)

        _write_single_basin_h5(
            out_path=tmp_path,
            basin=basin,
            x_period=x_period,
            x_stat_std=x_stat_std,
            target_idx=target_idx,
            target_raw=target_raw,
            target_valid=target_valid,
            target_dates=target_dates,
            target_years=target_years,
            q_std_array=q_std_array,
            q_mean=float(np.asarray(q_scaler.mean).reshape(-1)[0]),
            q_std=float(np.asarray(q_scaler.std).reshape(-1)[0]),
            q_std_loss=float(q_std_loss),
            precip_period=precip_period,
            pet_period=pet_period,
            dist_valid=dist_valid,
            area_m2=area_m2,
            fraction=fraction.astype(np.float32, copy=False),
            dyn_vars=job.dyn_vars,
            stat_vars=job.stat_vars,
            seq_length=job.seq_length,
            fraction_var=job.fraction_var,
            pet_var=job.pet_var,
            pet_method=job.pet_method,
            water_year_start_month=job.water_year_start_month,
            target_start=target_start,
            target_end=target_end,
            dates_period=dates_period,
            compression=job.compression,
        )
        tmp_path.replace(output_path)
        message = " | ".join(warnings)
        if n_valid_qobs == 0:
            message = (message + " | " if message else "") + "qobs contains no valid finite samples in selected input period"
        return H5BuildResult(
            basin_id=basin,
            station_id=station_id,
            status="built",
            output_path=str(output_path),
            message=message,
            n_time=int(x_dyn_raw.shape[1]),
            n_valid_qobs=n_valid_qobs,
        )
    except Exception as exc:
        _remove_file_if_exists(tmp_path)
        return H5BuildResult(
            basin_id=basin,
            station_id=station_id,
            status="failed",
            output_path=None,
            message=f"{type(exc).__name__}: {exc}",
        )


def _scan_valid_h5_files(out_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for h5_path in sorted(out_dir.glob("*.h5")):
        ok, reason = _check_basin_h5_integrity(h5_path)
        basin_id = h5_path.stem
        if not ok:
            invalid_rows.append({"basin_id": basin_id, "station_id": "", "status": "failed", "reason": f"invalid_h5:{reason}"})
            _remove_file_if_exists(h5_path)
            continue
        manifest_rows.append(_manifest_row_from_h5(h5_path))
    manifest_rows.sort(key=lambda row: str(row["basin_id"]))
    return manifest_rows, invalid_rows


def create_h5_files(
    nc_dir: str | Path,
    q_file: str | Path,
    scalers_path: str | Path,
    out_file: str | Path,
    dyn_vars: Sequence[str],
    stat_vars: Sequence[str],
    seq_length: int = 270,
    mask_var: str = "elv",
    dist_var: str = "dist_map",
    time_name: str = "time",
    qobs_var: str = "discharge",
    precip_var: Optional[str] = None,
    pet_var: Optional[str] = None,
    pet_method: str = "hamon",
    target_start_date: str | None = None,
    target_end_date: str | None = None,
    fraction_var: str = "fraction",
    water_year_start_month: int = 10,
    compression: str | None = "lzf",
    h5_build_workers: int = 1,
    fail_fast: bool = False,
) -> None:
    nc_dir = Path(nc_dir)
    out_dir = Path(out_file)
    basin_table = read_basin_table(q_file)
    basin_lookup = basin_table.set_index("basin_id", drop=False)
    station_ids = {sid for sid in basin_table["station_id"].tolist() if sid}
    basin_ids = set(basin_table["basin_id"].tolist())

    scalers = load_scalers(str(scalers_path))
    dyn_scaler_payload = _scaler_to_payload(scalers["dyn"])
    q_scaler_payload = _scaler_to_payload(scalers["q"])

    nc_files = sorted(glob.glob(str(nc_dir / "*.nc")))
    if station_ids:
        nc_files = [p for p in nc_files if _nc_station_id(p) in station_ids]
    else:
        nc_files = [p for p in nc_files if _nc_station_id(p) in basin_ids]
    if not nc_files:
        raise FileNotFoundError(f"No matching .nc files found under {nc_dir}")

    dyn_vars = tuple(dyn_vars)
    stat_vars = tuple(stat_vars)
    if precip_var is None:
        precip_var = dyn_vars[0]
    if precip_var not in dyn_vars:
        raise ValueError(f"precip_var={precip_var} not found in dyn_vars={list(dyn_vars)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    missing_report_path = out_dir / "missing_or_skipped_basins.csv"
    existing_missing_report = _load_existing_missing_report(missing_report_path)

    target_basin_jobs = _collect_expected_basin_jobs(basin_table=basin_table, nc_files=nc_files)
    initial_existing_count = sum(1 for basin_id, _, _ in target_basin_jobs if (out_dir / f"{basin_id}.h5").exists())
    initial_known_missing_count = sum(1 for basin_id, _, _ in target_basin_jobs if basin_id in existing_missing_report)
    pending_count = len(target_basin_jobs) - initial_existing_count - initial_known_missing_count
    print(
        f"H5 cache summary: target basins={len(target_basin_jobs)}, "
        f"existing={initial_existing_count}, known_missing={initial_known_missing_count}, to_build={pending_count}, "
        f"workers={max(1, int(h5_build_workers))}, dir={out_dir}"
    )
    # H5 build is CPU/I/O bound. Keep the default at 1; on local SSDs try 4-8, on shared/network storage prefer 2-4 or 1.

    built_count = 0
    skipped_existing_count = 0
    skipped_invalid_count = 0
    failed_count = 0
    missing_rows: list[dict[str, object]] = []
    jobs_to_run: list[H5BuildJob] = []

    progress = tqdm(
        total=len(target_basin_jobs),
        desc="build per-basin h5",
        unit="basin",
        dynamic_ncols=True,
    )

    for basin, station_id, nc_path in target_basin_jobs:
        basin_out = out_dir / f"{basin}.h5"
        progress.set_postfix(
            basin=basin,
            existing=skipped_existing_count,
            built=built_count,
            skipped=skipped_invalid_count,
            failed=failed_count,
        )
        if basin not in basin_lookup.index:
            missing_rows.append({"basin_id": basin, "station_id": station_id or "", "status": "skipped", "reason": "basin_id_not_found_in_table"})
            skipped_invalid_count += 1
            tqdm.write(f"[skip] {basin}: basin_id not found in basin table {q_file}")
            progress.update(1)
            continue

        if basin in existing_missing_report:
            prev = existing_missing_report[basin]
            missing_rows.append(
                {
                    "basin_id": basin,
                    "station_id": station_id or prev.get("station_id", ""),
                    "status": prev.get("status", "skipped") or "skipped",
                    "reason": prev.get("reason", "listed_in_existing_missing_report") or "listed_in_existing_missing_report",
                }
            )
            skipped_invalid_count += 1
            tqdm.write(
                f"[skip] {basin}: already listed in missing_or_skipped_basins.csv "
                f"(status={prev.get('status', '')}, reason={prev.get('reason', '')})"
            )
            progress.update(1)
            continue

        valid_existing, reason = _check_basin_h5_integrity(basin_out)
        if valid_existing:
            skipped_existing_count += 1
            progress.update(1)
            continue
        if basin_out.exists():
            tqdm.write(f"[warn] {basin}: existing H5 is incomplete ({reason}); rebuilding")
            _remove_file_if_exists(basin_out)

        if nc_path is None:
            missing_rows.append({"basin_id": basin, "station_id": station_id or "", "status": "missing", "reason": "nc_file_not_found"})
            skipped_invalid_count += 1
            tqdm.write(f"[skip] {basin}: no matching nc file found from basin table")
            progress.update(1)
            continue

        basin_row = basin_lookup.loc[basin]
        jobs_to_run.append(
            H5BuildJob(
                basin_id=basin,
                station_id=station_id,
                nc_path=str(nc_path),
                output_h5_path=str(basin_out),
                basin_row=basin_row.to_dict(),
                dyn_scaler=dyn_scaler_payload,
                q_scaler=q_scaler_payload,
                dyn_vars=dyn_vars,
                stat_vars=stat_vars,
                seq_length=int(seq_length),
                mask_var=mask_var,
                dist_var=dist_var,
                time_name=time_name,
                qobs_var=qobs_var,
                precip_var=precip_var,
                pet_var=pet_var,
                pet_method=pet_method,
                target_start_date=target_start_date,
                target_end_date=target_end_date,
                fraction_var=fraction_var,
                water_year_start_month=int(water_year_start_month),
                compression=compression,
            )
        )

    if int(h5_build_workers) <= 1:
        for job in jobs_to_run:
            progress.set_postfix(
                basin=job.basin_id,
                existing=skipped_existing_count,
                built=built_count,
                skipped=skipped_invalid_count,
                failed=failed_count,
            )
            result = _build_one_basin_h5(job)
            if result.message:
                prefix = "warn" if result.status in {"built", "exists", "skipped"} else "error"
                tqdm.write(f"[{prefix}] {result.basin_id}: {result.message}")
            if result.status == "built":
                built_count += 1
            elif result.status == "exists":
                skipped_existing_count += 1
            elif result.status == "skipped":
                skipped_invalid_count += 1
                missing_rows.append(
                    {"basin_id": result.basin_id, "station_id": result.station_id or "", "status": "skipped", "reason": result.message}
                )
            else:
                failed_count += 1
                missing_rows.append(
                    {"basin_id": result.basin_id, "station_id": result.station_id or "", "status": "failed", "reason": result.message}
                )
                if fail_fast:
                    progress.update(1)
                    progress.close()
                    raise RuntimeError(f"H5 build failed for basin {result.basin_id}: {result.message}")
            progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=int(h5_build_workers)) as executor:
            future_to_job = {executor.submit(_build_one_basin_h5, job): job for job in jobs_to_run}
            try:
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = H5BuildResult(
                            basin_id=job.basin_id,
                            station_id=job.station_id,
                            status="failed",
                            message=f"worker_exception:{type(exc).__name__}:{exc}",
                        )
                    progress.set_postfix(
                        basin=result.basin_id,
                        existing=skipped_existing_count,
                        built=built_count,
                        skipped=skipped_invalid_count,
                        failed=failed_count,
                    )
                    if result.message:
                        prefix = "warn" if result.status in {"built", "exists", "skipped"} else "error"
                        tqdm.write(f"[{prefix}] {result.basin_id}: {result.message}")
                    if result.status == "built":
                        built_count += 1
                    elif result.status == "exists":
                        skipped_existing_count += 1
                    elif result.status == "skipped":
                        skipped_invalid_count += 1
                        missing_rows.append(
                            {"basin_id": result.basin_id, "station_id": result.station_id or "", "status": "skipped", "reason": result.message}
                        )
                    else:
                        failed_count += 1
                        missing_rows.append(
                            {"basin_id": result.basin_id, "station_id": result.station_id or "", "status": "failed", "reason": result.message}
                        )
                        if fail_fast:
                            for pending_future in future_to_job:
                                pending_future.cancel()
                            raise RuntimeError(f"H5 build failed for basin {result.basin_id}: {result.message}")
                    progress.update(1)
            finally:
                progress.close()
    if progress.n < progress.total:
        progress.close()

    manifest_rows, invalid_rows = _scan_valid_h5_files(out_dir)
    missing_rows.extend(invalid_rows)
    missing_df = pd.DataFrame(missing_rows, columns=["basin_id", "station_id", "status", "reason"])
    if not missing_df.empty:
        missing_df = missing_df.drop_duplicates(subset=["basin_id"], keep="last").sort_values("basin_id")
    missing_df.to_csv(missing_report_path, index=False)

    print(
        f"H5 build result: total={len(target_basin_jobs)}, existing={skipped_existing_count}, "
        f"built={built_count}, skipped={skipped_invalid_count}, failed={failed_count}"
    )

    if failed_count > 0:
        print(f"[warn] {failed_count} basins failed during H5 build; see {missing_report_path}")

    if not manifest_rows:
        raise RuntimeError(f"No basin H5 files were created under {out_dir}")

    manifest = pd.DataFrame(manifest_rows).sort_values("basin_id")
    manifest.to_csv(manifest_path, index=False)
    if not missing_df.empty:
        print(f"Saved missing/skip report -> {missing_report_path}")
    print(f"Saved per-basin H5 directory -> {out_dir}")
