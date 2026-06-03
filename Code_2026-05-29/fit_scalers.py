from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from tqdm import tqdm

from utils import StandardScaler, load_cube_vars, read_basin_table, read_qobs_series_from_ds, save_scalers


def _nc_station_id(nc_path: str | Path) -> str:
    stem = Path(nc_path).stem
    return stem[:-15] if stem.endswith("_Local_Assemble") else stem


def _update_running_stats(
    x: np.ndarray,
    sum_: np.ndarray | None,
    sum_sq: np.ndarray | None,
    count: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x64)
    safe_x = np.where(finite, x64, 0.0)
    batch_sum = safe_x.sum(axis=0)
    batch_sum_sq = np.square(safe_x).sum(axis=0)
    batch_count = finite.sum(axis=0, dtype=np.int64)

    if sum_ is None:
        return batch_sum, batch_sum_sq, batch_count
    return sum_ + batch_sum, sum_sq + batch_sum_sq, count + batch_count


def _finalize_scaler(sum_: np.ndarray, sum_sq: np.ndarray, count: np.ndarray) -> StandardScaler:
    if np.any(count <= 0):
        raise RuntimeError("Scaler fit encountered one or more features with no finite samples.")
    mean = sum_ / count
    var = np.maximum(sum_sq / count - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-12, 1.0, std)
    return StandardScaler(mean=mean.astype(np.float64), std=std.astype(np.float64))


def fit_scalers_on_nc_dir(
        data_dir: str | Path,
        out_path: str | Path,
        q_file: str | Path,
        dyn_vars: Sequence[str],
        stat_vars: Sequence[str],
        mask_var: str = "elv",
        qobs_var: str = "discharge",
        time_name: str = "time",
) -> None:
    data_dir = Path(data_dir)
    nc_dir = data_dir / "Basins_data"
    basin_table = read_basin_table(q_file)
    station_ids = {sid for sid in basin_table["station_id"].tolist() if sid}
    basin_ids = set(basin_table["basin_id"].tolist())

    nc_files = sorted(glob.glob(str(nc_dir / "*.nc")))
    if station_ids:
        nc_files = [p for p in nc_files if _nc_station_id(p) in station_ids]
    else:
        nc_files = [p for p in nc_files if _nc_station_id(p) in basin_ids]
    if len(nc_files) == 0:
        raise FileNotFoundError(f"No matching .nc files found under {nc_dir}")

    dyn_sum: np.ndarray | None = None
    dyn_sum_sq: np.ndarray | None = None
    dyn_count: np.ndarray | None = None
    q_sum: np.ndarray | None = None
    q_sum_sq: np.ndarray | None = None
    q_count: np.ndarray | None = None
    n_q_basins = 0
    station_to_basin = {
        str(row["station_id"]).strip(): str(row["basin_id"]).strip()
        for _, row in basin_table.iterrows()
        if str(row["station_id"]).strip()
    }

    for p in tqdm(nc_files, desc="fit scalers"):
        station_id = _nc_station_id(p)
        basin = station_to_basin.get(station_id, station_id)
        with xr.open_dataset(p) as ds:
            x_dyn, _, _, _ = load_cube_vars(
                ds,
                dyn_vars=tuple(dyn_vars),
                stat_vars=tuple(stat_vars),
                mask_var=mask_var,
                read_qobs=False,
            )
            q_obs = read_qobs_series_from_ds(ds, qobs_var=qobs_var, time_name=time_name)

        _, _, d_dyn = x_dyn.shape
        dyn_flat = x_dyn.reshape(-1, d_dyn)
        dyn_sum, dyn_sum_sq, dyn_count = _update_running_stats(dyn_flat, dyn_sum, dyn_sum_sq, dyn_count)

        q_obs = q_obs[np.isfinite(q_obs)]
        if q_obs.size == 0:
            print(f"[warn] skip basin {basin}: no finite qobs available for q scaler")
            continue
        q_flat = q_obs.reshape(-1, 1)
        q_sum, q_sum_sq, q_count = _update_running_stats(q_flat, q_sum, q_sum_sq, q_count)
        n_q_basins += 1

    if dyn_sum is None or dyn_sum_sq is None or dyn_count is None:
        raise RuntimeError("No dynamic samples were collected; cannot fit dynamic scaler.")
    if n_q_basins == 0:
        raise RuntimeError("No finite qobs samples found for any basin; cannot fit q scaler.")
    if q_sum is None or q_sum_sq is None or q_count is None:
        raise RuntimeError("Finite qobs basin counter is non-zero but q scaler stats were not accumulated.")

    dyn_scaler = _finalize_scaler(dyn_sum, dyn_sum_sq, dyn_count)
    q_scaler = _finalize_scaler(q_sum, q_sum_sq, q_count)

    save_scalers(str(out_path), {"dyn": dyn_scaler, "q": q_scaler})
    print(f"Saved scalers -> {out_path}")
