from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from tqdm import tqdm

from utils import StandardScaler, load_cube_vars, read_basin_table, read_qobs_series_from_ds, save_scalers


def _resolve_nc_path(nc_dir: Path, basin_id: str) -> Path | None:
    candidates = [
        nc_dir / f"{basin_id}.nc",
        nc_dir / f"{basin_id}_Local_Assemble.nc",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(nc_dir.glob(f"{basin_id}*.nc"))
    return matches[0] if matches else None


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
    q_file: str | Path | None = None,
    dyn_vars: Sequence[str] = (),
    stat_vars: Sequence[str] = (),
    mask_var: str = "elv",
    qobs_var: str = "discharge",
    time_name: str = "time",
    basins_file: str | Path | None = None,
    nc_dir: str | Path | None = None,
) -> None:
    data_dir = Path(data_dir)
    basin_path = Path(basins_file or q_file) if (basins_file or q_file) is not None else None
    if basin_path is None:
        raise ValueError("fit_scalers_on_nc_dir requires basins_file or q_file")
    nc_root = Path(nc_dir) if nc_dir is not None else data_dir / "Basins_data"

    basin_table = read_basin_table(basin_path)
    basin_ids = [str(v).strip() for v in basin_table["basin_id"].tolist() if str(v).strip()]
    nc_files = [(basin_id, _resolve_nc_path(nc_root, basin_id)) for basin_id in basin_ids]
    nc_files = [(basin_id, path) for basin_id, path in nc_files if path is not None]
    if not nc_files:
        raise FileNotFoundError(f"No matching basin .nc files found under {nc_root}")

    dyn_sum: np.ndarray | None = None
    dyn_sum_sq: np.ndarray | None = None
    dyn_count: np.ndarray | None = None
    q_sum: np.ndarray | None = None
    q_sum_sq: np.ndarray | None = None
    q_count: np.ndarray | None = None
    n_q_basins = 0

    for basin_id, nc_path in tqdm(nc_files, desc="fit scalers"):
        with xr.open_dataset(nc_path) as ds:
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
            print(f"[warn] skip basin {basin_id}: no finite qobs available for q scaler")
            continue
        q_sum, q_sum_sq, q_count = _update_running_stats(q_obs.reshape(-1, 1), q_sum, q_sum_sq, q_count)
        n_q_basins += 1

    if dyn_sum is None or dyn_sum_sq is None or dyn_count is None:
        raise RuntimeError("No dynamic samples were collected; cannot fit dynamic scaler.")
    if n_q_basins == 0 or q_sum is None or q_sum_sq is None or q_count is None:
        raise RuntimeError("No finite qobs samples found; cannot fit q scaler.")

    save_scalers(str(out_path), {"dyn": _finalize_scaler(dyn_sum, dyn_sum_sq, dyn_count), "q": _finalize_scaler(q_sum, q_sum_sq, q_count)})
    print(f"Saved scalers -> {out_path}")
