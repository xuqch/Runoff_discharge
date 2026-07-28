"""
utils.py
--------

从 ealstm 借鉴到你项目里的“工程优点”主要落在这里：
1) 标准化 scaler 与模型权重一起固化，推演端加载同一 scaler，避免分布漂移；
2) Dataset 预处理/缓存（npz/hdf5）接口，减少反复读取与 reshape；
3) 可复现：seed、cudnn 配置、日志辅助。

注意：你原文里的 prepare_basin_data 直接用“全时空均值/方差”做归一化，会引入数据泄漏。
这里改为：scaler 在训练集上 fit（离线或训练前一次），并序列化保存。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import xarray as xr
import pandas as pd

try:
    from pysheds.grid import Grid  # type: ignore
except Exception:
    Grid = None


# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 复现优先（会略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


# --------------------------
# JSON helpers
# --------------------------
def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = make_json_serializable(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# Scalers (fit on training set, reuse for inference)
# --------------------------
@dataclass
class StandardScaler:
    mean: np.ndarray

    std: np.ndarray
    eps: float = 1e-6

    @classmethod
    def fit(cls, x: np.ndarray, axis: Tuple[int, ...]) -> "StandardScaler":
        mean = np.nanmean(x, axis=axis)
        std = np.nanstd(x, axis=axis)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def rescale(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}

    @classmethod
    def from_dict(cls, d: Dict) -> "StandardScaler":
        return cls(mean=np.array(d["mean"], dtype=float), std=np.array(d["std"], dtype=float), eps=float(d.get("eps", 1e-6)))


def save_scalers(path: str, scalers: Dict[str, StandardScaler]) -> None:
    payload = {k: v.to_dict() for k, v in scalers.items()}
    save_json(payload, path)


def load_scalers(path: str) -> Dict[str, StandardScaler]:
    raw = load_json(path)
    return {k: StandardScaler.from_dict(v) for k, v in raw.items()}


# --------------------------
# Data IO and shaping
# --------------------------
def _flatten_valid(mask_2d: np.ndarray) -> np.ndarray:
    return mask_2d.reshape(-1)


def resolve_input_path(path_like: str | os.PathLike[str], search_roots: List[str | os.PathLike[str]] | None = None) -> Path:
    path = Path(path_like)
    tried: List[Path] = []

    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Input path not found: {path}")

    candidate_bases: List[Path] = []
    if search_roots:
        candidate_bases.extend(Path(root) for root in search_roots if root is not None)

    code_dir = Path(__file__).resolve().parent
    repo_root = code_dir.parent
    candidate_bases.extend(
        [
            Path.cwd(),
            code_dir,
            repo_root,
            repo_root / "data",
        ]
    )

    seen: set[Path] = set()
    for base in candidate_bases:
        candidate = (base / path).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        tried.append(candidate)
        if candidate.exists():
            return candidate

    tried_text = "\n".join(f"  - {candidate}" for candidate in tried)
    raise FileNotFoundError(
        f"Input path not found for {path_like!r}. Tried:\n{tried_text}"
    )


def read_basin_ids(q_file: str | os.PathLike[str]) -> List[str]:
    q_path = Path(q_file)
    basins_df = pd.read_csv(q_path, dtype={"basin_id": str})
    if "basin_id" not in basins_df.columns:
        raise KeyError(
            f"{q_path} must contain a 'basin_id' column. "
            f"Available columns: {list(basins_df.columns)}"
        )

    basin_ids = (
        basins_df["basin_id"]
        .dropna()
        .astype(str)
        .map(lambda value: value.strip())
    )
    basin_ids = [value for value in basin_ids.tolist() if value]
    return basin_ids


def read_basin_table(q_file: str | os.PathLike[str]) -> pd.DataFrame:
    q_path = Path(q_file)
    basins_df = pd.read_csv(q_path, dtype={"basin_id": str})
    if "basin_id" not in basins_df.columns:
        raise KeyError(
            f"{q_path} must contain a 'basin_id' column. "
            f"Available columns: {list(basins_df.columns)}"
        )

    basins_df = basins_df.copy()
    basins_df["basin_id"] = basins_df["basin_id"].astype(str).map(lambda value: value.strip())
    basins_df = basins_df[basins_df["basin_id"] != ""]
    basins_df = basins_df.dropna(subset=["basin_id"])
    return basins_df


def read_qobs_series_from_ds(
        ds: xr.Dataset,
        qobs_var: str,
        time_name: str = "time",
) -> np.ndarray:
    if not qobs_var:
        raise ValueError("qobs_var must be provided when reading qobs from basin nc.")
    if qobs_var not in ds:
        raise KeyError(f"qobs variable {qobs_var!r} not found in dataset")

    q_da = ds[qobs_var]
    q_values = np.asarray(q_da.values, dtype=np.float32)
    if time_name in q_da.dims:
        time_axis = q_da.dims.index(time_name)
        if time_axis != 0:
            q_values = np.moveaxis(q_values, time_axis, 0)
        if q_values.ndim > 1:
            q_values = q_values.reshape(q_values.shape[0], -1)
            if q_values.shape[1] != 1:
                raise ValueError(
                    f"Expected a single qobs series in basin nc, got shape={q_da.shape} "
                    f"for variable {qobs_var!r}"
                )
            q_values = q_values[:, 0]
    else:
        q_values = q_values.reshape(-1)

    q_valid = np.isfinite(q_values)
    q_valid &= q_values >= 0.0
    q_values[~q_valid] = np.nan
    return q_values.astype(np.float32, copy=False)


def load_cube_vars(
        ds: xr.Dataset,
        dyn_vars: Tuple[str, ...],
        stat_vars: Tuple[str, ...],
        mask_var: str = "elv",
        qobs_var: Optional[str] = None,
        time_name: str = "time",
        read_qobs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    返回：
      x_dyn_raw:  (P, T, D)
      x_stat_raw: (P, S)
      valid_mask_2d: (Y, X) bool
    """
    mask_2d = ~np.isnan(ds[mask_var].values)
    valid_flat = _flatten_valid(mask_2d)

    # dynamic: (T, Y, X) -> (P, T)
    dyn_list = []
    for v in dyn_vars:
        arr = ds[v].values  # (T, Y, X)
        arr2 = arr.reshape(arr.shape[0], -1)[:, valid_flat].T  # # (P, T)
        dyn_list.append(arr2)
    x_dyn = np.stack(dyn_list, axis=-1)  # (P, T, D)

    # static: (Y, X) -> (P,)
    stat_list = []
    for v in stat_vars:
        arr = ds[v].values.reshape(-1)[valid_flat]  # [valid_flat]  # (P,)
        stat_list.append(arr)
    x_stat = np.stack(stat_list, axis=-1)  # (P, S)
    qobs = None
    if read_qobs:
        if not qobs_var:
            raise ValueError("qobs_var must be provided when read_qobs=True.")
        qobs = read_qobs_series_from_ds(ds, qobs_var=qobs_var, time_name=time_name)
    return x_dyn, x_stat, qobs, mask_2d


def load_nc_vars(
        ds: xr.Dataset,
        dyn_vars: Tuple[str, ...],
        stat_vars: Tuple[str, ...],
        fraction_var: str = "fraction",
        mask_var: str = "elv",
        qobs_var: Optional[str] = None,
        time_name: str = "time",
        read_qobs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    返回：
      x_dyn_raw:  (P, T, D)
      x_stat_raw: (P, S)
      valid_mask_2d: (Y, X) bool
    """
    mask_2d = ~np.isnan(ds[mask_var].values)
    valid_flat = _flatten_valid(mask_2d)

    # dynamic: (T, Y, X) -> (P, T)
    dyn_list = []
    for v in dyn_vars:
        arr = ds[v].values  # (T, Y, X)
        arr2 = arr.reshape(arr.shape[0], -1)[:, valid_flat].T  # # (P, T)
        dyn_list.append(arr2)
    x_dyn = np.stack(dyn_list, axis=-1)  # (P, T, D)

    # static: (Y, X) -> (P,)
    stat_list = []
    for v in stat_vars:
        arr = ds[v].values.reshape(-1)[valid_flat]  # [valid_flat]  # (P,)
        stat_list.append(arr)
    x_stat = np.stack(stat_list, axis=-1)  # (P, S)

    fraction = ds[fraction_var].values
    fraction = fraction.reshape(-1)[valid_flat]

    qobs = None
    if read_qobs:
        if not qobs_var:
            raise ValueError("qobs_var must be provided when read_qobs=True.")
        qobs = read_qobs_series_from_ds(ds, qobs_var=qobs_var, time_name=time_name)
    return x_dyn, x_stat, qobs, fraction, mask_2d


def load_dist_map(ds: xr.Dataset, dist_var: str = "dist_to_outlet_m") -> Optional[np.ndarray]:
    if dist_var in ds:
        return ds[dist_var].values.reshape(-1)
    return None


def compute_dist_map_pysheds(
        ds: xr.Dataset,
        flowdir_var: str = "flow_dir",
        mask_var: str = "elev",
        dirmap: Tuple[int, ...] = (64, 128, 1, 2, 4, 8, 16, 32),
) -> np.ndarray:
    # Legacy offline helper; unused by the current training / validation pipeline.
    """
    不推荐在训练中频繁调用：非常慢。
    建议离线一次性计算 dist_to_outlet_m 并存入每个 basin cube / caravan nc。
    """
    if Grid is None:
        raise RuntimeError("pysheds 未安装，无法计算 dist_map。请 `pip install pysheds` 或离线预计算 dist_to_outlet_m。")

    flowdir = ds[flowdir_var].values
    mask_2d = ~np.isnan(ds[mask_var].values)

    grid = Grid()
    # 简化：无 affine 时按 index 计算距离（单位将变成像元步长，需要你换算为米）
    grid.add_gridded_data(flowdir, data_name="dir")

    acc = grid.accumulation(data="dir")
    y_idx, x_idx = np.unravel_index(acc.argmax(), acc.shape)
    dist = grid.flow_distance(data="dir", x=x_idx, y=y_idx, xytype="index", dirmap=dirmap)

    dist = np.where(mask_2d, dist, np.nan)
    return dist.reshape(-1)


# --------------------------
# Cache helpers (ealstm 的 HDF5 打包思想，这里提供轻量 NPZ 版本)
# --------------------------
def save_npz(path: str, payload: Dict[str, np.ndarray]) -> None:
    # Legacy cache helper; unused by the current H5-based pipeline.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    # Legacy cache helper; unused by the current H5-based pipeline.
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# --------------------------
# Torch helpers
# --------------------------
def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


def _infer_lat_lon_names(ds, lat_name=None, lon_name=None):
    """尽量从 ds 中推断 lat/lon 坐标名。支持 lat/lon 或 y/x（GEE 常见）。"""
    if lat_name is not None and lon_name is not None:
        return lat_name, lon_name

    cand_lat = [lat_name, "lat", "latitude", "y"]
    cand_lon = [lon_name, "lon", "longitude", "x"]

    lat = next((n for n in cand_lat if n and n in ds.coords), None)
    lon = next((n for n in cand_lon if n and n in ds.coords), None)

    # 有些数据把 lat/lon 放在 data_vars 里（少见），也兼容一下
    if lat is None:
        lat = next((n for n in cand_lat if n and n in ds.variables), None)
    if lon is None:
        lon = next((n for n in cand_lon if n and n in ds.variables), None)

    if lat is None or lon is None:
        raise KeyError(f"Cannot infer lat/lon names from dataset. Found coords={list(ds.coords)}")
    return lat, lon


def _infer_deg_resolution(coord_like, fallback: float | None = None) -> float:
    """从 1D 坐标或其属性推断分辨率（度）。"""
    coord_values = np.asarray(getattr(coord_like, "values", coord_like), dtype=np.float64)
    diffs = np.diff(coord_values)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        attrs = getattr(coord_like, "attrs", {}) or {}
        for key in ("resolution", "res", "step", "cellsize", "delta"):
            value = attrs.get(key)
            if value is None:
                continue
            try:
                return float(abs(value))
            except (TypeError, ValueError):
                continue
        if fallback is not None:
            return float(fallback)
        raise ValueError("Cannot infer resolution from coordinate (diffs empty).")
    return float(np.median(np.abs(diffs)))


def area_m2_latlon_grid(lat_center_deg: np.ndarray, dlat_deg: float, dlon_deg: float,
                        radius_m: float = 6371007.2) -> np.ndarray:
    """
    给定网格中心纬度（任意形状）+ 分辨率（度），计算格点面积（m²）。
    """
    lat_center_deg = np.asarray(lat_center_deg, dtype=np.float64)

    lat1_deg = np.clip(lat_center_deg - dlat_deg / 2.0, -90.0, 90.0)
    lat2_deg = np.clip(lat_center_deg + dlat_deg / 2.0, -90.0, 90.0)

    dlon = np.deg2rad(dlon_deg)
    lat1 = np.deg2rad(lat1_deg)
    lat2 = np.deg2rad(lat2_deg)

    area = (radius_m ** 2) * dlon * (np.sin(lat2) - np.sin(lat1))
    return area.astype(np.float64)


def build_area_m2_vector_from_ds(ds, mask_var: str = "elv",
                                 lat_name: str = None, lon_name: str = None,
                                 dlat_deg: float = None, dlon_deg: float = None) -> np.ndarray:
    """
    从 NetCDF 的 lat/lon 坐标 + mask_var 生成每个有效像元的 area_m2 向量（P,）。
    - mask_var: 用来定义有效像元的变量（与你 load_cube_vars 使用同一个）
    - 返回 shape=(P,), 与 x_dyn_raw 的 P 对齐（假设 P 是按 mask_flat 选出来的像元数）
    """
    latn, lonn = _infer_lat_lon_names(ds, lat_name=lat_name, lon_name=lon_name)

    # 1) 取 mask_flat（与训练代码一致：非 NaN 为有效像元）
    mask_flat = ~np.isnan(ds[mask_var].values).reshape(-1)
    # 2) 推断分辨率（如果没手动给 dlat/dlon）
    lat_da = ds[latn]
    lon_da = ds[lonn]
    lat_coord = lat_da.values
    lon_coord = lon_da.values

    # 情况A：lat/lon 是 1D 坐标（最常见）
    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        if dlat_deg is None:
            dlat_deg = _infer_deg_resolution(lat_da, fallback=0.1)
        if dlon_deg is None:
            dlon_deg = _infer_deg_resolution(lon_da, fallback=0.1)

        # 生成 2D 纬度中心（每列同一纬度）
        lat2d = lat_coord[:, None]  # (ny, 1) -> broadcast to (ny, nx)
        area2d = area_m2_latlon_grid(lat2d, dlat_deg=dlat_deg, dlon_deg=dlon_deg)
        area2d = np.broadcast_to(area2d, (lat_coord.size, lon_coord.size))

    # 情况B：lat 是 2D（如 curvilinear），lon 也可能 2D（较少）
    elif lat_coord.ndim == 2:
        if dlat_deg is None or dlon_deg is None:
            raise ValueError("Curvilinear grid detected (2D lat/lon). Please provide dlat_deg & dlon_deg explicitly.")
        area2d = area_m2_latlon_grid(lat_coord, dlat_deg=dlat_deg, dlon_deg=dlon_deg)

    else:
        raise ValueError(f"Unsupported lat/lon shapes: lat.ndim={lat_coord.ndim}, lon.ndim={lon_coord.ndim}")

    # 3) flatten 并按 mask_flat 取有效像元，得到 (P,)
    # print('area: ', area2d.shape, mask_flat)
    area_vec = area2d.reshape(-1)[mask_flat].astype(np.float32)
    return area_vec


def build_latitude_vector_from_ds(
        ds,
        mask_var: str = "elv",
        lat_name: str = None,
        lon_name: str = None,
) -> np.ndarray:
    """Build per-pixel latitude vector aligned with valid pixels."""
    latn, lonn = _infer_lat_lon_names(ds, lat_name=lat_name, lon_name=lon_name)
    mask_flat = ~np.isnan(ds[mask_var].values).reshape(-1)

    lat_coord = ds[latn].values
    lon_coord = ds[lonn].values

    if lat_coord.ndim == 1 and lon_coord.ndim == 1:
        lat2d = np.broadcast_to(lat_coord[:, None], (lat_coord.size, lon_coord.size))
    elif lat_coord.ndim == 2:
        lat2d = lat_coord
    else:
        raise ValueError(f"Unsupported lat/lon shapes: lat.ndim={lat_coord.ndim}, lon.ndim={lon_coord.ndim}")

    return lat2d.reshape(-1)[mask_flat].astype(np.float32)


def extract_rectilinear_grid_metadata(
        ds,
        mask_var: str = "elv",
        lat_name: str = None,
        lon_name: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lat, lon, valid_mask)`` for a rectilinear basin grid.

    The flattened valid cells are deliberately in C order so that they align
    with the point dimension produced by :func:`load_nc_vars` and validation
    can restore gridded outputs from the original basin NetCDF file.
    """
    latn, lonn = _infer_lat_lon_names(ds, lat_name=lat_name, lon_name=lon_name)
    lat = np.asarray(ds[latn].values, dtype=np.float64)
    lon = np.asarray(ds[lonn].values, dtype=np.float64)
    valid_mask = ~np.isnan(np.asarray(ds[mask_var].values))

    if valid_mask.ndim != 2:
        raise ValueError(f"{mask_var} must be a 2D spatial mask, got shape={valid_mask.shape}")
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            "Only rectilinear 1D latitude/longitude coordinates are supported for "
            f"(time, lat, lon) validation output; got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}."
        )
    if valid_mask.shape != (lat.size, lon.size):
        raise ValueError(
            "Spatial mask shape does not match latitude/longitude coordinates: "
            f"mask={valid_mask.shape}, lat={lat.size}, lon={lon.size}"
        )
    return lat.astype(np.float32), lon.astype(np.float32), valid_mask.astype(bool)


def _solar_declination_rad(day_of_year: np.ndarray) -> np.ndarray:
    return np.deg2rad(23.45) * np.sin(2.0 * np.pi * (284.0 + day_of_year) / 365.0)


def _sunset_hour_angle(lat_rad: np.ndarray, decl_rad: np.ndarray) -> np.ndarray:
    x = -np.tan(lat_rad) * np.tan(decl_rad)
    return np.arccos(np.clip(x, -1.0, 1.0))


def estimate_pet_hamon_mmday(
        temp_c: np.ndarray,
        dates,
        lat_deg: np.ndarray,
) -> np.ndarray:
    """Estimate daily PET in mm/day using the Hamon method."""
    temp_c = np.asarray(temp_c, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    if temp_c.ndim != 2:
        raise ValueError(f"temp_c must have shape (P, T), got {temp_c.shape}")
    if temp_c.shape[0] != lat_deg.shape[0]:
        raise ValueError(f"temp_c pixel count {temp_c.shape[0]} != lat count {lat_deg.shape[0]}")
    if np.nanmedian(temp_c) > 100.0:
        temp_c = temp_c - 273.15

    dates = np.asarray(dates)
    day_of_year = (dates.astype('datetime64[D]') - dates.astype('datetime64[Y]')).astype(np.int32) + 1
    lat_rad = np.deg2rad(np.clip(lat_deg, -89.999, 89.999))[:, None]
    decl = _solar_declination_rad(day_of_year.astype(np.float64))[None, :]
    omega_s = _sunset_hour_angle(lat_rad, decl)
    day_length = 24.0 / np.pi * omega_s

    temp_pos = np.maximum(temp_c, 0.0)
    sat_vapor_density = 216.7 * (6.108 * np.exp((17.27 * temp_pos) / (temp_pos + 237.3))) / (temp_pos + 273.3)
    pet_mmday = 0.1651 * day_length * sat_vapor_density
    pet_mmday[~np.isfinite(pet_mmday)] = 0.0
    return pet_mmday.astype(np.float32)


def compute_water_year(target_dates, start_month: int = 10) -> np.ndarray:
    dates = np.asarray(target_dates).astype('datetime64[ns]')
    month_index = dates.astype('datetime64[M]')
    year = month_index.astype('datetime64[Y]').astype(np.int32) + 1970
    month = (month_index.astype(np.int32) % 12) + 1
    water_year = year + (month >= int(start_month))
    return water_year.astype(np.int32)


def prepare_basin_data(nc_path, qobs_var: str = "discharge", time_name: str = "time"):
    # Legacy manual inspection helper; unused by the current training / validation pipeline.
    """
    读取数据立方体，计算流路距离，并打包为 Tensor
    """
    print(f"正在加载数据立方体: {nc_path} ...")
    ds = xr.open_dataset(nc_path)

    # --- A. 准备静态与动态特征 ---
    # 展平空间维度 (Y, X) -> (Pixels)
    # 剔除无效格点 (如边界外的 NaN)
    mask = ~np.isnan(ds['elev'].values)
    valid_pixels = mask.flatten()

    # 1. 动态特征: (Time, Y, X) -> (Pixels, Time, Features)
    # 注意维度变换：先取有效格点，再转置为 (Pixels, Time)
    precip = ds['precip'].values.reshape(ds.dims['time'], -1)[:, valid_pixels].T
    temp = ds['temp'].values.reshape(ds.dims['time'], -1)[:, valid_pixels].T

    # 归一化 (Z-Score)
    precip_norm = (precip - np.mean(precip)) / (np.std(precip) + 1e-6)
    temp_norm = (temp - np.mean(temp)) / (np.std(temp) + 1e-6)

    # 堆叠: (Pixels, Time, 2)
    x_dyn = np.stack([precip_norm, temp_norm], axis=2)

    # 2. 静态特征: (Pixels, Features)
    statics = []
    for var in ['elev', 'slope', 'clay', 'sand', 'cti']:
        val = ds[var].values.flatten()[valid_pixels]
        val = (val - np.mean(val)) / (np.std(val) + 1e-6)  # 归一化
        statics.append(val)
    x_stat = np.stack(statics, axis=1)

    # --- B. 计算水力流路距离 (Distance Map) ---
    print("正在计算水力流路距离 (可能需要几秒钟)...")
    grid = Grid()
    # 加载流向数据 (假设已经是有地理参考的)
    grid.add_gridded_data(ds['flow_dir'].values, data_name='dir', affine=ds.rio.transform())

    # 自动寻找流域出口 (汇流累积量最大的点)
    acc = grid.accumulation(data='dir')
    y_idx, x_idx = np.unravel_index(acc.argmax(), acc.shape)

    # 计算距离 (单位: 米)
    # dirmap 需根据流向数据的编码调整，这里假设是常用的 ESRI 格式
    dist = grid.flow_distance(data='dir', x=x_idx, y=y_idx, xytype='index', dirmap=(64, 128, 1, 2, 4, 8, 16, 32))
    dist_flat = dist.flatten()[valid_pixels]
    # 填充无效值为0
    dist_flat = np.nan_to_num(dist_flat, nan=0.0)

    # --- C. 从当前 basin nc 读取观测流量 ---
    q_obs = read_qobs_series_from_ds(ds, qobs_var=qobs_var, time_name=time_name)

    # --- D. 转换为 Tensor ---
    return {
        'x_dyn': torch.from_numpy(x_dyn).float(),
        'x_stat': torch.from_numpy(x_stat).float(),
        'dist_map': torch.from_numpy(dist_flat).float(),
        'q_obs': torch.from_numpy(q_obs).float(),
        'precip_raw': torch.from_numpy(precip).float(),  # 原始降雨用于物理约束
        'valid_mask': mask,  # 保存掩码以便后续还原地图
        'shape': (ds.dims['y'], ds.dims['x'])
    }
