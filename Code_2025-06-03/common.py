from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def maybe_empty_cuda_cache(step: int, enabled: bool, interval: int) -> None:
    if not enabled or interval <= 0 or step % interval != 0:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_everything(seed: int) -> None:
    from utils import set_seed

    set_seed(int(seed))


def get_local_rank(default: int | None = 0) -> int:
    for key in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None and value != "":
            return int(value)
    return int(0 if default is None else default)


def setup_distributed(local_rank: int | None = None):
    import torch
    import torch.distributed as dist

    local_rank = get_local_rank(local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            try:
                dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))
            except TypeError:
                dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend=backend)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size, world_size > 1, local_rank


def ddp_barrier(local_rank: Optional[int] = None) -> None:
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        local_rank = get_local_rank(local_rank)
        dist.barrier(device_ids=[int(local_rank)])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def is_main_process() -> bool:
    try:
        import torch.distributed as dist

        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def resolve_device(local_rank: int = 0, requested: str = "cuda"):
    import torch

    if requested.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))
        return torch.device("cuda", int(local_rank))
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scaler,
    completed_epoch: int,
    global_step: int,
    best_loss: float,
    best_epoch: int,
    config: Dict,
) -> None:
    import torch

    raw_model = model.module if hasattr(model, "module") else model
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": int(completed_epoch),
            "completed_epoch": int(completed_epoch),
            "global_step": int(global_step),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, *, model, optimizer=None, scaler=None, map_location: str = "cpu") -> Dict:
    import torch

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def find_latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    latest = ckpt_dir / "latest.pth"
    if latest.exists():
        return latest
    candidates = sorted(ckpt_dir.glob("ckpt_epoch*.pth"))
    return candidates[-1] if candidates else None


def build_or_update_h5_if_needed(cfg: Dict) -> None:
    from build_h5 import create_h5_files, find_missing_basin_h5s
    from fit_scalers import fit_scalers_on_nc_dir

    if not Path(cfg["scalers_path"]).exists():
        fit_scalers_on_nc_dir(
            data_dir=cfg["data_dir"],
            nc_dir=cfg["nc_dir"],
            basins_file=cfg["basins_file"],
            out_path=cfg["scalers_path"],
            dyn_vars=cfg["dyn_vars"],
            stat_vars=cfg["stat_vars"],
            mask_var=cfg["mask_var"],
            qobs_var=cfg["qobs_var"],
            time_name=cfg["time_name"],
        )

    h5_dir = Path(cfg["h5_dir"])
    if h5_dir.exists():
        missing = find_missing_basin_h5s(cfg["basins_file"], h5_dir)
        requested_count = len(pd.read_csv(cfg["basins_file"], dtype={"basin_id": str})["basin_id"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique())
        existing_count = requested_count - len(missing)
        if not missing:
            print(f"[H5] h5_dir exists and all requested basin H5 files are present: {h5_dir}")
            return

        print(f"[H5] h5_dir exists: {h5_dir}")
        print(f"[H5] Requested basin count: {requested_count}")
        print(f"[H5] Existing requested basin H5 count: {existing_count}")
        print(f"[H5] Missing basin H5 count: {len(missing)}")
        print("[H5] Missing basin IDs, first 50:")
        for item in missing[:50]:
            print(f"  {item['basin_id']}")
        if len(missing) > 50:
            print(f"[H5] ... {len(missing) - 50} more missing basins not shown")

        report_path = h5_dir / "missing_basin_h5s.csv"
        pd.DataFrame(missing, columns=["basin_id", "expected_h5_path"]).to_csv(report_path, index=False)
        print(f"[H5] Full missing list written to: {report_path}")
        print("[H5] No basin H5 files were built because h5_dir already exists.")
        return

    print(f"[H5] h5_dir does not exist, building all basin H5 files: {h5_dir}")
    h5_dir.mkdir(parents=True, exist_ok=True)

    create_h5_files(
        nc_dir=cfg["nc_dir"],
        basins_file=cfg["basins_file"],
        scalers_path=cfg["scalers_path"],
        h5_dir=cfg["h5_dir"],
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
        target_start_date=cfg["train_start_date"],
        target_end_date=cfg["train_end_date"],
        fraction_var=cfg["fraction_var"],
        water_year_start_month=int(cfg.get("budyko_year_start_month", 10)),
        h5_build_workers=int(cfg.get("h5_build_workers", 1)),
    )


def build_criterion(cfg: Dict):
    from loss import HydroLoss, HydroMFMLoss

    common = {
        "w_q": float(cfg.get("w_q", 1.0)),
        "w_balance": float(cfg.get("w_balance", 0.0)),
        "balance_loss": str(cfg.get("balance_loss", "budyko_annual")),
        "budyko_alpha": float(cfg.get("budyko_alpha", 2.6)),
        "budyko_min_days_per_year": int(cfg.get("budyko_min_days_per_year", 300)),
    }
    if cfg["Loss"] == "NSEstd":
        return HydroLoss(**common)
    return HydroMFMLoss(
        **common,
        p=float(cfg["mfm_p"]),
        bins_suse=int(cfg["mfm_bins_suse"]),
        bins_phi=int(cfg["mfm_bins_phi"]),
        phase_penalty_scaling=float(cfg["mfm_phase_penalty_scaling"]),
        phase=bool(cfg["mfm_phase"]),
        eps=float(cfg["mfm_eps"]),
        soft_hist_sigma_scale=float(cfg["mfm_soft_hist_sigma_scale"]),
        w_peak=float(cfg.get("w_peak", 0.0)),
        peak_quantile=float(cfg.get("peak_quantile", 0.9)),
        peak_weight=float(cfg.get("peak_weight", 1.0)),
        peak_eps=float(cfg.get("peak_eps", 0.5)),
        peak_min_valid_count=int(cfg.get("peak_min_valid_count", 100)),
        peak_min_peak_count=int(cfg.get("peak_min_peak_count", 10)),
        peak_huber_beta=float(cfg.get("peak_huber_beta", 1.0)),
    )


def find_checkpoint(cfg: Dict) -> Path:
    if cfg.get("eval_ckpt"):
        p = Path(cfg["eval_ckpt"])
        if not p.exists():
            raise FileNotFoundError(f"eval_ckpt not found: {p}")
        return p

    run_dir = Path(cfg["run_dir"])
    for p in [run_dir / "best_model.pth", run_dir / "checkpoints" / "best_model.pth"]:
        if p.exists():
            return p

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_epoch*.pth"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def resolve_eval_checkpoint(run_dir: Path, eval_model: str, eval_ckpt: Optional[str] = None) -> Path:
    if eval_ckpt:
        p = Path(eval_ckpt)
        if not p.exists():
            raise FileNotFoundError(f"eval_ckpt not found: {p}")
        return p

    model_name = str(eval_model).strip().lower()
    if model_name not in {"best", "last"}:
        raise ValueError("eval_model must be either 'best' or 'last'.")

    if model_name == "best":
        candidates = [run_dir / "best_model.pth", run_dir / "checkpoints" / "best_model.pth"]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"No best checkpoint found under {run_dir}")

    latest_path = run_dir / "checkpoints" / "latest.pth"
    if latest_path.exists():
        return latest_path
    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_epoch*.pth"))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No last/latest checkpoint found under {run_dir}")


def resolve_eval_output_dir(run_dir: Path, eval_model: str) -> Path:
    model_name = str(eval_model).strip().lower()
    if model_name == "best":
        return run_dir / "eval_best"
    if model_name == "last":
        return run_dir / "eval"
    raise ValueError("eval_model must be either 'best' or 'last'.")


def setup_subdir(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    path.mkdir(parents=True, exist_ok=True)
    print(f"{name.capitalize()} directory: {path}")
    return path


def configure_process_tmp_dir(base_dir: Path) -> Path:
    env_override = os.environ.get("RUNOFF_SHORT_TMPDIR")
    if env_override:
        tmp_dir = Path(env_override)
    else:
        short_name = base_dir.name[-16:] if base_dir.name else "run"
        tmp_dir = Path("/data/xuqch3/.tmp/runoff_ealstm") / f"ealstm_{short_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir_str = str(tmp_dir)
    os.environ["TMPDIR"] = tmp_dir_str
    os.environ["TEMP"] = tmp_dir_str
    os.environ["TMP"] = tmp_dir_str
    tempfile.tempdir = tmp_dir_str
    return tmp_dir


def configure_run_artifact_dirs(base_dir: Path) -> Dict[str, Path]:
    runs_root = base_dir.parent if base_dir.parent != base_dir else base_dir
    temp_root = runs_root / "temp"
    tmp_path = configure_process_tmp_dir(base_dir)
    cache_root_env = os.environ.get("PHASE_CACHE_ROOT")
    cache_root = Path(cache_root_env) if cache_root_env else temp_root
    artifact_dirs = {
        "tmp": tmp_path,
        "cache_root": cache_root,
        "torchelastic_logs": cache_root / "torchelastic_logs" / base_dir.name,
        "torchinductor_cache": cache_root / "torchinductor_cache",
        "triton_cache": cache_root / "triton_cache",
        "cuda_cache": cache_root / "cuda_cache",
        "pycache": cache_root / "pycache",
    }
    for path in artifact_dirs.values():
        if path != artifact_dirs["tmp"]:
            path.mkdir(parents=True, exist_ok=True)
    os.environ["PHASE_CACHE_ROOT"] = str(cache_root)
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(artifact_dirs["torchelastic_logs"] / "torchelastic_error.json")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(artifact_dirs["torchinductor_cache"])
    os.environ["TRITON_CACHE_DIR"] = str(artifact_dirs["triton_cache"])
    os.environ["CUDA_CACHE_PATH"] = str(artifact_dirs["cuda_cache"])
    os.environ["PYTHONPYCACHEPREFIX"] = str(artifact_dirs["pycache"])
    return artifact_dirs


def int64ns_to_datetime(arr: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(arr.astype("int64"), unit="ns")


def _validate_inputs(s, o):
    """
    Validate and align input DataArrays.

    Args:
        s (xr.DataArray): Simulated data
        o (xr.DataArray): Observed data

    Returns:
        tuple: Aligned and validated DataArrays
    """
    # Remove NaN values
    mask = np.isfinite(s) & np.isfinite(o)
    return s.where(mask), o.where(mask)


def mfm_1d_numpy(
    obs: np.ndarray,
    sim: np.ndarray,
    p: float = 1.0,
    bins_suse: int = 10,
    bins_phi: int = 10,
    phase_penalty_scaling: float = 4.0,
    phase: bool = True,
) -> float:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    sim = np.asarray(sim, dtype=np.float64).reshape(-1)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if obs.size < 3 or sim.size < 3:
        return np.nan
    mean_obs = float(np.mean(obs))
    if mean_obs == 0.0 or not np.isfinite(mean_obs):
        return np.nan

    def _phi_component(sim_series: np.ndarray, obs_series: np.ndarray) -> float:
        bin_min = min(float(np.min(sim_series)), float(np.min(obs_series)))
        bin_max = max(float(np.max(sim_series)), float(np.max(obs_series)))
        if bin_min == bin_max:
            return 1.0
        bin_edges = np.linspace(bin_min, bin_max, bins_phi + 1)
        hist_sim, _ = np.histogram(sim_series, bins=bin_edges, density=False)
        hist_obs, _ = np.histogram(obs_series, bins=bin_edges, density=False)
        obs_total = np.sum(hist_obs)
        if obs_total <= 0:
            return np.nan
        return float(np.sum(np.minimum(hist_sim, hist_obs)) / obs_total)

    def _entropy(values: np.ndarray) -> float:
        values = values[values > 0]
        if values.size == 0:
            return 0.0
        return float(-np.sum(values * np.log(values)))

    def _suse_component(sim_series: np.ndarray, obs_series: np.ndarray) -> float:
        min_val = min(float(sim_series.min()), float(obs_series.min()))
        max_val = max(float(sim_series.max()), float(obs_series.max()))
        if min_val == max_val:
            return 0.0

        bin_edges_scaled = np.linspace(min_val, max_val, bins_suse + 1)
        hist_sim_s, _ = np.histogram(sim_series, bins=bin_edges_scaled, density=False)
        hist_obs_s, _ = np.histogram(obs_series, bins=bin_edges_scaled, density=False)

        p_sim_s = hist_sim_s / np.sum(hist_sim_s) if np.sum(hist_sim_s) > 0 else np.zeros_like(hist_sim_s, dtype=np.float64)
        p_obs_s = hist_obs_s / np.sum(hist_obs_s) if np.sum(hist_obs_s) > 0 else np.zeros_like(hist_obs_s, dtype=np.float64)
        hs = abs(_entropy(p_sim_s) - _entropy(p_obs_s))

        if sim_series.min() == sim_series.max():
            hu_sim = 0.0
        else:
            sim_edges = np.linspace(float(sim_series.min()), float(sim_series.max()), bins_suse + 1)
            hist_sim_u, _ = np.histogram(sim_series, bins=sim_edges, density=False)
            p_sim_u = hist_sim_u / np.sum(hist_sim_u) if np.sum(hist_sim_u) > 0 else np.zeros_like(hist_sim_u, dtype=np.float64)
            hu_sim = _entropy(p_sim_u)

        if obs_series.min() == obs_series.max():
            hu_obs = 0.0
        else:
            obs_edges = np.linspace(float(obs_series.min()), float(obs_series.max()), bins_suse + 1)
            hist_obs_u, _ = np.histogram(obs_series, bins=obs_edges, density=False)
            p_obs_u = hist_obs_u / np.sum(hist_obs_u) if np.sum(hist_obs_u) > 0 else np.zeros_like(hist_obs_u, dtype=np.float64)
            hu_obs = _entropy(p_obs_u)
        return float(max(hs, abs(hu_sim - hu_obs)))

    def _fft_phase_component(sim_series: np.ndarray, obs_series: np.ndarray) -> float:
        n = len(obs_series)
        if n != len(sim_series) or n < 3:
            return 0.0
        fft_obs = np.fft.fft(obs_series)
        fft_sim = np.fft.fft(sim_series)
        if n // 2 < 1:
            return 0.0
        if len(sim_series) > 365:
            dominant_freq_idx = max(np.argmax(np.abs(fft_obs[1:n // 2 + 1])), 33) + 1
        else:
            dominant_freq_idx = np.argmax(np.abs(fft_obs[1:n // 2 + 1])) + 1
        phase_obs = np.angle(fft_obs)
        phase_sim = np.angle(fft_sim)
        phase_difference_rad = phase_sim[dominant_freq_idx] - phase_obs[dominant_freq_idx]
        return float((phase_difference_rad + np.pi) % (2 * np.pi) - np.pi)

    nmaep = np.power(np.mean(np.power(np.abs(sim - obs), p)), 1.0 / p) / abs(mean_obs)
    if phase:
        phase_penalty = np.cos(_fft_phase_component(sim, obs) / phase_penalty_scaling)
        normalized_error = phase_penalty * np.e ** (-nmaep)
    else:
        normalized_error = np.e ** (-nmaep)
    variability_capture = np.e ** (-_suse_component(sim, obs))
    distribution_similarity = _phi_component(sim, obs)
    if not np.isfinite(distribution_similarity):
        return np.nan

    mfm_value = 1.0 - np.sqrt(
        (
            (1.0 - normalized_error) ** 2
            + (1.0 - variability_capture) ** 2
            + (1.0 - distribution_similarity) ** 2
        ) / 3.0
    )
    return float(mfm_value)


def MFM(s, o, p=1, bins_suse=10, bins_phi=10, phase_penalty_scaling=4, phase=True):
    """
    Calculate Model Fidelity Metric (MFM) for each grid cell.

    MFM integrates four components:
    1. Normalized Mean Absolute p-Error (NMAEp) - relative error
    2. Scaled and Unscaled Entropy difference (SUSE) - variability capture
    3. Percentage of Histogram Intersection (PHI) - distribution matching
    4. Phase Difference Radius - phase difference (optional)

    Args:
        s (xr.DataArray): Simulated data (time, lat, lon)
        o (xr.DataArray): Observed data (time, lat, lon)
        p (float): Exponent for error calculation (default=1, p=1 gives MAE, p=2 gives RMSE)
        bins_suse (int): Number of bins for entropy calculation (default=10)
        bins_phi (int): Number of bins for histogram intersection (default=10)
        phase_penalty_scaling (float): Scaling factor for phase difference penalty (default=4)
        phase (bool): Whether to include phase difference component (default=True)

    Returns:
        xr.DataArray: Model Fidelity Metric value (lat, lon)
    """

    # Validate and align inputs
    s, o = _validate_inputs(s, o)

    # Helper functions for single time series
    def PHI_component(sim, obs, bins_phi):
        """Calculate Percentage of Histogram Intersection"""
        if len(sim) == 0 or len(obs) == 0:
            return np.nan
        bin_min = min(np.min(sim), np.min(obs))
        bin_max = max(np.max(sim), np.max(obs))
        if bin_min == bin_max:
            return 1.0  # Perfect match if all values are the same
        bin_edges = np.linspace(bin_min, bin_max, bins_phi + 1)
        hist_sim, _ = np.histogram(sim, bins=bin_edges, density=False)
        hist_obs, _ = np.histogram(obs, bins=bin_edges, density=False)
        min_sum = np.sum(np.minimum(hist_sim, hist_obs))
        obs_total = np.sum(hist_obs)
        if obs_total == 0:
            return np.nan
        return min_sum / obs_total

    def SUSE_component(sim, obs, bins_suse):
        """Calculate Scaled and Unscaled Entropy difference"""
        if len(sim) == 0 or len(obs) == 0:
            return np.nan

        # Scaled case
        min_val = min(sim.min(), obs.min())
        max_val = max(sim.max(), obs.max())
        if min_val == max_val:
            return 0.0  # No entropy difference if all values are the same
        bin_edges_scaled = np.linspace(min_val, max_val, bins_suse + 1)

        hist_sim_s, _ = np.histogram(sim, bins=bin_edges_scaled, density=False)
        hist_obs_s, _ = np.histogram(obs, bins=bin_edges_scaled, density=False)

        total_s_sim = np.sum(hist_sim_s)
        total_s_obs = np.sum(hist_obs_s)

        p_sim_s = hist_sim_s / total_s_sim if total_s_sim > 0 else np.zeros_like(hist_sim_s)
        p_obs_s = hist_obs_s / total_s_obs if total_s_obs > 0 else np.zeros_like(hist_obs_s)

        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log(p)) if len(p) > 0 else 0.0

        Hs = abs(entropy(p_sim_s) - entropy(p_obs_s))

        # Unscaled case
        if sim.min() == sim.max():
            Hu_sim = 0.0
        else:
            bin_edges_u_sim = np.linspace(sim.min(), sim.max(), bins_suse + 1)
            hist_sim_u, _ = np.histogram(sim, bins=bin_edges_u_sim, density=False)
            p_sim_u = hist_sim_u / np.sum(hist_sim_u) if np.sum(hist_sim_u) > 0 else np.zeros_like(hist_sim_u)
            Hu_sim = entropy(p_sim_u)

        if obs.min() == obs.max():
            Hu_obs = 0.0
        else:
            bin_edges_u_obs = np.linspace(obs.min(), obs.max(), bins_suse + 1)
            hist_obs_u, _ = np.histogram(obs, bins=bin_edges_u_obs, density=False)
            p_obs_u = hist_obs_u / np.sum(hist_obs_u) if np.sum(hist_obs_u) > 0 else np.zeros_like(hist_obs_u)
            Hu_obs = entropy(p_obs_u)

        Hu = abs(Hu_sim - Hu_obs)

        return max(Hs, Hu)

    def FFT_component(sim, obs):
        """Calculate phase difference using Fast Fourier Transform"""
        N = len(obs)
        if N != len(sim) or N < 3:
            return 0.0

        fft_obs = np.fft.fft(obs)
        fft_sim = np.fft.fft(sim)

        freqs = np.fft.fftfreq(N, d=1.0)

        # Find dominant frequency
        if N // 2 < 1:
            return 0.0

        if len(sim) > 365:
            dominant_freq_idx = max(np.argmax(np.abs(fft_obs[1:N // 2 + 1])), 33) + 1
        else:
            dominant_freq_idx = np.argmax(np.abs(fft_obs[1:N // 2 + 1])) + 1

        # Calculate phase difference
        phase_obs = np.angle(fft_obs)
        phase_sim = np.angle(fft_sim)
        phase_difference_rad = phase_sim[dominant_freq_idx] - phase_obs[dominant_freq_idx]
        phase_difference_rad = (phase_difference_rad + np.pi) % (2 * np.pi) - np.pi

        return phase_difference_rad

    def calculate_mfm_1d(sim, obs):
        """Calculate MFM for a single time series"""
        # Remove NaN values
        mask = np.isfinite(sim) & np.isfinite(obs)
        sim_clean = sim[mask]
        obs_clean = obs[mask]

        if len(sim_clean) < 3 or len(obs_clean) < 3:
            return np.nan

        if np.mean(obs_clean) == 0:
            return np.nan

        # Calculate components
        # 1. Normalized error with phase penalty
        nmaep = np.power(np.mean(np.power(np.abs(sim_clean - obs_clean), p)), 1 / p) / abs(np.mean(obs_clean))

        if phase:
            phase_difference_rad = FFT_component(sim_clean, obs_clean)
            phase_penalty = np.cos(phase_difference_rad / phase_penalty_scaling)
            normalized_error = phase_penalty * np.e ** (-nmaep)
        else:
            normalized_error = np.e ** (-nmaep)

        # 2. Variability capture
        suse = SUSE_component(sim_clean, obs_clean, bins_suse)
        if np.isnan(suse):
            return np.nan
        variability_capture = np.e ** (-suse)

        # 3. Distribution similarity
        distribution_similarity = PHI_component(sim_clean, obs_clean, bins_phi)
        if np.isnan(distribution_similarity):
            return np.nan

        # Calculate MFM
        mfm_value = 1 - np.sqrt((
                                        (1 - normalized_error) ** 2 +
                                        (1 - variability_capture) ** 2 +
                                        (1 - distribution_similarity) ** 2
                                ) / 3)

        return mfm_value

    # Apply MFM to each grid cell
    # Get dimensions
    if 'time' in s.dims:
        # Rechunk time dimension to single chunk for apply_ufunc with dask
        # This is required because time is a core dimension
        if hasattr(s, 'chunks') and s.chunks is not None:
            s = s.chunk({'time': -1})
        if hasattr(o, 'chunks') and o.chunks is not None:
            o = o.chunk({'time': -1})

        # Stack spatial dimensions for easier iteration
        result = xr.apply_ufunc(
            calculate_mfm_1d,
            s,
            o,
            input_core_dims=[['time'], ['time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )
    else:
        # No time dimension, return NaN
        result = xr.full_like(s.isel(time=0) if 'time' in s.dims else s, np.nan)

    return result


def calc_metrics(obs: np.ndarray, sim: np.ndarray, include_mfm: bool = True) -> Dict[str, float]:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    sim = np.asarray(sim, dtype=np.float64).reshape(-1)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) == 0:
        metrics = {"NSE": np.nan, "KGE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "Corr": np.nan}
        if include_mfm:
            metrics["MFM"] = np.nan
        return metrics

    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    mae = float(np.mean(np.abs(sim - obs)))
    bias = float(np.mean(sim - obs))
    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse = float(1.0 - np.sum((sim - obs) ** 2) / denom) if denom > 0 else np.nan
    corr = float(np.corrcoef(obs, sim)[0, 1]) if len(obs) > 1 and np.std(obs) > 0 and np.std(sim) > 0 else np.nan

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs, ddof=0)
    std_sim = np.std(sim, ddof=0)
    if mean_obs != 0 and std_obs != 0 and np.isfinite(corr):
        alpha = std_sim / std_obs
        beta = mean_sim / mean_obs
        kge = float(1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = np.nan
    metrics = {"NSE": nse, "KGE": kge, "RMSE": rmse, "MAE": mae, "Bias": bias, "Corr": corr}
    if include_mfm:
        metrics["MFM"] = mfm_1d_numpy(obs=obs, sim=sim)
    return metrics


def append_basin_batch_note() -> str:
    return (
        "Recommended basin batching: single GPU starts with basin_batch_size=1 or 2; "
        "multi-GPU DDP starts with per-rank basin_batch_size=1, then increase to 2/4 if memory allows. "
        "A future optimization can group same-prefix_len basins for generator batching while keeping routing per basin."
    )
