from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import mfm_1d_numpy
from config import get_args, update_run_config


METRIC_FILES = ("train_metrics.csv", "validation_metrics.csv")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid discharge values before calculating MFM."""
    return (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["qobs", "qsim"])
        .loc[lambda frame: frame["qobs"] >= 0]
    )


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def load_basin_metrics(eval_dir: Path) -> pd.DataFrame:
    """Load saved NSE/KGE values for both evaluation splits.

    Validation values take precedence if a basin appears in both files, which
    matches the purpose of this plotting script.
    """
    metric_frames: list[pd.DataFrame] = []
    for filename in METRIC_FILES:
        path = eval_dir / filename
        if not path.exists():
            print(f"Metric file not found, skipped: {path}")
            continue
        metrics = pd.read_csv(path, dtype={"basin_id": str})
        required_columns = {"basin_id", "nse", "kge"}
        missing_columns = required_columns.difference(metrics.columns)
        if missing_columns:
            raise KeyError(f"{path} is missing columns: {sorted(missing_columns)}")
        metric_frames.append(metrics)

    if not metric_frames:
        expected = ", ".join(str(eval_dir / filename) for filename in METRIC_FILES)
        raise FileNotFoundError(f"No metric files were found. Expected one of: {expected}")

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics["basin_id"] = metrics["basin_id"].astype(str).str.strip()
    return metrics.drop_duplicates(subset="basin_id", keep="last").set_index("basin_id")


def plot_results(
    basin_id: str,
    eval_dir: Path,
    out_dir: Path,
    basin_metrics: dict[str, dict[str, float]],
) -> None:
    """Load and plot a single basin. Suitable for joblib process parallelism."""
    # Load the time-series inside the worker so it is not serialized to workers.
    basin_file = eval_dir / f"{basin_id}.csv"
    if not basin_file.exists():
        print(f"Basin result not found, skipped: {basin_file}")
        return
    df = pd.read_csv(basin_file, dtype={"basin_id": str})
    required_columns = {"date", "qobs", "qsim", "precip_sum_mmday"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(f"{basin_file} is missing columns: {sorted(missing_columns)}")

    metric_row = basin_metrics.get(basin_id)
    if metric_row is None:
        print(f"Saved NSE/KGE not found for basin {basin_id}, skipped: {basin_file}")
        return
    nse = metric_row["nse"]
    kge = metric_row["kge"]

    valid_df = clean_df(df)
    mfm = mfm_1d_numpy(valid_df["qobs"].to_numpy(), valid_df["qsim"].to_numpy())
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(13, 4))
    ax1.plot(df["date"], df["qobs"], label="Observed", linewidth=1.0)
    ax1.plot(df["date"], df["qsim"], label="Simulated", linewidth=1.0)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Discharge")

    ax2 = ax1.twinx()
    ax2.invert_yaxis()
    ax2.bar(df["date"], df["precip_sum_mmday"], alpha=0.3, width=1, label="Precip")
    ax2.set_ylabel("Precipitation")

    ax1.set_title(f"Basin {basin_id} | NSE={nse:.3f}, KGE={kge:.3f}, MFM={mfm:.3f}")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / f"{basin_id}.png", dpi=200)
    plt.close(fig)


def plot_hist_results(score_df: pd.DataFrame, out_dir: Path) -> None:
    nse = score_df["nse"].dropna()
    if nse.empty:
        return

    plt.figure(figsize=(7, 4.5))
    plt.hist(nse, bins=30)
    plt.xlim([-2, 1])
    plt.xlabel("NSE")
    plt.ylabel("Number of basins")
    plt.title("Distribution of basin-wise NSE")
    plt.tight_layout()
    plt.savefig(out_dir / "nse_hist.png", dpi=200)
    plt.close()

    x, y = ecdf(nse.to_numpy())
    plt.figure(figsize=(6.5, 4.5))
    plt.step(x, y, where="post")
    plt.xlim([-2, 1])
    plt.xlabel("NSE")
    plt.ylabel("ECDF")
    plt.title("ECDF of basin-wise NSE")
    plt.tight_layout()
    plt.savefig(out_dir / "nse_ecdf.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    config = get_args()
    cfg = update_run_config(config)
    basins_info = pd.read_csv(cfg["q_file"], dtype={"basin_id": str})
    if "basin_id" not in basins_info.columns:
        raise KeyError(
            f"{cfg['q_file']} must contain a 'basin_id' column. "
            f"Available columns: {list(basins_info.columns)}"
        )

    basin_ids = basins_info["basin_id"].astype(str).str.strip().drop_duplicates().tolist()
    eval_dir = cfg["run_dir"] / "eval"
    figure_dir = cfg["run_dir"] / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_basin_metrics(eval_dir)
    metrics_by_basin = metrics_df[["nse", "kge"]].to_dict(orient="index")
    n_jobs = min(len(basin_ids), os.cpu_count() or 1)
    if basin_ids:
        Parallel(n_jobs=n_jobs)(
            delayed(plot_results)(basin_id, eval_dir, figure_dir, metrics_by_basin)
            for basin_id in basin_ids
        )

    plot_hist_results(metrics_df, figure_dir)
    print(f"All figures have been saved to: {figure_dir}")
