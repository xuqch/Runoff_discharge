from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from config import get_args, update_run_config


def load_results(run_dir: Path):
    result_file = "metrics_summary.csv"
    print(f"Loading: {result_file}")
    results = pd.read_csv(run_dir / result_file, header=0)
    return results


def calc_nse(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((sim - obs) ** 2) / denom


def calc_kge(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)

    if len(obs) < 2:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1) if np.std(obs, ddof=1) > 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def clean_df(df):
    df = df.copy()
    # 仓库保存的是 qobs / qsim；画图和算指标时建议去掉无效观测值
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["qobs", "qsim"])
    df = df[df["qobs"] >= 0]
    return df


def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_results(df, BASIN_ID, OUT_DIR):
    nse = calc_nse(df["qobs"].values, df["qsim"].values)
    kge = calc_kge(df["qobs"].values, df["qsim"].values)
    fig, ax1 = plt.subplots(figsize=(13, 4))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # ===== 左轴：流量 =====
    ax1.plot(df["date"], df["qobs"], label="Observed", linewidth=1.0)
    ax1.plot(df["date"], df["qsim"], label="Simulated", linewidth=1.0)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Discharge")

    # ===== 右轴：降水 =====
    ax2 = ax1.twinx()
    ax2.invert_yaxis()
    ax2.bar(df["date"], df["precip_sum_mmday"], alpha=0.3, width=1, label="Precip")
    ax2.set_ylabel("Precipitation")

    # ===== 标题 =====
    plt.title(f"Basin {BASIN_ID} | NSE={nse:.3f}, KGE={kge:.3f}")

    # ===== 合并图例 =====
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{BASIN_ID}_hydrograph.png", dpi=200)
    plt.close()

def plot_hist_results(score_df, OUT_DIR):
    # NSE直方图
    plt.figure(figsize=(7, 4.5))
    plt.hist(score_df["NSE"].dropna(), bins=30)
    plt.xlim([-2, 1])
    plt.xlabel("NSE")
    plt.ylabel("ECDF")
    plt.xlabel("NSE")
    plt.ylabel("Number of basins")
    plt.title("Distribution of basin-wise NSE")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "nse_hist.png", dpi=200)
    plt.close()

    # NSE ECDF
    x, y = ecdf(score_df["NSE"].dropna().values)
    plt.figure(figsize=(6.5, 4.5))
    plt.step(x, y, where="post")
    plt.xlim([-2, 1])
    plt.xlabel("NSE")
    plt.ylabel("ECDF")
    plt.title("ECDF of basin-wise NSE")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "nse_ecdf.png", dpi=200)
    plt.close()

    print(f"\n所有图已保存到: {OUT_DIR}")


if __name__ == "__main__":
    config = get_args()
    cfg = update_run_config(config)
    basins_info = pd.read_csv(cfg.get('q_file'), dtype={'basin_id': str})
    if 'basin_id' not in basins_info.columns:
        raise KeyError(
            f"{cfg.get('q_file')} must contain a 'basin_id' column. "
            f"Available columns: {list(basins_info.columns)}"
        )
    basins_info['basin_id'] = basins_info['basin_id'].astype(str).str.strip()

    eval_dir = cfg['run_dir'] / 'eval'
    figure_dir = cfg['run_dir'] / 'figure'
    for basin_id in basins_info['basin_id']:
        df = pd.read_csv(eval_dir / f'{basin_id}.csv', dtype={'basin_id': str})
        plot_results(df, basin_id, figure_dir)

    metrics_df = pd.read_csv(eval_dir / 'metrics_summary.csv', dtype={'basin_id': str}, index_col='basin_id')
    plot_hist_results(metrics_df, figure_dir)
