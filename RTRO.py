# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# -----------------------------
# Global style (Matplotlib-only)
# -----------------------------
def set_global_style():
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman'],
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False
    })


def apply_axes_style(ax):
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=4,
        width=1,
        color='black',
        pad=6,
        bottom=True,
        top=False,
        left=True,
        right=False,
        zorder=100,
    )


AXES_BOX = [0.12, 0.15, 0.86, 0.78]
X_TICK_PAD = 4
SAVEFIG_PAD_INCHES = 0.02


def set_fixed_axes_box(ax, box=None):
    box = AXES_BOX if box is None else box
    ax.set_position(box)


# -----------------------------
# Default settings
# -----------------------------
DAY_TO_PLOT = "2022-04-29"
SYSTEM_TO_PLOT = 69

ONLINE_STEP_MIN = 15
FIXED_SOLVES_PER_DAY = int(24 * 60 / ONLINE_STEP_MIN)

TARGET_TRIGGERS_PER_DAY = 10
EVAL_START_DATE = "2022-04-10"
EVAL_NUM_DAYS = 20
RNG_SEED = 42


# -----------------------------
# Methods
# -----------------------------
METHODS = [
    "LSTM-FRO",
    "Transformer-FRO",
    "DFL-FRO",
    "CVaR-DFL-FRO",
    "CVaR-DFL-RTRO",
]

METHOD_NAME_MAP = {
    "LSTM-FixedRH": "LSTM-FRO",
    "Transformer-FixedRH": "Transformer-FRO",
    "DFL-FixedRH": "DFL-FRO",
    "RADFL-FixedRH": "CVaR-DFL-FRO",
    "RADFL-ARH": "CVaR-DFL-RTRO",
    "LSTM-FRO": "LSTM-FRO",
    "Transformer-FRO": "Transformer-FRO",
    "DFL-FRO": "DFL-FRO",
    "CVaR-DFL-FRO": "CVaR-DFL-FRO",
    "CVaR-DFL-RTRO": "CVaR-DFL-RTRO",
}

PALETTE = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#B279A2"]
PALETTE_DARK = ["#2F4B7C", "#3E7C79", "#B05A16", "#2E6B2E", "#6D4B7C"]


def standardize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).map(lambda x: METHOD_NAME_MAP.get(x, x))
    return df


def build_placeholder_trigger_log(ext_task_csv: str, pred_quantiles_csv: str, day_to_plot: str, target_triggers: int) -> pd.DataFrame:
    df_task = pd.read_csv(ext_task_csv, parse_dates=["timestamp"])
    df_true = df_task[df_task["split"] == "target"][["timestamp", "load"]].copy()

    df_pred = pd.read_csv(pred_quantiles_csv, parse_dates=["timestamp"])
    required_cols = {"timestamp", "load_0.05", "load_0.95"}
    if not required_cols.issubset(set(df_pred.columns)):
        raise ValueError(f"pred_quantiles must contain columns: {required_cols}")

    df_true_day = df_true[df_true["timestamp"].dt.strftime("%Y-%m-%d") == day_to_plot]
    df_pred_day = df_pred[df_pred["timestamp"].dt.strftime("%Y-%m-%d") == day_to_plot]
    df = df_true_day.merge(df_pred_day, on="timestamp", how="inner").sort_values("timestamp")
    if df.empty:
        raise ValueError("No aligned timestamps found for the specified day.")

    y = df["load"].to_numpy(float)
    q05 = df["load_0.05"].to_numpy(float)
    q95 = df["load_0.95"].to_numpy(float)
    width = np.maximum(1.0, q95 - q05)

    exceed = np.maximum(0.0, (y - q95) / width)
    ramp = np.concatenate([[0.0], np.abs(np.diff(y))]) / np.maximum(1.0, np.median(y))
    roll_exceed = pd.Series(exceed).rolling(4, min_periods=1).sum().to_numpy()

    psi = 0.8 * exceed + 0.4 * ramp + 0.6 * roll_exceed

    psi_sorted = np.sort(psi)
    epsilon = float(psi_sorted[-target_triggers]) if target_triggers < len(psi_sorted) else float(psi_sorted.max())
    if epsilon <= 0:
        epsilon = float(np.percentile(psi, 90) + 1e-6)

    trigger = (psi >= epsilon).astype(int)

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "Psi": psi,
        "epsilon": np.full(len(df), epsilon, dtype=float),
        "trigger": trigger,
    })


def build_placeholder_runtime_log(eval_start: str, n_days: int, fixed_solves_per_day: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(eval_start, periods=n_days, freq="D")

    params = {
        33: {
            "LSTM-FRO": (27.17, 0.24),
            "Transformer-FRO": (29.65, 0.24),
            "DFL-FRO": (32.13, 0.25),
            "CVaR-DFL-FRO": (37.81, 0.26),
            "CVaR-DFL-RTRO": (40.18, 0.28),
            "lam_trig": 8.0,
            "n_calls_clip": (6, 14),
        },
        69: {
            "LSTM-FRO": (42.44, 0.24),
            "Transformer-FRO": (47.18, 0.24),
            "DFL-FRO": (54.94, 0.25),
            "CVaR-DFL-FRO": (60.03, 0.26),
            "CVaR-DFL-RTRO": (69.34, 0.28),
            "lam_trig": 11.0,
            "n_calls_clip": (8, 16),
        },
    }

    fixed_methods = METHODS[:-1]
    rtro_method = METHODS[-1]

    rows = []
    for system in [33, 69]:
        lam_trig = params[system]["lam_trig"]
        n_call_low, n_call_high = params[system]["n_calls_clip"]

        method_mean = {m: params[system][m][0] for m in METHODS}
        mean_vals = np.array(list(method_mean.values()), dtype=float)
        mean_min = float(np.min(mean_vals))
        mean_max = float(np.max(mean_vals))
        mean_span = max(mean_max - mean_min, 1e-9)

        method_scale = {m: 0.7 + 1.0 * ((method_mean[m] - mean_min) / mean_span) for m in METHODS}

        for day in days:
            ts_grid = pd.date_range(day, periods=fixed_solves_per_day, freq=f"{ONLINE_STEP_MIN}min")

            for m in fixed_methods:
                mean_s, sigma = params[system][m]
                s = method_scale[m]
                day_factor = np.clip(1.0 + rng.normal(0.0, 0.02 + 0.04 * s), 0.85, 1.20)
                mean_eff = max(mean_s * day_factor, 1.0)
                sigma_eff = sigma * (0.92 + 0.18 * s)
                rt = rng.lognormal(mean=np.log(mean_eff), sigma=sigma_eff, size=len(ts_grid))
                for t, r in zip(ts_grid, rt):
                    rows.append((t, m, float(r), system))

            s_rt = method_scale[rtro_method]
            lam_eff = max(1.0, lam_trig * np.clip(1.0 + rng.normal(0.0, 0.02 + 0.02 * s_rt), 0.90, 1.15))
            n_calls = int(np.clip(rng.poisson(lam=lam_eff), n_call_low, n_call_high))

            hours = (ts_grid.hour + ts_grid.minute / 60.0).astype(float)
            w = np.exp(-0.5 * ((hours - 18.0) / 2.2) ** 2) + 0.20
            w = np.asarray(w, float)
            w = w / w.sum()

            idx = rng.choice(len(ts_grid), size=n_calls, replace=False, p=w)
            ts_rt = ts_grid[np.sort(idx)]

            mean_s, sigma = params[system][rtro_method]
            day_factor_rt = np.clip(1.0 + rng.normal(0.0, 0.015 + 0.02 * s_rt), 0.92, 1.08)
            mean_eff_rt = max(mean_s * day_factor_rt, 1.0)
            sigma_eff_rt = sigma * (0.88 + 0.12 * s_rt)
            rt_rt = rng.lognormal(mean=np.log(mean_eff_rt), sigma=sigma_eff_rt, size=len(ts_rt))
            for t, r in zip(ts_rt, rt_rt):
                rows.append((t, rtro_method, float(r), system))

    return pd.DataFrame(rows, columns=["timestamp", "method", "runtime_s", "system"])


def _p(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else np.nan


def plot_a_trigger_timeline(trigger_log: pd.DataFrame, day_to_plot: str, out_pdf: str, out_png: str):
    df = trigger_log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"].dt.strftime("%Y-%m-%d") == day_to_plot].sort_values("timestamp")
    if df.empty:
        raise ValueError("Trigger log does not contain the specified day.")

    df["Psi"] = df["Psi"].astype(float)
    df["epsilon"] = df["epsilon"].astype(float)
    df["trigger"] = df["trigger"].astype(int)
    df["psi_norm"] = df["Psi"] / df["epsilon"].replace(0, np.nan)

    t = df["timestamp"].reset_index(drop=True)
    y = df["psi_norm"].to_numpy(dtype=float)
    tr = df["trigger"].to_numpy(dtype=int) == 1

    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.fill_between(t, 0, y, color="#4FB57F", alpha=0.22, zorder=0)

    mask_red = np.asarray(y >= 1.0, dtype=bool)
    ax.fill_between(t, 0, y, where=mask_red, interpolate=True, color="#D73A49", alpha=0.22, zorder=1)

    for i in range(len(y)):
        if mask_red[i]:
            left_red = (i > 0 and mask_red[i - 1])
            right_red = (i < len(y) - 1 and mask_red[i + 1])
            if (not left_red) and (not right_red):
                ax.vlines(t.iloc[i], 0, y[i], color="#D73A49", linewidth=1.0, alpha=0.08, zorder=2)

    ax.plot(t, y, linewidth=2.6, color="#4FB57F", alpha=0.32, zorder=2)
    ax.plot(t, y, linewidth=1.8, color="#1F8A57", label=r"$\Psi_t/\epsilon$", zorder=3)
    ax.axhline(1.0, linestyle="--", linewidth=1.3, color="#D73A49", label="Threshold", zorder=2)

    ax.scatter(t[tr], y[tr], s=34, facecolor="white", edgecolor="#D73A49", linewidth=1.2, zorder=4, label="Activation")
    ax.set_ylabel(r"Risk indicator $\Psi_t/\epsilon$")

    ticks = pd.date_range(t.min().normalize(), t.min().normalize() + pd.Timedelta(hours=24), freq="6h")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    y_top = np.nanpercentile(y, 99.5) * 1.08
    ax.set_ylim(bottom=0, top=max(1.35, y_top))

    apply_axes_style(ax)
    ax.tick_params(axis="x", pad=X_TICK_PAD)
    set_fixed_axes_box(ax)

    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.0, borderpad=0.2, labelspacing=0.3)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    plt.close(fig)


def plot_b1_runtime_box(df_rt: pd.DataFrame, system_to_plot: int, out_pdf: str, out_png: str):
    df = df_rt[df_rt["system"] == system_to_plot].copy()
    df = df.dropna(subset=["runtime_s"])
    df["runtime_s"] = df["runtime_s"].astype(float)

    data = [df[df["method"] == m]["runtime_s"].to_numpy() for m in METHODS]
    positions = np.arange(1, len(METHODS) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 2.4))

    vp = ax.violinplot(data, positions=positions, widths=0.82, showmeans=False, showmedians=False, showextrema=False)
    for body, c in zip(vp["bodies"], PALETTE):
        body.set_facecolor(c)
        body.set_edgecolor("none")
        body.set_alpha(0.20)

    bp = ax.boxplot(data, positions=positions, widths=0.24, labels=METHODS, showfliers=False, whis=(5, 95), patch_artist=True)
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor("white")
        patch.set_alpha(0.95)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.4)

    for whisker in bp["whiskers"]:
        whisker.set_color("#666666")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("#666666")
        cap.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.4)

    rng = np.random.default_rng(2026)
    for x0, y, c in zip(positions, data, PALETTE):
        if len(y) == 0:
            continue
        if len(y) > 350:
            idx = rng.choice(len(y), size=350, replace=False)
            y_plot = y[idx]
        else:
            y_plot = y

        x_jitter = x0 + rng.uniform(-0.09, 0.09, size=len(y_plot))
        ax.scatter(x_jitter, y_plot, s=10, color=c, alpha=0.30, edgecolors="none", zorder=2)

    means = [np.mean(y) if len(y) > 0 else np.nan for y in data]
    ax.scatter(positions, means, marker="D", s=28, color="#1F1F1F", edgecolors="white", linewidths=0.6, zorder=4, label="Mean value")

    ax.set_ylabel("Solve time (s)")
    ax.set_xlabel("")
    ax.set_xticks(positions)
    ax.set_xticklabels(METHODS)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    ax.set_ylim(0, 180)

    apply_axes_style(ax)
    ax.tick_params(axis="x", pad=X_TICK_PAD)
    set_fixed_axes_box(ax)

    ax.legend(loc="upper left", frameon=False, fontsize=12, handletextpad=0.4, borderpad=0.2)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    plt.close(fig)


def plot_b2_cpu_day(df_rt: pd.DataFrame, system_to_plot: int, out_pdf: str, out_png: str):
    df = df_rt[df_rt["system"] == system_to_plot].copy()
    df = df.dropna(subset=["runtime_s"])
    df["runtime_s"] = df["runtime_s"].astype(float)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    daily = (
        df.groupby(["date", "method"])
        .agg(calls=("runtime_s", "size"), cpu_min=("runtime_s", lambda x: x.sum() / 60.0))
        .reset_index()
    )

    cpu_day_data = {m: daily[daily["method"] == m]["cpu_min"].to_numpy() for m in METHODS}
    cpu_med = np.array([_p(cpu_day_data[m], 50) for m in METHODS])
    cpu_p5 = np.array([_p(cpu_day_data[m], 5) for m in METHODS])
    cpu_p95 = np.array([_p(cpu_day_data[m], 95) for m in METHODS])
    yerr = np.vstack([cpu_med - cpu_p5, cpu_p95 - cpu_med])

    calls_median = np.array([float(np.median(daily[daily["method"] == m]["calls"])) for m in METHODS])

    x = np.arange(len(METHODS))

    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax2 = ax.twinx()

    ax.bar(x, cpu_med, width=0.52, color=PALETTE, edgecolor="#3A3A3A", linewidth=0.9, alpha=0.4, zorder=2)
    ax.errorbar(x, cpu_med, yerr=yerr, fmt="none", capsize=3.2, linewidth=1.1, ecolor="#222222", zorder=3)

    rng = np.random.default_rng(123)
    for i, m in enumerate(METHODS):
        yy = cpu_day_data[m]
        if len(yy) == 0:
            continue
        xx = np.full(len(yy), i, dtype=float) + (rng.random(len(yy)) - 0.5) * 0.16
        ax.scatter(xx, yy, s=14, alpha=0.30, color=PALETTE_DARK[i], edgecolors="none", zorder=4)

    ax2.scatter(x, calls_median, s=42, marker="o", facecolor="white", edgecolor="#222222", linewidth=1.2, zorder=5)
    ax2.plot(x, calls_median, color="#4B4B4B", linewidth=1.0, linestyle='dashed', alpha=0.90, zorder=1)

    ax.set_ylabel("Total time (min)")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10)

    all_cpu = np.concatenate([v for v in cpu_day_data.values() if len(v) > 0])
    y_top = np.percentile(all_cpu, 99.5) * 1.36
    ax.set_ylim(0, y_top)

    ax2.set_ylim(0, max(calls_median) * 1.61)
    ax2.tick_params(axis='y', right=False, labelright=False)

    apply_axes_style(ax)
    ax.tick_params(axis="x", pad=X_TICK_PAD)
    set_fixed_axes_box(ax)

    ax2.spines["right"].set_linewidth(1.0)
    ax2.spines["right"].set_color("#4B4B4B")

    legend_handles = [
        Patch(facecolor=PALETTE[0], edgecolor="#3A3A3A", alpha=0.82, label="Median CPU time/day"),
        Line2D([0], [0], marker="o", color="#4B4B4B", markerfacecolor="white", markeredgecolor="#222222", linewidth=1.0, linestyle="--", label="Median re-optimizations/day"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE_DARK[0], markeredgecolor="none", alpha=0.30, markersize=6, label="Daily runs"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=12, handlelength=1.8, borderpad=0.2, labelspacing=0.3, ncol=2)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=SAVEFIG_PAD_INCHES)
    plt.close(fig)


def resolve_paths(base_dir: str, out_dir: str = None):
    base_dir = os.path.abspath(base_dir)
    out_dir = os.path.abspath(out_dir) if out_dir else os.path.join(base_dir, "fig_iv03_parts")
    os.makedirs(out_dir, exist_ok=True)

    return {
        "ext_task_csv": os.path.join(base_dir, "extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv"),
        "pred_quantiles_csv": os.path.join(base_dir, "pred_quantiles_extreme.csv"),
        "trigger_log_real": os.path.join(base_dir, "rtro_trigger_day.csv"),
        "runtime_log_real": os.path.join(base_dir, "solver_runtime_all.csv"),
        "out_dir": out_dir,
        "out_a_pdf": os.path.join(out_dir, "a_trigger_timeline.pdf"),
        "out_a_png": os.path.join(out_dir, "a_trigger_timeline.png"),
        "out_b1_pdf": os.path.join(out_dir, "b_runtime_per_solve.pdf"),
        "out_b1_png": os.path.join(out_dir, "b_runtime_per_solve.png"),
        "out_b2_pdf": os.path.join(out_dir, "c_cpu_time_day.pdf"),
        "out_b2_png": os.path.join(out_dir, "c_cpu_time_day.png"),
        "out_trigger_csv": os.path.join(out_dir, "placeholder_rtro_trigger_day_extreme.csv"),
        "out_runtime_csv": os.path.join(out_dir, "placeholder_solver_runtime_all.csv"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run RTRO/ARH trigger-runtime plotting workflow.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory. Default: ./datasets/RADFL")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory. Default: <dataset-dir>/fig_iv03_parts")
    parser.add_argument("--use-real-logs", action="store_true", help="Use rtro_trigger_day.csv and solver_runtime_all.csv instead of placeholder generation.")
    parser.add_argument("--day-to-plot", type=str, default=DAY_TO_PLOT)
    parser.add_argument("--system-to-plot", type=int, choices=[33, 69], default=SYSTEM_TO_PLOT)
    parser.add_argument("--eval-start-date", type=str, default=EVAL_START_DATE)
    parser.add_argument("--eval-num-days", type=int, default=EVAL_NUM_DAYS)
    parser.add_argument("--target-triggers-per-day", type=int, default=TARGET_TRIGGERS_PER_DAY)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    set_global_style()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir if args.dataset_dir else str(script_dir / "datasets" / "RADFL")
    paths = resolve_paths(dataset_dir, args.out_dir)

    if not args.use_real_logs:
        if not os.path.exists(paths["ext_task_csv"]):
            raise FileNotFoundError(f"Missing file: {paths['ext_task_csv']}")
        if not os.path.exists(paths["pred_quantiles_csv"]):
            raise FileNotFoundError(f"Missing file: {paths['pred_quantiles_csv']}")

        trigger_log = build_placeholder_trigger_log(
            ext_task_csv=paths["ext_task_csv"],
            pred_quantiles_csv=paths["pred_quantiles_csv"],
            day_to_plot=args.day_to_plot,
            target_triggers=args.target_triggers_per_day,
        )
        runtime_log = build_placeholder_runtime_log(
            eval_start=args.eval_start_date,
            n_days=args.eval_num_days,
            fixed_solves_per_day=FIXED_SOLVES_PER_DAY,
            seed=args.seed,
        )
        runtime_log = standardize_method_names(runtime_log)

        trigger_log.to_csv(paths["out_trigger_csv"], index=False)
        runtime_log.to_csv(paths["out_runtime_csv"], index=False)
        print(f"[placeholder] trigger log saved: {paths['out_trigger_csv']}")
        print(f"[placeholder] runtime log saved: {paths['out_runtime_csv']}")
    else:
        if not os.path.exists(paths["trigger_log_real"]):
            raise FileNotFoundError(f"Missing file: {paths['trigger_log_real']}")
        if not os.path.exists(paths["runtime_log_real"]):
            raise FileNotFoundError(f"Missing file: {paths['runtime_log_real']}")
        trigger_log = pd.read_csv(paths["trigger_log_real"], parse_dates=["timestamp"])
        runtime_log = pd.read_csv(paths["runtime_log_real"], parse_dates=["timestamp"])
        runtime_log = standardize_method_names(runtime_log)

    plot_a_trigger_timeline(trigger_log, args.day_to_plot, paths["out_a_pdf"], paths["out_a_png"])
    plot_b1_runtime_box(runtime_log, args.system_to_plot, paths["out_b1_pdf"], paths["out_b1_png"])
    plot_b2_cpu_day(runtime_log, args.system_to_plot, paths["out_b2_pdf"], paths["out_b2_png"])

    print("Saved figure parts to:", paths["out_dir"])
    print(" -", paths["out_a_pdf"])
    print(" -", paths["out_b1_pdf"])
    print(" -", paths["out_b2_pdf"])


if __name__ == "__main__":
    main()
