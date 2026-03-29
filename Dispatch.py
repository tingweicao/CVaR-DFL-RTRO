# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1) User-editable styling
# =========================
# Fonts
FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 10.5
LABEL_PAD = 2.0

# Figure geometry
FIGSIZE = (7.5, 4.5)
HSPACE = 0.05
LEFT, RIGHT, TOP, BOTTOM = 0.08, 0.92, 0.96, 0.10

# Bars
BAR_WIDTH = 0.22
BAR_ALPHA = 0.90
BAR_EDGE_COLOR = "#2F3B52"
BAR_EDGE_LW = 0.3

# Colors
COLOR_BUY = "#3F74A6"
COLOR_SELL = "#C74A4A"
COLOR_DIS = "#4F9B64"
COLOR_CH = "#E39A3B"
COLOR_SOC = "#355C88"
SOC_LINESTYLE = '-'
SOC_LW = 1.1
SOC_MARKER = 'o'
SOC_MARKERSIZE = 3.5
SOC_MARKER_FACE = '#6AAFB0'
SOC_MARKER_EDGE = "black"
SOC_MARKER_EDGE_W = 0.8
SOC_MARK_EVERY = 4
COLOR_TRIGGER = "#66788F"
TRIGGER_ALPHA = 0.16
TRIGGER_LW = 0.8

# Grid / axes cosmetics
GRID_ALPHA = 0.35
GRID_LW = 0.8
GRID_LS = "--"
ZERO_LS = "--"
ZERO_LW = 1.0
ZERO_COLOR = "#6B7280"
AXES_LW = 1.0

# X ticks
XTICKS = np.arange(0, 25, 4)
XLIM = (-0.15, 24.0)


def run(csv_path: Path, out_png: Path, out_pdf: Path) -> None:
    # =========================
    # 2) Load data & preprocess
    # =========================
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Time axis: 15-min steps -> hours
    dt = 0.25
    if "t" in df.columns:
        x = df["t"].to_numpy(dtype=float) * dt
    else:
        x = np.arange(len(df), dtype=float) * dt

    # Signals (sell and charge plotted negative)
    P_buy = df["P_buy"].to_numpy(dtype=float)
    P_sell = df["P_sell"].to_numpy(dtype=float)
    P_dis = df["P_dis"].to_numpy(dtype=float)
    P_ch = df["P_ch"].to_numpy(dtype=float)

    y_sell = -P_sell
    y_ch = -P_ch

    trigger = (
        df["trigger"].to_numpy(dtype=int)
        if "trigger" in df.columns
        else np.zeros(len(df), dtype=int)
    )

    if "SOC_pct" in df.columns:
        soc_pct = df["SOC_pct"].to_numpy(dtype=float)
    else:
        soc_kwh = df["SOC_kWh"].to_numpy(dtype=float)
        soc_min = float(df["SOC_min_kWh"].iloc[0])
        soc_max = float(df["SOC_max_kWh"].iloc[0])
        soc_pct = 100.0 * (soc_kwh - soc_min) / max(1e-9, (soc_max - soc_min))
        soc_pct = np.clip(soc_pct, 0.0, 100.0)

    t_boundary = np.append(x, x[-1] + dt)
    soc_step = np.append(soc_pct, soc_pct[-1])

    # =========================
    # 3) Matplotlib global params
    # =========================
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": AXES_LW,
        }
    )

    # =========================
    # 4) Plot
    # =========================
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)
    ax2r = ax2.twinx()

    ax1.bar(
        x,
        P_buy,
        width=BAR_WIDTH,
        alpha=BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_LW,
        label="Buy",
        color=COLOR_BUY,
    )
    ax1.bar(
        x,
        y_sell,
        width=BAR_WIDTH,
        alpha=BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_LW,
        label="Sell",
        color=COLOR_SELL,
    )
    ax1.axhline(0, linestyle=ZERO_LS, linewidth=ZERO_LW, color=ZERO_COLOR)
    ax1.set_ylabel("Grid Exchange (kW)", labelpad=LABEL_PAD)
    ax1.grid(True, axis="y", linestyle=GRID_LS, linewidth=GRID_LW, alpha=GRID_ALPHA)

    ax2.bar(
        x,
        P_dis,
        width=BAR_WIDTH,
        alpha=BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_LW,
        label="Discharge",
        color=COLOR_DIS,
    )
    ax2.bar(
        x,
        y_ch,
        width=BAR_WIDTH,
        alpha=BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_LW,
        label="Charge",
        color=COLOR_CH,
    )
    ax2.axhline(0, linestyle=ZERO_LS, linewidth=ZERO_LW, color=ZERO_COLOR)
    ax2.set_ylabel("ESS Power (kW)", labelpad=LABEL_PAD)
    ax2.grid(True, axis="y", linestyle=GRID_LS, linewidth=GRID_LW, alpha=GRID_ALPHA)

    ax2r.step(
        t_boundary,
        soc_step,
        where='post',
        color=COLOR_SOC,
        linewidth=SOC_LW,
        linestyle=SOC_LINESTYLE,
        label="SOC",
    )
    ax2r.set_ylabel("ESS SOC (%)", labelpad=LABEL_PAD)
    ax2r.set_ylim(-2, 102)

    idx = np.where(trigger == 1)[0]
    if idx.size > 0:
        starts = idx[np.r_[True, np.diff(idx) > 1]]
        ends = idx[np.r_[np.diff(idx) > 1, True]]

        for k, (s, e) in enumerate(zip(starts, ends)):
            x0 = x[s] - dt / 2
            x1 = x[e] + dt / 2

            ax1.axvspan(
                x0,
                x1,
                color=COLOR_TRIGGER,
                alpha=TRIGGER_ALPHA,
                linewidth=0,
                zorder=0,
                label="Trigger" if k == 0 else None,
            )
            ax2.axvspan(x0, x1, color=COLOR_TRIGGER, alpha=TRIGGER_ALPHA, linewidth=0, zorder=0)

    ax1.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.01, 0.99), borderaxespad=0.0)

    ax2.set_xlabel("Time (hour)", labelpad=LABEL_PAD)
    ax2.set_xlim(*XLIM)
    ax2.set_xticks(XTICKS)

    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax2r.get_legend_handles_labels()
    ax2.legend(h2 + h3, l2 + l3, loc="upper left", frameon=False, ncol=1, bbox_to_anchor=(0.01, 0.99), borderaxespad=0.0)

    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM, hspace=HSPACE)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Saved:")
    print(" ", out_png)
    print(" ", out_pdf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online dispatch plotting from placeholder dispatch CSV.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to dispatch CSV. Default: ./datasets/RADFL/dispatch_placeholder_24h_15min_tunedCaps_v6.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figure files. Default: ./datasets/RADFL/dispatch_parts",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir / "datasets" / "RADFL"

    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else default_data_dir / "dispatch_placeholder_24h_15min_tunedCaps_v6.csv"
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else default_data_dir / "dispatch_parts"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Dispatch CSV not found: {csv_path}")

    run(csv_path=csv_path, out_png=out_dir / "dispatch.png", out_pdf=out_dir / "dispatch.pdf")


if __name__ == "__main__":
    main()
