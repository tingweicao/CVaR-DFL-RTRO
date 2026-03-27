import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import r2_score, rmse, smape
from darts.models import TFTModel

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

QUANTILES = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
]

AXES_BOX = [0.01, 0.03, 0.99, 0.99]
SAVEFIG_PAD_INCHES = 0.00


@dataclass(frozen=True)
class CaseConfig:
    name: str
    dataset_file: str
    split_ts: str
    checkpoint_name: str
    y_major_step: int
    y_min: float
    y_max: float
    quantiles_csv: str
    ci_suffix: str
    fig_prefix: str


def set_fixed_axes_box(ax, box=None) -> None:
    box = AXES_BOX if box is None else box
    ax.set_position(box)


def auto_fit_axes_box(fig, ax, pad=0.004) -> None:
    fig.canvas.draw()
    tight = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    left = max(0.0, tight.x0 - pad)
    bottom = max(0.0, tight.y0 - pad)
    right = min(1.0, tight.x1 + pad)
    top = min(1.0, tight.y1 + pad)
    ax.set_position([left, bottom, max(1e-6, right - left), max(1e-6, top - bottom)])


def apply_case_study_axes_style(ax) -> None:
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.2)

    ax.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_axisbelow(True)


def load_data_and_predict(
    case: CaseConfig, dataset_dir: Path, work_dir: Path
):
    data_path = dataset_dir / case.dataset_file
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    series = TimeSeries.from_csv(
        str(data_path), time_col="timestamp", value_cols=["load"]
    ).astype(np.float32)
    train, val = series.split_after(pd.Timestamp(case.split_ts))

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)

    model_dir = work_dir / case.checkpoint_name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint folder not found: {model_dir}. "
            f"Expected under work_dir={work_dir}."
        )

    load_model = TFTModel.load_from_checkpoint(
        case.checkpoint_name, best=True, work_dir=str(work_dir)
    )
    best_pred = load_model.predict(
        n=len(val), series=train_scaled, num_samples=1000
    )
    pred = scaler.inverse_transform(best_pred)
    return val, pred


def plot_forecast(
    case: CaseConfig, val, pred, output_dir: Path, show_plot: bool
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.4), facecolor="white", dpi=300)

    # Keep the same plotting style definition order as the source notebooks.
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    apply_case_study_axes_style(ax)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(case.y_major_step))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    pred.plot(
        low_quantile=0.05,
        high_quantile=0.95,
        label="_nolegend_",
        color="#D1E4FF",
        lw=1.0,
        zorder=10,
    )
    pred.plot(
        low_quantile=0.10,
        high_quantile=0.90,
        label="_nolegend_",
        color="#9FC2FF",
        lw=1.0,
        zorder=20,
    )
    pred.plot(
        low_quantile=0.15,
        high_quantile=0.85,
        label="_nolegend_",
        color="#739AF4",
        lw=1.0,
        zorder=30,
    )
    pred.plot(
        low_quantile=0.20,
        high_quantile=0.80,
        label="_nolegend_",
        color="#516DDF",
        lw=1.0,
        zorder=40,
    )
    pred.plot(
        low_quantile=0.25,
        high_quantile=0.75,
        label="_nolegend_",
        color="#3E2E97",
        lw=1.0,
        zorder=50,
    )
    val.plot(label="_nolegend_", color="#C62828", lw=1.2, zorder=60)

    ax.set_title(None)
    ax.set_ylabel("Load (kW)", color="black", labelpad=6, fontweight="normal", fontsize=14)
    ax.set_xlabel("")
    ax.xaxis.label.set_visible(False)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(case.y_major_step))
    ax.set_ylim(case.y_min, case.y_max)
    ax.minorticks_off()
    ax.tick_params(axis="y", which="major", colors="black")
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    legend_handles = [
        Patch(facecolor="#D1E4FF", edgecolor="none", label="90% PI"),
        Patch(facecolor="#9FC2FF", edgecolor="none", label="80% PI"),
        Patch(facecolor="#739AF4", edgecolor="none", label="70% PI"),
        Patch(facecolor="#516DDF", edgecolor="none", label="60% PI"),
        Patch(facecolor="#3E2E97", edgecolor="none", label="50% PI"),
        Line2D([0], [0], color="#C62828", lw=1.2, label="Actual"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        handlelength=1.8,
        borderpad=0.2,
        labelspacing=0.3,
    )

    set_fixed_axes_box(ax)
    auto_fit_axes_box(fig, ax)

    fig.savefig(
        output_dir / f"{case.fig_prefix}.pdf",
        bbox_inches="tight",
        pad_inches=SAVEFIG_PAD_INCHES,
    )
    fig.savefig(
        output_dir / f"{case.fig_prefix}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=SAVEFIG_PAD_INCHES,
    )
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def export_quantiles(case: CaseConfig, pred, output_dir: Path) -> None:
    qdf = pred.quantiles_df(tuple(QUANTILES))
    qdf.to_csv(output_dir / case.quantiles_csv)

    intervals = [0.9, 0.8, 0.7, 0.6, 0.5]
    for ci in intervals:
        low = (1 - ci) / 2
        high = 1 - low
        df_ci = pd.concat([pred.quantile_df(low), pred.quantile_df(high)], axis=1)
        df_ci.to_csv(output_dir / f"pred_ci_{int(ci * 100)}{case.ci_suffix}.csv")


def run_case(
    case: CaseConfig,
    dataset_dir: Path,
    work_dir: Path,
    output_dir: Path,
    show_plot: bool,
) -> None:
    print(f"\n========== {case.name.upper()} ==========")
    val, pred = load_data_and_predict(case, dataset_dir=dataset_dir, work_dir=work_dir)
    plot_forecast(case, val=val, pred=pred, output_dir=output_dir, show_plot=show_plot)
    export_quantiles(case, pred=pred, output_dir=output_dir)

    print(f"RMSE = {rmse(val, pred):.6f}")
    print(f"SMAPE = {smape(val, pred):.6f}")
    print(f"R^2 = {r2_score(val, pred):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run typical/extreme load inference from checkpoints."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures only, do not display plot windows.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Directory containing RADFL dataset CSVs. Default: ./datasets/RADFL",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory containing checkpoint folders (typical/extreme). Default: ./darts_logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for exported figures/CSVs. Default: script directory.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = (
        Path(args.dataset_dir).expanduser().resolve()
        if args.dataset_dir
        else script_dir / "datasets" / "RADFL"
    )
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else script_dir / "darts_logs"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else script_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        CaseConfig(
            name="typical",
            dataset_file="typical_forecast_task_L14d_2022-10-13_to_2022-10-27.csv",
            split_ts="2022/10/27 0:00:00",
            checkpoint_name="typical",
            y_major_step=1000,
            y_min=-200,
            y_max=6500,
            quantiles_csv="pred_quantiles.csv",
            ci_suffix="",
            fig_prefix="typical_day_forecast",
        ),
        CaseConfig(
            name="extreme",
            dataset_file="extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv",
            split_ts="2022/04/29 0:00:00",
            checkpoint_name="extreme",
            y_major_step=1500,
            y_min=-200,
            y_max=8500,
            quantiles_csv="pred_quantiles_extreme.csv",
            ci_suffix="_extreme",
            fig_prefix="extreme_day_forecast",
        ),
    ]

    for case in cases:
        run_case(
            case,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            show_plot=not args.no_show,
        )


if __name__ == "__main__":
    main()

