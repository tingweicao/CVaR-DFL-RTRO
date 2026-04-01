from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from RTRO import (
    EVAL_NUM_DAYS,
    EVAL_START_DATE,
    FIXED_SOLVES_PER_DAY,
    RNG_SEED,
    build_placeholder_runtime_log,
    standardize_method_names,
)
from optimization_cvxpy_demo import _resolve_solver_name, run_dispatch, run_rtro


DEFAULT_XI_G = 0.06
DEFAULT_XI_C = 0.8
DEFAULT_DELTA_MIN = 4
DEFAULT_DELTA_MAX = 12
DEFAULT_DAY = "2022-04-29"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate paper reproduction path without Gurobi. "
            "Generates a CVXPY dispatch result, a tuned RTRO trigger/day result, and a paper-style runtime log."
        )
    )
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory. Default: ./datasets/RADFL")
    parser.add_argument("--day", type=str, default=DEFAULT_DAY, help="Target day in YYYY-MM-DD")
    parser.add_argument("--solver", type=str, default="auto", help="CVXPY solver name. Default: auto")
    parser.add_argument("--system-id", type=int, choices=[33, 69], default=69, help="System ID for runtime-day CSV")
    parser.add_argument("--xi-g", type=float, default=DEFAULT_XI_G, help="Recommended RTRO xi_g")
    parser.add_argument("--xi-c", type=float, default=DEFAULT_XI_C, help="Recommended RTRO xi_c")
    parser.add_argument("--delta-min", type=int, default=DEFAULT_DELTA_MIN, help="Recommended RTRO min trigger window")
    parser.add_argument("--delta-max", type=int, default=DEFAULT_DELTA_MAX, help="Recommended RTRO max trigger window")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else script_dir / "datasets" / "RADFL"
    fig_dir = dataset_dir / "fig_iv03_parts"
    fig_dir.mkdir(parents=True, exist_ok=True)

    solver_name = _resolve_solver_name(args.solver, requires_mip=False)

    dispatch_csv = dataset_dir / "dispatch_cvxpy_paper_repro.csv"
    dispatch_rtro_csv = dataset_dir / "dispatch_cvxpy_paper_repro_rtro.csv"
    trigger_csv = dataset_dir / "rtro_trigger_day.csv"
    trigger_archive_csv = fig_dir / "rtro_trigger_day_cvxpy_paper_repro.csv"
    runtime_day_csv = fig_dir / "solver_runtime_day_cvxpy_paper_repro.csv"
    runtime_all_csv = dataset_dir / "solver_runtime_all.csv"
    meta_json = dataset_dir / "cvxpy_paper_repro_metadata.json"

    run_dispatch(
        dataset_dir=dataset_dir,
        day=args.day,
        out_dispatch_csv=dispatch_csv,
        solver_name=solver_name,
        output_flag=0,
        use_binary_stage1=False,
    )

    _, _, runtime_day_path = run_rtro(
        dataset_dir=dataset_dir,
        day=args.day,
        out_dispatch_csv=dispatch_rtro_csv,
        out_trigger_csv=trigger_archive_csv,
        out_runtime_csv=runtime_day_csv,
        solver_name=solver_name,
        output_flag=0,
        use_binary_stage1=False,
        system_id=args.system_id,
        include_baselines=False,
        xi_g=args.xi_g,
        xi_c=args.xi_c,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
    )

    shutil.copyfile(trigger_archive_csv, trigger_csv)

    runtime_log = build_placeholder_runtime_log(
        eval_start=EVAL_START_DATE,
        n_days=EVAL_NUM_DAYS,
        fixed_solves_per_day=FIXED_SOLVES_PER_DAY,
        seed=RNG_SEED,
    )
    runtime_log = standardize_method_names(runtime_log)
    runtime_log.to_csv(runtime_all_csv, index=False)

    trigger_df = pd.read_csv(trigger_archive_csv)
    runtime_day_df = pd.read_csv(runtime_day_path)
    runtime_day_df = runtime_day_df[runtime_day_df["method"] == "CVaR-DFL-RTRO"].copy()

    meta = {
        "mode": "approximate_paper_reproduction_without_gurobi",
        "day": args.day,
        "solver": solver_name,
        "recommended_rtro_params": {
            "xi_g": float(args.xi_g),
            "xi_c": float(args.xi_c),
            "delta_min": int(args.delta_min),
            "delta_max": int(args.delta_max),
        },
        "outputs": {
            "dispatch_csv": str(dispatch_csv),
            "dispatch_rtro_csv": str(dispatch_rtro_csv),
            "trigger_csv": str(trigger_csv),
            "trigger_archive_csv": str(trigger_archive_csv),
            "runtime_day_csv": str(runtime_day_csv),
            "runtime_all_csv": str(runtime_all_csv),
        },
        "summary": {
            "trigger_count": int(trigger_df["trigger"].sum()),
            "trigger_rows": int(len(trigger_df)),
            "runtime_day_avg_s": float(runtime_day_df["runtime_s"].mean()) if len(runtime_day_df) else None,
            "runtime_day_max_s": float(runtime_day_df["runtime_s"].max()) if len(runtime_day_df) else None,
            "runtime_all_mode": "paper_calibrated_placeholder",
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Solver:", solver_name)
    print("Saved:")
    print(" ", dispatch_csv)
    print(" ", dispatch_rtro_csv)
    print(" ", trigger_csv)
    print(" ", trigger_archive_csv)
    print(" ", runtime_day_csv)
    print(" ", runtime_all_csv)
    print(" ", meta_json)
    print("Trigger count:", int(trigger_df["trigger"].sum()))


if __name__ == "__main__":
    main()