from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from optimization_core import (
    MicrogridParams,
    blend_recourse_step,
    build_dispatch_dataframe,
    build_runtime_table,
    build_tou_price_profile,
    build_updated_quantile_slices,
    evaluate_schedule_risk,
    load_extreme_day_inputs,
    propagate_soc,
    solve_two_stage_robust_plan,
    update_params_from_reference,
)


def run_rtro_demo(
    dataset_dir: Path,
    day: str | None,
    out_dispatch_csv: Path,
    out_trigger_csv: Path,
    out_runtime_csv: Path,
    system_id: int,
    include_baselines: bool,
    output_flag: int,
    xi_g: float | None,
    xi_c: float | None,
    delta_min: int | None,
    delta_max: int | None,
) -> Tuple[Path, Path, Path]:
    day_inputs = load_extreme_day_inputs(dataset_dir=dataset_dir, day=day)

    params = MicrogridParams()
    params = update_params_from_reference(params, day_inputs)

    if xi_g is not None:
        params.rtro_xi_g = float(xi_g)
    if xi_c is not None:
        params.rtro_xi_c = float(xi_c)
    if delta_min is not None:
        params.rtro_delta_min_steps = int(delta_min)
    if delta_max is not None:
        params.rtro_delta_max_steps = int(delta_max)

    n = len(day_inputs.timestamps)
    buy_price_day, sell_price_day = build_tou_price_profile(day_inputs.timestamps)

    # Initial full-horizon robust solve.
    current_plan = solve_two_stage_robust_plan(
        timestamps=day_inputs.timestamps,
        load_lo=day_inputs.q05,
        load_nom=day_inputs.q50,
        load_hi=day_inputs.q95,
        p_pv=day_inputs.p_pv,
        p_wt=day_inputs.p_wt,
        e0_kwh=params.soc0_kwh,
        buy_price=buy_price_day,
        sell_price=sell_price_day,
        params=params,
        output_flag=output_flag,
    )

    plan_start_idx = 0
    t_last_reopt = 0
    solve_records: List[Tuple[pd.Timestamp, float]] = [(day_inputs.timestamps[0], current_plan.solve_time_s)]

    p_buy = np.zeros(n, dtype=float)
    p_sell = np.zeros(n, dtype=float)
    p_ch = np.zeros(n, dtype=float)
    p_dis = np.zeros(n, dtype=float)
    p_dg = np.zeros(n, dtype=float)
    shed = np.zeros(n, dtype=float)
    soc_start = np.zeros(n, dtype=float)
    trigger = np.zeros(n, dtype=int)

    trigger_rows: List[Tuple[pd.Timestamp, float, float, int]] = []

    soc = float(params.soc0_kwh)
    for t in range(n):
        if t > 0:
            local_idx = t - plan_start_idx
            latest_error = float(day_inputs.load_actual[t - 1] - day_inputs.q50[t - 1])
            q05_u, q50_u, q95_u = build_updated_quantile_slices(
                day_inputs.q05,
                day_inputs.q50,
                day_inputs.q95,
                t,
                latest_error,
                params,
            )

            risk = evaluate_schedule_risk(
                plan=current_plan,
                local_idx=local_idx,
                updated_nominal_load=q50_u,
                p_pv_slice=day_inputs.p_pv[t:],
                p_wt_slice=day_inputs.p_wt[t:],
                buy_price_slice=buy_price_day[t:],
                sell_price_slice=sell_price_day[t:],
                params=params,
            )

            delta = t - t_last_reopt
            do_trigger = int(
                ((risk["psi"] > 1.0) and (delta >= params.rtro_delta_min_steps))
                or (delta >= params.rtro_delta_max_steps)
            )
            trigger[t] = do_trigger
            trigger_rows.append((day_inputs.timestamps[t], float(risk["psi"]), 1.0, do_trigger))

            if do_trigger == 1:
                current_plan = solve_two_stage_robust_plan(
                    timestamps=day_inputs.timestamps[t:],
                    load_lo=q05_u,
                    load_nom=q50_u,
                    load_hi=q95_u,
                    p_pv=day_inputs.p_pv[t:],
                    p_wt=day_inputs.p_wt[t:],
                    e0_kwh=soc,
                    buy_price=buy_price_day[t:],
                    sell_price=sell_price_day[t:],
                    params=params,
                    output_flag=output_flag,
                )
                plan_start_idx = t
                t_last_reopt = t
                solve_records.append((day_inputs.timestamps[t], current_plan.solve_time_s))

        local_exec = t - plan_start_idx
        soc_start[t] = soc
        rec = blend_recourse_step(current_plan, local_exec, realized_load=float(day_inputs.load_actual[t]))

        p_buy[t] = max(0.0, rec["P_buy"])
        p_sell[t] = max(0.0, rec["P_sell"])
        p_ch[t] = max(0.0, rec["P_ch"])
        p_dis[t] = max(0.0, rec["P_dis"])
        p_dg[t] = max(0.0, rec["P_dg"])
        shed[t] = max(0.0, rec["P_dlc"])

        soc = propagate_soc(soc, p_ch[t], p_dis[t], params)

    dispatch_df = build_dispatch_dataframe(
        timestamps=day_inputs.timestamps,
        p_buy=p_buy,
        p_sell=p_sell,
        p_ch=p_ch,
        p_dis=p_dis,
        soc_start=soc_start,
        p_dg=p_dg,
        p_pv=day_inputs.p_pv,
        p_wt=day_inputs.p_wt,
        shed=shed,
        trigger=trigger,
        q50=day_inputs.q50,
        q90=day_inputs.q90,
        q95=day_inputs.q95,
        load_actual=day_inputs.load_actual,
        params=params,
    )

    trigger_df = pd.DataFrame(trigger_rows, columns=["timestamp", "Psi", "epsilon", "trigger"])
    runtime_df = build_runtime_table(
        solve_records=solve_records,
        day_timestamps=day_inputs.timestamps,
        system_id=system_id,
        include_baselines=include_baselines,
    )

    out_dispatch_csv.parent.mkdir(parents=True, exist_ok=True)
    out_trigger_csv.parent.mkdir(parents=True, exist_ok=True)
    out_runtime_csv.parent.mkdir(parents=True, exist_ok=True)

    dispatch_df.to_csv(out_dispatch_csv, index=False)
    trigger_df.to_csv(out_trigger_csv, index=False)
    runtime_df.to_csv(out_runtime_csv, index=False)

    return out_dispatch_csv, out_trigger_csv, out_runtime_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RTRO online optimization demo using Gurobi. "
            "The script performs risk-triggered re-optimization and exports dispatch/trigger/runtime CSV files."
        )
    )
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory. Default: ./datasets/RADFL")
    parser.add_argument("--day", type=str, default=None, help="Target day in YYYY-MM-DD. Default: the latest target day")
    parser.add_argument("--out-dispatch-csv", type=str, default=None, help="Output dispatch CSV path")
    parser.add_argument("--out-trigger-csv", type=str, default=None, help="Output trigger CSV path")
    parser.add_argument("--out-runtime-csv", type=str, default=None, help="Output runtime CSV path")
    parser.add_argument("--system-id", type=int, choices=[33, 69], default=69, help="System ID label in runtime CSV")
    parser.add_argument("--no-baseline-runtime", action="store_true", help="Only write CVaR-DFL-RTRO runtime records")
    parser.add_argument("--solver-output", action="store_true", help="Enable Gurobi solver log output")
    parser.add_argument("--xi-g", type=float, default=None, help="RTRO grid-side threshold xi_g")
    parser.add_argument("--xi-c", type=float, default=None, help="RTRO cost-side threshold xi_c")
    parser.add_argument("--delta-min", type=int, default=None, help="RTRO min trigger window (steps)")
    parser.add_argument("--delta-max", type=int, default=None, help="RTRO max trigger window (steps)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else script_dir / "datasets" / "RADFL"

    default_out_dispatch = dataset_dir / "dispatch_rtro_demo.csv"
    default_out_trigger = dataset_dir / "fig_iv03_parts" / "rtro_trigger_day_demo.csv"
    default_out_runtime = dataset_dir / "fig_iv03_parts" / "solver_runtime_day_demo.csv"

    out_dispatch_csv = Path(args.out_dispatch_csv).expanduser().resolve() if args.out_dispatch_csv else default_out_dispatch
    out_trigger_csv = Path(args.out_trigger_csv).expanduser().resolve() if args.out_trigger_csv else default_out_trigger
    out_runtime_csv = Path(args.out_runtime_csv).expanduser().resolve() if args.out_runtime_csv else default_out_runtime

    p_dispatch, p_trigger, p_runtime = run_rtro_demo(
        dataset_dir=dataset_dir,
        day=args.day,
        out_dispatch_csv=out_dispatch_csv,
        out_trigger_csv=out_trigger_csv,
        out_runtime_csv=out_runtime_csv,
        system_id=args.system_id,
        include_baselines=not args.no_baseline_runtime,
        output_flag=1 if args.solver_output else 0,
        xi_g=args.xi_g,
        xi_c=args.xi_c,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
    )

    print("Saved:")
    print(" ", p_dispatch)
    print(" ", p_trigger)
    print(" ", p_runtime)


if __name__ == "__main__":
    main()
