from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from optimization_core import (
    MicrogridParams,
    blend_recourse_step,
    build_dispatch_dataframe,
    build_tou_price_profile,
    load_extreme_day_inputs,
    propagate_soc,
    solve_two_stage_robust_plan,
    update_params_from_reference,
)


def run_dispatch_demo(dataset_dir: Path, day: str | None, out_csv: Path, output_flag: int) -> Path:
    day_inputs = load_extreme_day_inputs(dataset_dir=dataset_dir, day=day)

    params = MicrogridParams()
    params = update_params_from_reference(params, day_inputs)

    buy_price, sell_price = build_tou_price_profile(day_inputs.timestamps)

    plan = solve_two_stage_robust_plan(
        timestamps=day_inputs.timestamps,
        load_lo=day_inputs.q05,
        load_nom=day_inputs.q50,
        load_hi=day_inputs.q95,
        p_pv=day_inputs.p_pv,
        p_wt=day_inputs.p_wt,
        e0_kwh=params.soc0_kwh,
        buy_price=buy_price,
        sell_price=sell_price,
        params=params,
        output_flag=output_flag,
    )

    n = len(day_inputs.timestamps)
    p_buy = np.zeros(n, dtype=float)
    p_sell = np.zeros(n, dtype=float)
    p_ch = np.zeros(n, dtype=float)
    p_dis = np.zeros(n, dtype=float)
    p_dg = np.zeros(n, dtype=float)
    shed = np.zeros(n, dtype=float)
    soc_start = np.zeros(n, dtype=float)
    trigger = np.zeros(n, dtype=int)

    soc = float(params.soc0_kwh)
    for t in range(n):
        soc_start[t] = soc
        rec = blend_recourse_step(plan=plan, step_idx=t, realized_load=float(day_inputs.load_actual[t]))

        p_buy[t] = max(0.0, rec["P_buy"])
        p_sell[t] = max(0.0, rec["P_sell"])
        p_ch[t] = max(0.0, rec["P_ch"])
        p_dis[t] = max(0.0, rec["P_dis"])
        p_dg[t] = max(0.0, rec["P_dg"])
        shed[t] = max(0.0, rec["P_dlc"])

        soc = propagate_soc(soc, p_ch[t], p_dis[t], params)

    out_df = build_dispatch_dataframe(
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TSRO dispatch demo using Gurobi. "
            "This script solves a two-stage robust dispatch once over a full day and exports dispatch CSV."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory. Default: ./datasets/RADFL",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Target day in YYYY-MM-DD. Default: the latest day in target split.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output dispatch CSV path. Default: <dataset-dir>/dispatch_tsro_demo.csv",
    )
    parser.add_argument(
        "--solver-output",
        action="store_true",
        help="Enable Gurobi solver log output.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else script_dir / "datasets" / "RADFL"
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else dataset_dir / "dispatch_tsro_demo.csv"

    saved = run_dispatch_demo(
        dataset_dir=dataset_dir,
        day=args.day,
        out_csv=out_csv,
        output_flag=1 if args.solver_output else 0,
    )
    print("Saved:", saved)


if __name__ == "__main__":
    main()
