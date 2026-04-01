from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None

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
    update_params_from_reference,
)


@dataclass
class TSROPlanCVXPY:
    timestamps: pd.DatetimeIndex
    load_lo: np.ndarray
    load_nom: np.ndarray
    load_hi: np.ndarray
    p_buy_sch: np.ndarray
    p_sell_sch: np.ndarray
    p_ch_sch: np.ndarray
    p_dis_sch: np.ndarray
    p_dg_sch: np.ndarray
    soc_sch: np.ndarray
    scenario: Dict[str, Dict[str, np.ndarray]]
    objective_value: float
    solve_time_s: float


_AUTO_CONTINUOUS_SOLVERS = ("CLARABEL", "ECOS", "SCS", "OSQP", "SCIPY", "GUROBI")
_AUTO_MIP_SOLVERS = ("ECOS_BB", "GUROBI")
_SUPPORTED_MIP_SOLVERS = frozenset(_AUTO_MIP_SOLVERS)


def _require_cvxpy() -> None:
    if cp is None:
        raise RuntimeError("cvxpy is not available. Install it first, e.g. `pip install cvxpy ecos`.")


def _installed_cvxpy_solvers() -> set[str]:
    _require_cvxpy()
    return {str(name).upper() for name in cp.installed_solvers()}


def _resolve_solver_name(requested: str, requires_mip: bool) -> str:
    installed = _installed_cvxpy_solvers()
    req = requested.strip().upper()

    if req == "AUTO":
        candidates = _AUTO_MIP_SOLVERS if requires_mip else _AUTO_CONTINUOUS_SOLVERS
        for cand in candidates:
            if cand in installed:
                return cand
        need = "mixed-integer" if requires_mip else "continuous"
        raise RuntimeError(
            f"No supported {need} CVXPY solver is available. Installed solvers: {sorted(installed)}."
        )

    if req not in installed:
        raise RuntimeError(
            f"Requested CVXPY solver `{req}` is not available in current environment. "
            f"Installed solvers: {sorted(installed)}."
        )
    if requires_mip and req not in _SUPPORTED_MIP_SOLVERS:
        raise RuntimeError(
            f"Solver `{req}` is not used for binary stage-1 variables in this demo. "
            f"Use one of {sorted(_SUPPORTED_MIP_SOLVERS)} or rerun without `--use-binary-stage1`."
        )
    return req


def _solve_problem(problem: "cp.Problem", solver_name: str, output_flag: int) -> Tuple[float, float]:
    solve_kwargs = {
        "solver": solver_name,
        "verbose": bool(output_flag),
        "warm_start": True,
    }

    t0 = time.perf_counter()
    objective_value = problem.solve(**solve_kwargs)
    solve_time = time.perf_counter() - t0

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY solve failed with status `{problem.status}` (solver={solver_name}).")

    if objective_value is None:
        objective_value = problem.value
    if objective_value is None:
        raise RuntimeError(f"CVXPY solve finished without an objective value (solver={solver_name}).")

    return float(objective_value), float(solve_time)


def _value_to_numpy(var: "cp.Variable", expected_len: int) -> np.ndarray:
    if var.value is None:
        raise RuntimeError("CVXPY variable has no value after solve.")
    arr = np.asarray(var.value, dtype=float).reshape(-1)
    if arr.size != expected_len:
        raise RuntimeError(f"Unexpected variable size {arr.size}; expected {expected_len}.")
    return arr


def solve_two_stage_robust_plan_cvxpy(
    timestamps: pd.DatetimeIndex,
    load_lo: np.ndarray,
    load_nom: np.ndarray,
    load_hi: np.ndarray,
    p_pv: np.ndarray,
    p_wt: np.ndarray,
    e0_kwh: float,
    buy_price: np.ndarray,
    sell_price: np.ndarray,
    params: MicrogridParams,
    solver_name: str,
    output_flag: int,
) -> TSROPlanCVXPY:
    _require_cvxpy()

    n = len(timestamps)
    if not (len(load_lo) == len(load_nom) == len(load_hi) == len(p_pv) == len(p_wt) == len(buy_price) == len(sell_price) == n):
        raise ValueError("Input vector lengths are inconsistent.")

    load_lo = np.maximum(np.asarray(load_lo, dtype=float), 0.0)
    load_hi = np.maximum(np.asarray(load_hi, dtype=float), load_lo + 1e-3)
    load_nom = np.clip(np.maximum(np.asarray(load_nom, dtype=float), 0.0), load_lo + 1e-3, load_hi - 1e-3)
    p_pv = np.asarray(p_pv, dtype=float)
    p_wt = np.asarray(p_wt, dtype=float)
    buy_price = np.asarray(buy_price, dtype=float)
    sell_price = np.asarray(sell_price, dtype=float)

    P_buy_s = cp.Variable(n, nonneg=True)
    P_sell_s = cp.Variable(n, nonneg=True)
    P_ch_s = cp.Variable(n, nonneg=True)
    P_dis_s = cp.Variable(n, nonneg=True)
    P_dg_s = cp.Variable(n, nonneg=True)
    P_cur_wt_s = cp.Variable(n, nonneg=True)
    P_cur_pv_s = cp.Variable(n, nonneg=True)
    P_emg_s = cp.Variable(n, nonneg=True)
    SOC_s = cp.Variable(n + 1)

    if params.use_binary_stage1:
        u_grid = cp.Variable(n, boolean=True)
        u_ess = cp.Variable(n, boolean=True)
    else:
        u_grid = None
        u_ess = None

    constraints = [
        P_buy_s <= params.p_grid_max_kw,
        P_sell_s <= params.p_grid_max_kw,
        P_ch_s <= params.p_ess_max_kw,
        P_dis_s <= params.p_ess_max_kw,
        P_dg_s <= params.p_dg_max_kw,
        P_cur_wt_s <= p_wt,
        P_cur_pv_s <= p_pv,
        SOC_s >= params.soc_min_kwh,
        SOC_s <= params.soc_max_kwh,
        SOC_s[0] == float(e0_kwh),
        SOC_s[1:]
        == SOC_s[:-1]
        + params.eta_ch * P_ch_s * params.dt_hours
        - (P_dis_s / max(1e-8, params.eta_dis)) * params.dt_hours,
        p_wt - P_cur_wt_s + p_pv - P_cur_pv_s + P_dis_s + P_buy_s + P_dg_s + P_emg_s
        == load_nom + P_ch_s + P_sell_s,
    ]

    if params.use_binary_stage1:
        constraints.extend(
            [
                P_buy_s <= params.p_grid_max_kw * u_grid,
                P_sell_s <= params.p_grid_max_kw * (1.0 - u_grid),
                P_ch_s <= params.p_ess_max_kw * u_ess,
                P_dis_s <= params.p_ess_max_kw * (1.0 - u_ess),
            ]
        )

    scenario = {}
    scenario_vars: Dict[str, Dict[str, "cp.Variable"]] = {}
    scenario_load = {
        "lo": np.asarray(load_lo, dtype=float),
        "hi": np.asarray(load_hi, dtype=float),
    }

    worst_rt = cp.Variable()

    for s_name, s_load in scenario_load.items():
        P_buy = cp.Variable(n, nonneg=True)
        P_sell = cp.Variable(n, nonneg=True)
        P_ch = cp.Variable(n, nonneg=True)
        P_dis = cp.Variable(n, nonneg=True)
        P_dg = cp.Variable(n, nonneg=True)
        P_dlc = cp.Variable(n, nonneg=True)
        P_cur_wt = cp.Variable(n, nonneg=True)
        P_cur_pv = cp.Variable(n, nonneg=True)
        P_emg = cp.Variable(n, nonneg=True)
        SOC = cp.Variable(n + 1)

        d_buy_pos = cp.Variable(n, nonneg=True)
        d_buy_neg = cp.Variable(n, nonneg=True)
        d_sell_pos = cp.Variable(n, nonneg=True)
        d_sell_neg = cp.Variable(n, nonneg=True)
        d_ch_pos = cp.Variable(n, nonneg=True)
        d_ch_neg = cp.Variable(n, nonneg=True)
        d_dis_pos = cp.Variable(n, nonneg=True)
        d_dis_neg = cp.Variable(n, nonneg=True)
        d_dg_pos = cp.Variable(n, nonneg=True)
        d_dg_neg = cp.Variable(n, nonneg=True)

        constraints.extend(
            [
                P_buy <= params.p_grid_max_kw,
                P_sell <= params.p_grid_max_kw,
                P_ch <= params.p_ess_max_kw,
                P_dis <= params.p_ess_max_kw,
                P_dg <= params.p_dg_max_kw,
                P_cur_wt <= p_wt,
                P_cur_pv <= p_pv,
                P_dlc <= params.dlc_ratio * s_load,
                SOC >= params.soc_min_kwh,
                SOC <= params.soc_max_kwh,
                d_buy_pos <= params.recourse_grid_kw,
                d_buy_neg <= params.recourse_grid_kw,
                d_sell_pos <= params.recourse_grid_kw,
                d_sell_neg <= params.recourse_grid_kw,
                d_ch_pos <= params.recourse_ess_kw,
                d_ch_neg <= params.recourse_ess_kw,
                d_dis_pos <= params.recourse_ess_kw,
                d_dis_neg <= params.recourse_ess_kw,
                d_dg_pos <= params.recourse_dg_kw,
                d_dg_neg <= params.recourse_dg_kw,
                P_buy - P_buy_s == d_buy_pos - d_buy_neg,
                P_sell - P_sell_s == d_sell_pos - d_sell_neg,
                P_ch - P_ch_s == d_ch_pos - d_ch_neg,
                P_dis - P_dis_s == d_dis_pos - d_dis_neg,
                P_dg - P_dg_s == d_dg_pos - d_dg_neg,
                SOC[0] == float(e0_kwh),
                SOC[1:]
                == SOC[:-1]
                + params.eta_ch * P_ch * params.dt_hours
                - (P_dis / max(1e-8, params.eta_dis)) * params.dt_hours,
                p_wt - P_cur_wt + p_pv - P_cur_pv + P_dis + P_buy + P_dg + P_emg
                == s_load - P_dlc + P_ch + P_sell,
            ]
        )

        rt_cost = params.dt_hours * cp.sum(
            params.beta_buy_pos * d_buy_pos
            + params.beta_buy_neg * d_buy_neg
            + params.beta_sell_pos * d_sell_pos
            + params.beta_sell_neg * d_sell_neg
            + params.beta_ch_pos * d_ch_pos
            + params.beta_ch_neg * d_ch_neg
            + params.beta_dis_pos * d_dis_pos
            + params.beta_dis_neg * d_dis_neg
            + params.beta_dg_pos * d_dg_pos
            + params.beta_dg_neg * d_dg_neg
            + params.c_dlc * P_dlc
            + params.c_cur * (P_cur_wt + P_cur_pv)
            + params.c_dg_rt * P_dg
            + 10.0 * P_emg
        )
        constraints.append(worst_rt >= rt_cost)

        scenario_vars[s_name] = {
            "P_buy": P_buy,
            "P_sell": P_sell,
            "P_ch": P_ch,
            "P_dis": P_dis,
            "P_dg": P_dg,
            "P_dlc": P_dlc,
            "P_cur_wt": P_cur_wt,
            "P_cur_pv": P_cur_pv,
            "P_emg": P_emg,
            "SOC": SOC,
        }

    day_ahead_cost = params.dt_hours * cp.sum(
        cp.multiply(buy_price, P_buy_s)
        - cp.multiply(sell_price, P_sell_s)
        + params.c_ch * P_ch_s
        + params.c_dis * P_dis_s
        + params.c_dg_da * P_dg_s
        + 10.0 * P_emg_s
    )

    problem = cp.Problem(cp.Minimize(day_ahead_cost + worst_rt), constraints)
    objective_value, solve_time = _solve_problem(problem, solver_name=solver_name, output_flag=output_flag)

    for s_name, sv in scenario_vars.items():
        scenario[s_name] = {
            "P_buy": _value_to_numpy(sv["P_buy"], n),
            "P_sell": _value_to_numpy(sv["P_sell"], n),
            "P_ch": _value_to_numpy(sv["P_ch"], n),
            "P_dis": _value_to_numpy(sv["P_dis"], n),
            "P_dg": _value_to_numpy(sv["P_dg"], n),
            "P_dlc": _value_to_numpy(sv["P_dlc"], n),
            "P_cur_wt": _value_to_numpy(sv["P_cur_wt"], n),
            "P_cur_pv": _value_to_numpy(sv["P_cur_pv"], n),
            "P_emg": _value_to_numpy(sv["P_emg"], n),
            "SOC": _value_to_numpy(sv["SOC"], n + 1),
        }

    return TSROPlanCVXPY(
        timestamps=timestamps,
        load_lo=np.asarray(load_lo, dtype=float),
        load_nom=np.asarray(load_nom, dtype=float),
        load_hi=np.asarray(load_hi, dtype=float),
        p_buy_sch=_value_to_numpy(P_buy_s, n),
        p_sell_sch=_value_to_numpy(P_sell_s, n),
        p_ch_sch=_value_to_numpy(P_ch_s, n),
        p_dis_sch=_value_to_numpy(P_dis_s, n),
        p_dg_sch=_value_to_numpy(P_dg_s, n),
        soc_sch=_value_to_numpy(SOC_s, n + 1),
        scenario=scenario,
        objective_value=objective_value,
        solve_time_s=solve_time,
    )


def run_dispatch(
    dataset_dir: Path,
    day: str | None,
    out_dispatch_csv: Path,
    solver_name: str,
    output_flag: int,
    use_binary_stage1: bool,
) -> Path:
    day_inputs = load_extreme_day_inputs(dataset_dir=dataset_dir, day=day)
    params = update_params_from_reference(MicrogridParams(), day_inputs)
    params.use_binary_stage1 = bool(use_binary_stage1)
    params.recourse_grid_kw = max(params.recourse_grid_kw, params.p_grid_max_kw)
    params.recourse_ess_kw = max(params.recourse_ess_kw, params.p_ess_max_kw)
    params.recourse_dg_kw = max(params.recourse_dg_kw, params.p_dg_max_kw)

    buy_price, sell_price = build_tou_price_profile(day_inputs.timestamps)
    plan = solve_two_stage_robust_plan_cvxpy(
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
        solver_name=solver_name,
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

    out_dispatch_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dispatch_csv, index=False)
    return out_dispatch_csv


def run_rtro(
    dataset_dir: Path,
    day: str | None,
    out_dispatch_csv: Path,
    out_trigger_csv: Path,
    out_runtime_csv: Path,
    solver_name: str,
    output_flag: int,
    use_binary_stage1: bool,
    system_id: int,
    include_baselines: bool,
) -> Tuple[Path, Path, Path]:
    day_inputs = load_extreme_day_inputs(dataset_dir=dataset_dir, day=day)
    params = update_params_from_reference(MicrogridParams(), day_inputs)
    params.use_binary_stage1 = bool(use_binary_stage1)
    params.recourse_grid_kw = max(params.recourse_grid_kw, params.p_grid_max_kw)
    params.recourse_ess_kw = max(params.recourse_ess_kw, params.p_ess_max_kw)
    params.recourse_dg_kw = max(params.recourse_dg_kw, params.p_dg_max_kw)

    n = len(day_inputs.timestamps)
    buy_price_day, sell_price_day = build_tou_price_profile(day_inputs.timestamps)

    current_plan = solve_two_stage_robust_plan_cvxpy(
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
        solver_name=solver_name,
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
                delta >= params.rtro_delta_min_steps
                and (risk["psi"] >= 1.0 or delta >= params.rtro_delta_max_steps)
            )
            trigger[t] = do_trigger
            trigger_rows.append((day_inputs.timestamps[t], float(risk["psi"]), 1.0, do_trigger))

            if do_trigger == 1:
                current_plan = solve_two_stage_robust_plan_cvxpy(
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
                    solver_name=solver_name,
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
        description="CVXPY optimization demo for TSRO and RTRO (auto-selects CLARABEL/ECOS_BB/GUROBI)."
    )
    parser.add_argument("--mode", type=str, choices=["dispatch", "rtro"], default="rtro")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory. Default: ./datasets/RADFL")
    parser.add_argument("--day", type=str, default=None, help="Target day in YYYY-MM-DD")
    parser.add_argument("--solver", type=str, default="auto", help="CVXPY solver name: auto/CLARABEL/ECOS/SCS/OSQP/SCIPY/ECOS_BB/GUROBI")
    parser.add_argument("--solver-output", action="store_true", help="Enable solver logs")
    parser.add_argument("--use-binary-stage1", action="store_true", help="Use binary exclusiveness variables in stage-1")

    parser.add_argument("--out-dispatch-csv", type=str, default=None)
    parser.add_argument("--out-trigger-csv", type=str, default=None)
    parser.add_argument("--out-runtime-csv", type=str, default=None)

    parser.add_argument("--system-id", type=int, choices=[33, 69], default=69)
    parser.add_argument("--no-baseline-runtime", action="store_true")

    args = parser.parse_args()
    solver_name = _resolve_solver_name(args.solver, requires_mip=bool(args.use_binary_stage1))

    script_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else script_dir / "datasets" / "RADFL"

    if args.mode == "dispatch":
        out_dispatch_csv = (
            Path(args.out_dispatch_csv).expanduser().resolve()
            if args.out_dispatch_csv
            else dataset_dir / "dispatch_cvxpy_demo.csv"
        )
        saved = run_dispatch(
            dataset_dir=dataset_dir,
            day=args.day,
            out_dispatch_csv=out_dispatch_csv,
            solver_name=solver_name,
            output_flag=1 if args.solver_output else 0,
            use_binary_stage1=args.use_binary_stage1,
        )
        print("Solver:", solver_name)
        print("Saved:", saved)
    else:
        out_dispatch_csv = (
            Path(args.out_dispatch_csv).expanduser().resolve()
            if args.out_dispatch_csv
            else dataset_dir / "dispatch_cvxpy_rtro_demo.csv"
        )
        out_trigger_csv = (
            Path(args.out_trigger_csv).expanduser().resolve()
            if args.out_trigger_csv
            else dataset_dir / "fig_iv03_parts" / "rtro_trigger_day_cvxpy_demo.csv"
        )
        out_runtime_csv = (
            Path(args.out_runtime_csv).expanduser().resolve()
            if args.out_runtime_csv
            else dataset_dir / "fig_iv03_parts" / "solver_runtime_day_cvxpy_demo.csv"
        )

        p_dispatch, p_trigger, p_runtime = run_rtro(
            dataset_dir=dataset_dir,
            day=args.day,
            out_dispatch_csv=out_dispatch_csv,
            out_trigger_csv=out_trigger_csv,
            out_runtime_csv=out_runtime_csv,
            solver_name=solver_name,
            output_flag=1 if args.solver_output else 0,
            use_binary_stage1=args.use_binary_stage1,
            system_id=args.system_id,
            include_baselines=not args.no_baseline_runtime,
        )
        print("Solver:", solver_name)
        print("Saved:")
        print(" ", p_dispatch)
        print(" ", p_trigger)
        print(" ", p_runtime)


if __name__ == "__main__":
    main()