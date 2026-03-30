from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    ConstraintList,
    NonNegativeReals,
    Objective,
    RangeSet,
    Reals,
    Set,
    Var,
    minimize,
    value,
)
from pyomo.opt import SolverFactory

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
class TSROPlanPyomo:
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


def _solver_available(name: str) -> bool:
    try:
        return bool(SolverFactory(name).available(False))
    except Exception:
        return False


def _resolve_solver_name(requested: str) -> str:
    req = requested.strip().lower()
    if req == "auto":
        for cand in ["appsi_highs", "cplex", "gurobi", "highs"]:
            if _solver_available(cand):
                return cand
        raise RuntimeError("No supported Pyomo solver is available. Install HiGHS (`highspy`) or CPLEX/Gurobi.")
    if not _solver_available(req):
        raise RuntimeError(
            f"Requested solver `{req}` is not available in current environment. "
            "Try `--solver appsi_highs` or install the requested solver first."
        )
    return req


def _solve_model(model: ConcreteModel, solver_name: str, output_flag: int, params: MicrogridParams):
    solver = SolverFactory(solver_name)

    if solver_name == "appsi_highs":
        solver.highs_options["output_flag"] = bool(output_flag)
        solver.highs_options["time_limit"] = float(params.gurobi_time_limit_s)
        solver.highs_options["mip_rel_gap"] = float(params.gurobi_mip_gap)
    elif solver_name == "highs":
        solver.options["time_limit"] = float(params.gurobi_time_limit_s)
        solver.options["mip_rel_gap"] = float(params.gurobi_mip_gap)
    elif solver_name == "gurobi":
        solver.options["TimeLimit"] = float(params.gurobi_time_limit_s)
        solver.options["MIPGap"] = float(params.gurobi_mip_gap)
    elif solver_name == "cplex":
        solver.options["timelimit"] = float(params.gurobi_time_limit_s)
        solver.options["mip_tolerances_mipgap"] = float(params.gurobi_mip_gap)

    t0 = time.perf_counter()
    results = solver.solve(model, tee=bool(output_flag))
    solve_time = time.perf_counter() - t0

    tc = None
    if hasattr(results, "solver"):
        tc = getattr(results.solver, "termination_condition", None)
    if tc is None:
        tc = getattr(results, "termination_condition", None)
    tc_str = str(tc).lower()
    if not any(k in tc_str for k in ["optimal", "feasible"]):
        raise RuntimeError(f"Solver terminated with condition `{tc}` (solver={solver_name}).")

    return results, solve_time


def solve_two_stage_robust_plan_pyomo(
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
) -> TSROPlanPyomo:
    n = len(timestamps)
    if not (len(load_lo) == len(load_nom) == len(load_hi) == len(p_pv) == len(p_wt) == len(buy_price) == len(sell_price) == n):
        raise ValueError("Input vector lengths are inconsistent.")

    load_lo = np.maximum(np.asarray(load_lo, dtype=float), 0.0)
    load_hi = np.maximum(np.asarray(load_hi, dtype=float), load_lo + 1e-3)
    load_nom = np.clip(np.maximum(np.asarray(load_nom, dtype=float), 0.0), load_lo + 1e-3, load_hi - 1e-3)
    m = ConcreteModel(name="tsro_pyomo_demo")
    m.T = RangeSet(0, n - 1)
    m.Tp1 = RangeSet(0, n)
    m.S = Set(initialize=["lo", "hi"])

    load_map = {
        "lo": np.asarray(load_lo, dtype=float),
        "hi": np.asarray(load_hi, dtype=float),
    }

    # Stage-1 variables
    m.P_buy_s = Var(m.T, domain=NonNegativeReals, bounds=(0.0, params.p_grid_max_kw))
    m.P_sell_s = Var(m.T, domain=NonNegativeReals, bounds=(0.0, params.p_grid_max_kw))
    m.P_ch_s = Var(m.T, domain=NonNegativeReals, bounds=(0.0, params.p_ess_max_kw))
    m.P_dis_s = Var(m.T, domain=NonNegativeReals, bounds=(0.0, params.p_ess_max_kw))
    m.P_dg_s = Var(m.T, domain=NonNegativeReals, bounds=(0.0, params.p_dg_max_kw))
    m.P_cur_wt_s = Var(m.T, domain=NonNegativeReals)
    m.P_cur_pv_s = Var(m.T, domain=NonNegativeReals)
    m.P_emg_s = Var(m.T, domain=NonNegativeReals)
    m.SOC_s = Var(m.Tp1, domain=Reals, bounds=(params.soc_min_kwh, params.soc_max_kwh))

    if params.use_binary_stage1:
        m.u_grid = Var(m.T, domain=Binary)
        m.u_ess = Var(m.T, domain=Binary)

    # Stage-2 variables per scenario
    m.P_buy = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.p_grid_max_kw))
    m.P_sell = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.p_grid_max_kw))
    m.P_ch = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.p_ess_max_kw))
    m.P_dis = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.p_ess_max_kw))
    m.P_dg = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.p_dg_max_kw))
    m.P_dlc = Var(m.S, m.T, domain=NonNegativeReals)
    m.P_cur_wt = Var(m.S, m.T, domain=NonNegativeReals)
    m.P_cur_pv = Var(m.S, m.T, domain=NonNegativeReals)
    m.P_emg = Var(m.S, m.T, domain=NonNegativeReals)
    m.SOC = Var(m.S, m.Tp1, domain=Reals, bounds=(params.soc_min_kwh, params.soc_max_kwh))

    m.d_buy_pos = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_grid_kw))
    m.d_buy_neg = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_grid_kw))
    m.d_sell_pos = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_grid_kw))
    m.d_sell_neg = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_grid_kw))
    m.d_ch_pos = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_ess_kw))
    m.d_ch_neg = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_ess_kw))
    m.d_dis_pos = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_ess_kw))
    m.d_dis_neg = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_ess_kw))
    m.d_dg_pos = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_dg_kw))
    m.d_dg_neg = Var(m.S, m.T, domain=NonNegativeReals, bounds=(0.0, params.recourse_dg_kw))

    m.worst_rt = Var(domain=Reals)

    m.C = ConstraintList()

    # Stage-1 constraints
    m.C.add(m.SOC_s[0] == float(e0_kwh))
    for t in range(n):
        m.C.add(m.P_cur_wt_s[t] <= float(p_wt[t]))
        m.C.add(m.P_cur_pv_s[t] <= float(p_pv[t]))

        if params.use_binary_stage1:
            m.C.add(m.P_buy_s[t] <= params.p_grid_max_kw * m.u_grid[t])
            m.C.add(m.P_sell_s[t] <= params.p_grid_max_kw * (1.0 - m.u_grid[t]))
            m.C.add(m.P_ch_s[t] <= params.p_ess_max_kw * m.u_ess[t])
            m.C.add(m.P_dis_s[t] <= params.p_ess_max_kw * (1.0 - m.u_ess[t]))

        m.C.add(
            float(p_wt[t]) - m.P_cur_wt_s[t]
            + float(p_pv[t]) - m.P_cur_pv_s[t]
            + m.P_dis_s[t] + m.P_buy_s[t] + m.P_dg_s[t] + m.P_emg_s[t]
            == float(load_nom[t]) + m.P_ch_s[t] + m.P_sell_s[t]
        )
        m.C.add(
            m.SOC_s[t + 1]
            == m.SOC_s[t]
            + params.eta_ch * m.P_ch_s[t] * params.dt_hours
            - (m.P_dis_s[t] / max(1e-8, params.eta_dis)) * params.dt_hours
        )

    # Stage-2 constraints
    for s in ["lo", "hi"]:
        m.C.add(m.SOC[s, 0] == float(e0_kwh))
        for t in range(n):
            m.C.add(m.P_buy[s, t] - m.P_buy_s[t] == m.d_buy_pos[s, t] - m.d_buy_neg[s, t])
            m.C.add(m.P_sell[s, t] - m.P_sell_s[t] == m.d_sell_pos[s, t] - m.d_sell_neg[s, t])
            m.C.add(m.P_ch[s, t] - m.P_ch_s[t] == m.d_ch_pos[s, t] - m.d_ch_neg[s, t])
            m.C.add(m.P_dis[s, t] - m.P_dis_s[t] == m.d_dis_pos[s, t] - m.d_dis_neg[s, t])
            m.C.add(m.P_dg[s, t] - m.P_dg_s[t] == m.d_dg_pos[s, t] - m.d_dg_neg[s, t])

            m.C.add(m.P_cur_wt[s, t] <= float(p_wt[t]))
            m.C.add(m.P_cur_pv[s, t] <= float(p_pv[t]))
            m.C.add(m.P_dlc[s, t] <= params.dlc_ratio * float(load_map[s][t]))

            m.C.add(
                float(p_wt[t]) - m.P_cur_wt[s, t]
                + float(p_pv[t]) - m.P_cur_pv[s, t]
                + m.P_dis[s, t] + m.P_buy[s, t] + m.P_dg[s, t] + m.P_emg[s, t]
                == float(load_map[s][t]) - m.P_dlc[s, t] + m.P_ch[s, t] + m.P_sell[s, t]
            )
            m.C.add(
                m.SOC[s, t + 1]
                == m.SOC[s, t]
                + params.eta_ch * m.P_ch[s, t] * params.dt_hours
                - (m.P_dis[s, t] / max(1e-8, params.eta_dis)) * params.dt_hours
            )

    def _rt_cost_expr(s: str):
        return sum(
            (
                params.beta_buy_pos * m.d_buy_pos[s, t]
                + params.beta_buy_neg * m.d_buy_neg[s, t]
                + params.beta_sell_pos * m.d_sell_pos[s, t]
                + params.beta_sell_neg * m.d_sell_neg[s, t]
                + params.beta_ch_pos * m.d_ch_pos[s, t]
                + params.beta_ch_neg * m.d_ch_neg[s, t]
                + params.beta_dis_pos * m.d_dis_pos[s, t]
                + params.beta_dis_neg * m.d_dis_neg[s, t]
                + params.beta_dg_pos * m.d_dg_pos[s, t]
                + params.beta_dg_neg * m.d_dg_neg[s, t]
                + params.c_dlc * m.P_dlc[s, t]
                + params.c_cur * (m.P_cur_wt[s, t] + m.P_cur_pv[s, t])
                + params.c_dg_rt * m.P_dg[s, t]
                + 10.0 * m.P_emg[s, t]
            )
            * params.dt_hours
            for t in range(n)
        )

    for s in ["lo", "hi"]:
        m.C.add(m.worst_rt >= _rt_cost_expr(s))

    day_ahead_cost = sum(
        (
            float(buy_price[t]) * m.P_buy_s[t]
            - float(sell_price[t]) * m.P_sell_s[t]
            + params.c_ch * m.P_ch_s[t]
            + params.c_dis * m.P_dis_s[t]
            + params.c_dg_da * m.P_dg_s[t]
            + 10.0 * m.P_emg_s[t]
        )
        * params.dt_hours
        for t in range(n)
    )

    m.obj = Objective(expr=day_ahead_cost + m.worst_rt, sense=minimize)

    _, solve_time = _solve_model(m, solver_name=solver_name, output_flag=output_flag, params=params)

    p_buy_sch = np.array([value(m.P_buy_s[t]) for t in range(n)], dtype=float)
    p_sell_sch = np.array([value(m.P_sell_s[t]) for t in range(n)], dtype=float)
    p_ch_sch = np.array([value(m.P_ch_s[t]) for t in range(n)], dtype=float)
    p_dis_sch = np.array([value(m.P_dis_s[t]) for t in range(n)], dtype=float)
    p_dg_sch = np.array([value(m.P_dg_s[t]) for t in range(n)], dtype=float)
    soc_sch = np.array([value(m.SOC_s[t]) for t in range(n + 1)], dtype=float)

    scenario = {}
    for s in ["lo", "hi"]:
        scenario[s] = {
            "P_buy": np.array([value(m.P_buy[s, t]) for t in range(n)], dtype=float),
            "P_sell": np.array([value(m.P_sell[s, t]) for t in range(n)], dtype=float),
            "P_ch": np.array([value(m.P_ch[s, t]) for t in range(n)], dtype=float),
            "P_dis": np.array([value(m.P_dis[s, t]) for t in range(n)], dtype=float),
            "P_dg": np.array([value(m.P_dg[s, t]) for t in range(n)], dtype=float),
            "P_dlc": np.array([value(m.P_dlc[s, t]) for t in range(n)], dtype=float),
            "P_cur_wt": np.array([value(m.P_cur_wt[s, t]) for t in range(n)], dtype=float),
            "P_cur_pv": np.array([value(m.P_cur_pv[s, t]) for t in range(n)], dtype=float),
            "P_emg": np.array([value(m.P_emg[s, t]) for t in range(n)], dtype=float),
            "SOC": np.array([value(m.SOC[s, t]) for t in range(n + 1)], dtype=float),
        }

    return TSROPlanPyomo(
        timestamps=timestamps,
        load_lo=np.asarray(load_lo, dtype=float),
        load_nom=np.asarray(load_nom, dtype=float),
        load_hi=np.asarray(load_hi, dtype=float),
        p_buy_sch=p_buy_sch,
        p_sell_sch=p_sell_sch,
        p_ch_sch=p_ch_sch,
        p_dis_sch=p_dis_sch,
        p_dg_sch=p_dg_sch,
        soc_sch=soc_sch,
        scenario=scenario,
        objective_value=float(value(m.obj)),
        solve_time_s=float(solve_time),
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
    plan = solve_two_stage_robust_plan_pyomo(
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

    current_plan = solve_two_stage_robust_plan_pyomo(
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
            do_trigger = 0
            trigger[t] = do_trigger
            trigger_rows.append((day_inputs.timestamps[t], float(risk["psi"]), 1.0, do_trigger))

            if do_trigger == 1:
                current_plan = solve_two_stage_robust_plan_pyomo(
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
        description="Pyomo optimization demo for TSRO and RTRO (supports appsi_highs/cplex/gurobi)."
    )
    parser.add_argument("--mode", type=str, choices=["dispatch", "rtro"], default="rtro")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset directory. Default: ./datasets/RADFL")
    parser.add_argument("--day", type=str, default=None, help="Target day in YYYY-MM-DD")
    parser.add_argument("--solver", type=str, default="auto", help="Solver name: auto/appsi_highs/cplex/gurobi/highs")
    parser.add_argument("--solver-output", action="store_true", help="Enable solver logs")
    parser.add_argument("--use-binary-stage1", action="store_true", help="Use binary exclusiveness variables in stage-1")

    parser.add_argument("--out-dispatch-csv", type=str, default=None)
    parser.add_argument("--out-trigger-csv", type=str, default=None)
    parser.add_argument("--out-runtime-csv", type=str, default=None)

    parser.add_argument("--system-id", type=int, choices=[33, 69], default=69)
    parser.add_argument("--no-baseline-runtime", action="store_true")

    args = parser.parse_args()

    solver_name = _resolve_solver_name(args.solver)

    script_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else script_dir / "datasets" / "RADFL"

    if args.mode == "dispatch":
        out_dispatch_csv = (
            Path(args.out_dispatch_csv).expanduser().resolve()
            if args.out_dispatch_csv
            else dataset_dir / "dispatch_pyomo_demo.csv"
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
            else dataset_dir / "dispatch_pyomo_rtro_demo.csv"
        )
        out_trigger_csv = (
            Path(args.out_trigger_csv).expanduser().resolve()
            if args.out_trigger_csv
            else dataset_dir / "fig_iv03_parts" / "rtro_trigger_day_pyomo_demo.csv"
        )
        out_runtime_csv = (
            Path(args.out_runtime_csv).expanduser().resolve()
            if args.out_runtime_csv
            else dataset_dir / "fig_iv03_parts" / "solver_runtime_day_pyomo_demo.csv"
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






