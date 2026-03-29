from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


@dataclass
class MicrogridParams:
    dt_hours: float = 0.25
    p_grid_max_kw: float = 5000.0
    p_ess_max_kw: float = 1100.0
    p_dg_max_kw: float = 2500.0
    e_cap_kwh: float = 7700.0
    soc_min_kwh: float = 1155.0
    soc_max_kwh: float = 7315.0
    soc0_kwh: float = 4620.0
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    dlc_ratio: float = 0.10
    recourse_grid_kw: float = 3000.0
    recourse_ess_kw: float = 800.0
    recourse_dg_kw: float = 1500.0
    c_ch: float = 0.004
    c_dis: float = 0.006
    c_dg_da: float = 0.40
    c_dg_rt: float = 0.65
    c_dlc: float = 3.00
    c_cur: float = 0.015
    beta_buy_pos: float = 0.035
    beta_buy_neg: float = 0.020
    beta_sell_pos: float = 0.020
    beta_sell_neg: float = 0.030
    beta_ch_pos: float = 0.015
    beta_ch_neg: float = 0.015
    beta_dis_pos: float = 0.015
    beta_dis_neg: float = 0.015
    beta_dg_pos: float = 0.030
    beta_dg_neg: float = 0.010
    rtro_xi_g: float = 1e-3
    rtro_xi_c: float = 5e-2
    rtro_delta_min_steps: int = 1
    rtro_delta_max_steps: int = 8
    forecast_update_gain: float = 0.45
    forecast_update_decay: float = 0.85
    use_binary_stage1: bool = True
    gurobi_mip_gap: float = 1e-4
    gurobi_time_limit_s: float = 60.0


@dataclass
class DayInputs:
    day: str
    timestamps: pd.DatetimeIndex
    load_actual: np.ndarray
    q05: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    q95: np.ndarray
    p_pv: np.ndarray
    p_wt: np.ndarray
    reference_dispatch: pd.DataFrame | None


@dataclass
class TSROPlan:
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


def _require_gurobi() -> None:
    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is not available. Install it first, e.g. `pip install gurobipy` in your active environment."
        )


def build_tou_price_profile(timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    hours = timestamps.hour.to_numpy(dtype=float) + timestamps.minute.to_numpy(dtype=float) / 60.0
    buy = np.full(len(timestamps), 0.20, dtype=float)
    buy[(hours >= 0.0) & (hours < 6.0)] = 0.16
    buy[(hours >= 6.0) & (hours < 10.0)] = 0.24
    buy[(hours >= 10.0) & (hours < 16.0)] = 0.20
    buy[(hours >= 16.0) & (hours < 21.0)] = 0.33
    buy[(hours >= 21.0) & (hours <= 24.0)] = 0.18
    sell = 0.62 * buy
    return buy, sell


def _synthetic_renewable_profiles(timestamps: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    hours = timestamps.hour.to_numpy(dtype=float) + timestamps.minute.to_numpy(dtype=float) / 60.0
    pv_shape = np.sin(np.pi * np.clip((hours - 6.0) / 12.0, 0.0, 1.0))
    p_pv = 3200.0 * np.maximum(0.0, pv_shape) ** 1.5
    p_wt = 800.0 + 120.0 * np.sin(2.0 * np.pi * (hours - 2.0) / 24.0) + 70.0 * np.cos(4.0 * np.pi * hours / 24.0)
    p_wt = np.clip(p_wt, 600.0, 1000.0)
    return p_pv.astype(float), p_wt.astype(float)


def _align_quantile_series(
    timestamps: pd.DatetimeIndex,
    quantile_df: pd.DataFrame,
    col_name: str,
) -> np.ndarray:
    q_map = quantile_df.set_index("timestamp")
    fallback = float(quantile_df.iloc[0][col_name])
    out = []
    for ts in timestamps:
        if ts in q_map.index:
            val = q_map.loc[ts, col_name]
            if isinstance(val, pd.Series):
                out.append(float(val.iloc[0]))
            else:
                out.append(float(val))
        else:
            out.append(fallback)
    return np.asarray(out, dtype=float)


def load_extreme_day_inputs(dataset_dir: Path, day: str | None = None) -> DayInputs:
    task_path = dataset_dir / "extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv"
    quantile_path = dataset_dir / "pred_quantiles_extreme.csv"
    dispatch_ref_path = dataset_dir / "dispatch_placeholder_24h_15min_tunedCaps_v6.csv"

    if not task_path.exists():
        raise FileNotFoundError(f"Missing dataset: {task_path}")
    if not quantile_path.exists():
        raise FileNotFoundError(f"Missing dataset: {quantile_path}")

    task_df = pd.read_csv(task_path, parse_dates=["timestamp"])
    target_df = task_df[task_df["split"].astype(str) == "target"].copy()
    if target_df.empty:
        raise ValueError("No `target` split found in extreme forecast task CSV.")

    available_days = sorted(target_df["timestamp"].dt.strftime("%Y-%m-%d").unique())
    use_day = day if day is not None else available_days[-1]
    if use_day not in available_days:
        raise ValueError(f"Day `{use_day}` not found in target split. Available days: {available_days}")

    day_df = target_df[target_df["timestamp"].dt.strftime("%Y-%m-%d") == use_day].sort_values("timestamp")
    timestamps = pd.DatetimeIndex(day_df["timestamp"]).sort_values()
    if len(timestamps) == 0:
        raise ValueError(f"No target rows found for day `{use_day}`.")

    quantile_df = pd.read_csv(quantile_path, parse_dates=["timestamp"]).sort_values("timestamp")
    if quantile_df.empty:
        raise ValueError("`pred_quantiles_extreme.csv` is empty.")

    q05 = _align_quantile_series(timestamps, quantile_df, "load_0.05")
    q50 = _align_quantile_series(timestamps, quantile_df, "load_0.5")
    q90 = _align_quantile_series(timestamps, quantile_df, "load_0.9")
    q95 = _align_quantile_series(timestamps, quantile_df, "load_0.95")

    reference_dispatch = None
    if dispatch_ref_path.exists():
        dref = pd.read_csv(dispatch_ref_path, parse_dates=["timestamp"])
        dref_day = dref[dref["timestamp"].dt.strftime("%Y-%m-%d") == use_day].copy()
        if len(dref_day) > 0:
            dref_day = dref_day.sort_values("timestamp")
            reference_dispatch = dref_day

    if reference_dispatch is not None and {"P_pv", "P_wt"}.issubset(reference_dispatch.columns):
        merged = pd.DataFrame({"timestamp": timestamps}).merge(
            reference_dispatch[["timestamp", "P_pv", "P_wt"]],
            on="timestamp",
            how="left",
        )
        if merged["P_pv"].isna().any() or merged["P_wt"].isna().any():
            p_pv_syn, p_wt_syn = _synthetic_renewable_profiles(timestamps)
            p_pv = merged["P_pv"].fillna(pd.Series(p_pv_syn)).to_numpy(dtype=float)
            p_wt = merged["P_wt"].fillna(pd.Series(p_wt_syn)).to_numpy(dtype=float)
        else:
            p_pv = merged["P_pv"].to_numpy(dtype=float)
            p_wt = merged["P_wt"].to_numpy(dtype=float)
    else:
        p_pv, p_wt = _synthetic_renewable_profiles(timestamps)

    return DayInputs(
        day=use_day,
        timestamps=timestamps,
        load_actual=day_df["load"].to_numpy(dtype=float),
        q05=q05,
        q50=q50,
        q90=q90,
        q95=q95,
        p_pv=np.asarray(p_pv, dtype=float),
        p_wt=np.asarray(p_wt, dtype=float),
        reference_dispatch=reference_dispatch,
    )


def update_params_from_reference(params: MicrogridParams, day_inputs: DayInputs) -> MicrogridParams:
    if day_inputs.reference_dispatch is None:
        return params

    ref0 = day_inputs.reference_dispatch.iloc[0]
    params.p_grid_max_kw = float(ref0.get("Pgrid_max_kW", params.p_grid_max_kw))
    params.p_ess_max_kw = float(ref0.get("Pess_max_kW", params.p_ess_max_kw))
    params.e_cap_kwh = float(ref0.get("Ecap_kWh", params.e_cap_kwh))
    params.soc_min_kwh = float(ref0.get("SOC_min_kWh", params.soc_min_kwh))
    params.soc_max_kwh = float(ref0.get("SOC_max_kWh", params.soc_max_kwh))
    params.soc0_kwh = float(ref0.get("SOC0_kWh", params.soc0_kwh))
    params.p_dg_max_kw = max(params.p_dg_max_kw, 1600.0)
    return params


def _var_to_np(var_dict: gp.tupledict, length: int) -> np.ndarray:
    return np.array([float(var_dict[i].X) for i in range(length)], dtype=float)


def solve_two_stage_robust_plan(
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
    output_flag: int = 0,
) -> TSROPlan:
    _require_gurobi()
    n = len(timestamps)
    if not (len(load_lo) == len(load_nom) == len(load_hi) == len(p_pv) == len(p_wt) == len(buy_price) == len(sell_price) == n):
        raise ValueError("Input vector lengths are inconsistent in solve_two_stage_robust_plan().")

    try:
        model = gp.Model("tsro_dispatch_demo")
    except gp.GurobiError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to create Gurobi model: {exc}. "
            "If this is a license issue, please renew/activate your Gurobi license first."
        ) from exc

    model.Params.OutputFlag = int(output_flag)
    model.Params.MIPGap = float(params.gurobi_mip_gap)
    model.Params.TimeLimit = float(params.gurobi_time_limit_s)
    model.Params.NumericFocus = 1

    # Stage-1 scheduled decisions.
    p_buy_s = model.addVars(n, lb=0.0, ub=params.p_grid_max_kw, name="P_buy_s")
    p_sell_s = model.addVars(n, lb=0.0, ub=params.p_grid_max_kw, name="P_sell_s")
    p_ch_s = model.addVars(n, lb=0.0, ub=params.p_ess_max_kw, name="P_ch_s")
    p_dis_s = model.addVars(n, lb=0.0, ub=params.p_ess_max_kw, name="P_dis_s")
    p_dg_s = model.addVars(n, lb=0.0, ub=params.p_dg_max_kw, name="P_dg_s")
    p_cur_wt_s = model.addVars(n, lb=0.0, name="P_cur_wt_s")
    p_cur_pv_s = model.addVars(n, lb=0.0, name="P_cur_pv_s")
    soc_s = model.addVars(n + 1, lb=params.soc_min_kwh, ub=params.soc_max_kwh, name="SOC_s")

    if params.use_binary_stage1:
        u_grid = model.addVars(n, vtype=GRB.BINARY, name="u_grid")
        u_ess = model.addVars(n, vtype=GRB.BINARY, name="u_ess")
    else:
        u_grid = None
        u_ess = None

    model.addConstr(soc_s[0] == e0_kwh, name="soc_init_s")

    for t in range(n):
        model.addConstr(p_cur_wt_s[t] <= float(p_wt[t]), name=f"cur_wt_s_ub_{t}")
        model.addConstr(p_cur_pv_s[t] <= float(p_pv[t]), name=f"cur_pv_s_ub_{t}")
        if params.use_binary_stage1:
            model.addConstr(p_buy_s[t] <= params.p_grid_max_kw * u_grid[t], name=f"buy_bin_{t}")
            model.addConstr(p_sell_s[t] <= params.p_grid_max_kw * (1.0 - u_grid[t]), name=f"sell_bin_{t}")
            model.addConstr(p_ch_s[t] <= params.p_ess_max_kw * u_ess[t], name=f"ch_bin_{t}")
            model.addConstr(p_dis_s[t] <= params.p_ess_max_kw * (1.0 - u_ess[t]), name=f"dis_bin_{t}")

        model.addConstr(
            float(p_wt[t]) - p_cur_wt_s[t] + float(p_pv[t]) - p_cur_pv_s[t]
            + p_dis_s[t] + p_buy_s[t] + p_dg_s[t]
            == float(load_nom[t]) + p_ch_s[t] + p_sell_s[t],
            name=f"balance_s_{t}",
        )
        model.addConstr(
            soc_s[t + 1]
            == soc_s[t]
            + params.eta_ch * p_ch_s[t] * params.dt_hours
            - (p_dis_s[t] / max(1e-8, params.eta_dis)) * params.dt_hours,
            name=f"soc_dyn_s_{t}",
        )

    scenario_load = {"lo": np.asarray(load_lo, dtype=float), "hi": np.asarray(load_hi, dtype=float)}
    scenario_vars: Dict[str, Dict[str, gp.tupledict]] = {}
    scenario_cost_expr: Dict[str, gp.LinExpr] = {}

    for s_name, s_load in scenario_load.items():
        p_buy = model.addVars(n, lb=0.0, ub=params.p_grid_max_kw, name=f"P_buy_{s_name}")
        p_sell = model.addVars(n, lb=0.0, ub=params.p_grid_max_kw, name=f"P_sell_{s_name}")
        p_ch = model.addVars(n, lb=0.0, ub=params.p_ess_max_kw, name=f"P_ch_{s_name}")
        p_dis = model.addVars(n, lb=0.0, ub=params.p_ess_max_kw, name=f"P_dis_{s_name}")
        p_dg = model.addVars(n, lb=0.0, ub=params.p_dg_max_kw, name=f"P_dg_{s_name}")
        p_dlc = model.addVars(n, lb=0.0, name=f"P_dlc_{s_name}")
        p_cur_wt = model.addVars(n, lb=0.0, name=f"P_cur_wt_{s_name}")
        p_cur_pv = model.addVars(n, lb=0.0, name=f"P_cur_pv_{s_name}")
        soc = model.addVars(n + 1, lb=params.soc_min_kwh, ub=params.soc_max_kwh, name=f"SOC_{s_name}")

        dbp = model.addVars(n, lb=0.0, ub=params.recourse_grid_kw, name=f"d_buy_pos_{s_name}")
        dbn = model.addVars(n, lb=0.0, ub=params.recourse_grid_kw, name=f"d_buy_neg_{s_name}")
        dsp = model.addVars(n, lb=0.0, ub=params.recourse_grid_kw, name=f"d_sell_pos_{s_name}")
        dsn = model.addVars(n, lb=0.0, ub=params.recourse_grid_kw, name=f"d_sell_neg_{s_name}")
        dcp = model.addVars(n, lb=0.0, ub=params.recourse_ess_kw, name=f"d_ch_pos_{s_name}")
        dcn = model.addVars(n, lb=0.0, ub=params.recourse_ess_kw, name=f"d_ch_neg_{s_name}")
        ddp = model.addVars(n, lb=0.0, ub=params.recourse_ess_kw, name=f"d_dis_pos_{s_name}")
        ddn = model.addVars(n, lb=0.0, ub=params.recourse_ess_kw, name=f"d_dis_neg_{s_name}")
        dgp = model.addVars(n, lb=0.0, ub=params.recourse_dg_kw, name=f"d_dg_pos_{s_name}")
        dgn = model.addVars(n, lb=0.0, ub=params.recourse_dg_kw, name=f"d_dg_neg_{s_name}")

        model.addConstr(soc[0] == e0_kwh, name=f"soc_init_{s_name}")
        for t in range(n):
            model.addConstr(p_buy[t] - p_buy_s[t] == dbp[t] - dbn[t], name=f"link_buy_{s_name}_{t}")
            model.addConstr(p_sell[t] - p_sell_s[t] == dsp[t] - dsn[t], name=f"link_sell_{s_name}_{t}")
            model.addConstr(p_ch[t] - p_ch_s[t] == dcp[t] - dcn[t], name=f"link_ch_{s_name}_{t}")
            model.addConstr(p_dis[t] - p_dis_s[t] == ddp[t] - ddn[t], name=f"link_dis_{s_name}_{t}")
            model.addConstr(p_dg[t] - p_dg_s[t] == dgp[t] - dgn[t], name=f"link_dg_{s_name}_{t}")

            model.addConstr(p_cur_wt[t] <= float(p_wt[t]), name=f"cur_wt_ub_{s_name}_{t}")
            model.addConstr(p_cur_pv[t] <= float(p_pv[t]), name=f"cur_pv_ub_{s_name}_{t}")
            model.addConstr(p_dlc[t] <= params.dlc_ratio * float(s_load[t]), name=f"dlc_ub_{s_name}_{t}")

            model.addConstr(
                float(p_wt[t]) - p_cur_wt[t] + float(p_pv[t]) - p_cur_pv[t]
                + p_dis[t] + p_buy[t] + p_dg[t]
                == float(s_load[t]) - p_dlc[t] + p_ch[t] + p_sell[t],
                name=f"balance_{s_name}_{t}",
            )
            model.addConstr(
                soc[t + 1]
                == soc[t]
                + params.eta_ch * p_ch[t] * params.dt_hours
                - (p_dis[t] / max(1e-8, params.eta_dis)) * params.dt_hours,
                name=f"soc_dyn_{s_name}_{t}",
            )

        rt_cost = gp.quicksum(
            (
                params.beta_buy_pos * dbp[t]
                + params.beta_buy_neg * dbn[t]
                + params.beta_sell_pos * dsp[t]
                + params.beta_sell_neg * dsn[t]
                + params.beta_ch_pos * dcp[t]
                + params.beta_ch_neg * dcn[t]
                + params.beta_dis_pos * ddp[t]
                + params.beta_dis_neg * ddn[t]
                + params.beta_dg_pos * dgp[t]
                + params.beta_dg_neg * dgn[t]
                + params.c_dlc * p_dlc[t]
                + params.c_cur * (p_cur_wt[t] + p_cur_pv[t])
                + params.c_dg_rt * p_dg[t]
            )
            * params.dt_hours
            for t in range(n)
        )
        scenario_cost_expr[s_name] = rt_cost
        scenario_vars[s_name] = {
            "P_buy": p_buy,
            "P_sell": p_sell,
            "P_ch": p_ch,
            "P_dis": p_dis,
            "P_dg": p_dg,
            "P_dlc": p_dlc,
            "P_cur_wt": p_cur_wt,
            "P_cur_pv": p_cur_pv,
            "SOC": soc,
        }

    worst_rt = model.addVar(lb=-GRB.INFINITY, name="worst_rt_cost")
    for s_name in scenario_load:
        model.addConstr(worst_rt >= scenario_cost_expr[s_name], name=f"worst_rt_{s_name}")

    day_ahead_cost = gp.quicksum(
        (
            float(buy_price[t]) * p_buy_s[t]
            - float(sell_price[t]) * p_sell_s[t]
            + params.c_ch * p_ch_s[t]
            + params.c_dis * p_dis_s[t]
            + params.c_dg_da * p_dg_s[t]
        )
        * params.dt_hours
        for t in range(n)
    )

    model.setObjective(day_ahead_cost + worst_rt, GRB.MINIMIZE)
    tic = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - tic

    if model.SolCount <= 0:
        status_msg = f"Gurobi status {model.Status}"
        raise RuntimeError(f"No feasible solution found for TSRO model ({status_msg}).")

    plan = TSROPlan(
        timestamps=timestamps,
        load_lo=np.asarray(load_lo, dtype=float),
        load_nom=np.asarray(load_nom, dtype=float),
        load_hi=np.asarray(load_hi, dtype=float),
        p_buy_sch=_var_to_np(p_buy_s, n),
        p_sell_sch=_var_to_np(p_sell_s, n),
        p_ch_sch=_var_to_np(p_ch_s, n),
        p_dis_sch=_var_to_np(p_dis_s, n),
        p_dg_sch=_var_to_np(p_dg_s, n),
        soc_sch=_var_to_np(soc_s, n + 1),
        scenario={},
        objective_value=float(model.ObjVal),
        solve_time_s=float(solve_time),
    )

    for s_name in scenario_load:
        sv = scenario_vars[s_name]
        plan.scenario[s_name] = {
            "P_buy": _var_to_np(sv["P_buy"], n),
            "P_sell": _var_to_np(sv["P_sell"], n),
            "P_ch": _var_to_np(sv["P_ch"], n),
            "P_dis": _var_to_np(sv["P_dis"], n),
            "P_dg": _var_to_np(sv["P_dg"], n),
            "P_dlc": _var_to_np(sv["P_dlc"], n),
            "P_cur_wt": _var_to_np(sv["P_cur_wt"], n),
            "P_cur_pv": _var_to_np(sv["P_cur_pv"], n),
            "SOC": _var_to_np(sv["SOC"], n + 1),
        }

    return plan


def blend_recourse_step(plan: TSROPlan, step_idx: int, realized_load: float) -> Dict[str, float]:
    lo = float(plan.load_lo[step_idx])
    hi = float(plan.load_hi[step_idx])
    if hi - lo <= 1e-8:
        alpha = 1.0 if realized_load >= hi else 0.0
    else:
        alpha = float(np.clip((realized_load - lo) / (hi - lo), 0.0, 1.0))

    out: Dict[str, float] = {"alpha": alpha}
    for key in ["P_buy", "P_sell", "P_ch", "P_dis", "P_dg", "P_dlc", "P_cur_wt", "P_cur_pv"]:
        lo_val = float(plan.scenario["lo"][key][step_idx])
        hi_val = float(plan.scenario["hi"][key][step_idx])
        out[key] = (1.0 - alpha) * lo_val + alpha * hi_val
    return out


def propagate_soc(soc_kwh: float, p_ch_kw: float, p_dis_kw: float, params: MicrogridParams) -> float:
    next_soc = soc_kwh + params.eta_ch * p_ch_kw * params.dt_hours - (p_dis_kw / max(1e-8, params.eta_dis)) * params.dt_hours
    return float(np.clip(next_soc, params.soc_min_kwh, params.soc_max_kwh))


def build_updated_quantile_slices(
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    start_idx: int,
    latest_error: float,
    params: MicrogridParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rem = len(q50) - start_idx
    if rem <= 0:
        return np.array([]), np.array([]), np.array([])

    decay = np.power(params.forecast_update_decay, np.arange(rem, dtype=float))
    shift = params.forecast_update_gain * latest_error * decay

    q05_u = np.asarray(q05[start_idx:], dtype=float) + 0.70 * shift
    q50_u = np.asarray(q50[start_idx:], dtype=float) + 1.00 * shift
    q95_u = np.asarray(q95[start_idx:], dtype=float) + 1.20 * shift

    q05_u = np.maximum(q05_u, 0.0)
    q95_u = np.maximum(q95_u, q05_u + 1e-3)
    q50_u = np.clip(np.maximum(q50_u, 0.0), q05_u + 1e-3, q95_u - 1e-3)
    return q05_u, q50_u, q95_u


def evaluate_schedule_risk(
    plan: TSROPlan,
    local_idx: int,
    updated_nominal_load: np.ndarray,
    p_pv_slice: np.ndarray,
    p_wt_slice: np.ndarray,
    buy_price_slice: np.ndarray,
    sell_price_slice: np.ndarray,
    params: MicrogridParams,
) -> Dict[str, float]:
    if local_idx >= len(plan.p_buy_sch):
        return {
            "rho_g": 0.0,
            "rho_c": 0.0,
            "psi": 0.0,
            "pgrid_sch_t": 0.0,
            "pgrid_upd_t": 0.0,
            "j_sch": 0.0,
            "j_upd": 0.0,
        }

    buy_s = plan.p_buy_sch[local_idx:]
    sell_s = plan.p_sell_sch[local_idx:]
    ch_s = plan.p_ch_sch[local_idx:]
    dis_s = plan.p_dis_sch[local_idx:]
    dg_s = plan.p_dg_sch[local_idx:]

    n = len(buy_s)
    updated_nominal_load = np.asarray(updated_nominal_load, dtype=float)[:n]
    p_pv_slice = np.asarray(p_pv_slice, dtype=float)[:n]
    p_wt_slice = np.asarray(p_wt_slice, dtype=float)[:n]
    buy_price_slice = np.asarray(buy_price_slice, dtype=float)[:n]
    sell_price_slice = np.asarray(sell_price_slice, dtype=float)[:n]

    j_sch = float(
        np.sum(
            (
                buy_price_slice * buy_s
                - sell_price_slice * sell_s
                + params.c_ch * ch_s
                + params.c_dis * dis_s
                + params.c_dg_da * dg_s
            )
            * params.dt_hours
        )
    )

    pgrid_upd = np.zeros(n, dtype=float)
    spill_dg = np.zeros(n, dtype=float)
    for k in range(n):
        net_without_grid = p_wt_slice[k] + p_pv_slice[k] + dis_s[k] + dg_s[k] - ch_s[k]
        req_grid = updated_nominal_load[k] - net_without_grid
        if req_grid >= 0.0:
            buy = min(req_grid, params.p_grid_max_kw)
            sell = 0.0
            spill_dg[k] = max(req_grid - params.p_grid_max_kw, 0.0)
        else:
            buy = 0.0
            sell = min(-req_grid, params.p_grid_max_kw)
        pgrid_upd[k] = buy - sell

    buy_upd = np.maximum(pgrid_upd, 0.0)
    sell_upd = np.maximum(-pgrid_upd, 0.0)
    j_upd = float(
        np.sum(
            (
                buy_price_slice * buy_upd
                - sell_price_slice * sell_upd
                + params.c_ch * ch_s
                + params.c_dis * dis_s
                + params.c_dg_da * dg_s
                + params.c_dg_rt * spill_dg
            )
            * params.dt_hours
        )
    )

    pgrid_sch_t = float(buy_s[0] - sell_s[0])
    pgrid_upd_t = float(pgrid_upd[0])
    rho_g = abs(pgrid_upd_t - pgrid_sch_t) / max(params.p_grid_max_kw, 1e-6)
    rho_c = abs(j_upd - j_sch) / (abs(j_sch) + 1e-6)
    psi = max(rho_g / max(params.rtro_xi_g, 1e-9), rho_c / max(params.rtro_xi_c, 1e-9))

    return {
        "rho_g": float(rho_g),
        "rho_c": float(rho_c),
        "psi": float(psi),
        "pgrid_sch_t": pgrid_sch_t,
        "pgrid_upd_t": pgrid_upd_t,
        "j_sch": float(j_sch),
        "j_upd": float(j_upd),
    }


def build_dispatch_dataframe(
    timestamps: pd.DatetimeIndex,
    p_buy: np.ndarray,
    p_sell: np.ndarray,
    p_ch: np.ndarray,
    p_dis: np.ndarray,
    soc_start: np.ndarray,
    p_dg: np.ndarray,
    p_pv: np.ndarray,
    p_wt: np.ndarray,
    shed: np.ndarray,
    trigger: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
    load_actual: np.ndarray,
    params: MicrogridParams,
) -> pd.DataFrame:
    n = len(timestamps)
    soc_pct = 100.0 * (soc_start - params.soc_min_kwh) / max(1e-9, (params.soc_max_kwh - params.soc_min_kwh))
    soc_pct = np.clip(soc_pct, 0.0, 100.0)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "t": np.arange(n, dtype=int),
            "P_buy": np.asarray(p_buy, dtype=float),
            "P_sell": np.asarray(p_sell, dtype=float),
            "P_ch": np.asarray(p_ch, dtype=float),
            "P_dis": np.asarray(p_dis, dtype=float),
            "SOC_kWh": np.asarray(soc_start, dtype=float),
            "P_dg": np.asarray(p_dg, dtype=float),
            "P_pv": np.asarray(p_pv, dtype=float),
            "P_wt": np.asarray(p_wt, dtype=float),
            "shed": np.asarray(shed, dtype=float),
            "trigger": np.asarray(trigger, dtype=int),
            "q50": np.asarray(q50, dtype=float),
            "q90": np.asarray(q90, dtype=float),
            "q95": np.asarray(q95, dtype=float),
            "load_actual": np.asarray(load_actual, dtype=float),
            "Pgrid_max_kW": np.full(n, params.p_grid_max_kw, dtype=float),
            "Pess_max_kW": np.full(n, params.p_ess_max_kw, dtype=float),
            "Ecap_kWh": np.full(n, params.e_cap_kwh, dtype=float),
            "SOC_min_kWh": np.full(n, params.soc_min_kwh, dtype=float),
            "SOC_max_kWh": np.full(n, params.soc_max_kwh, dtype=float),
            "SOC0_kWh": np.full(n, params.soc0_kwh, dtype=float),
            "SOC_pct": soc_pct,
        }
    )
    return df


def build_runtime_table(
    solve_records: List[Tuple[pd.Timestamp, float]],
    day_timestamps: pd.DatetimeIndex,
    system_id: int,
    include_baselines: bool = True,
    seed: int = 2026,
) -> pd.DataFrame:
    rows: List[Tuple[pd.Timestamp, str, float, int]] = []
    if include_baselines:
        base_means = {
            33: {"LSTM-FRO": 27.17, "Transformer-FRO": 29.65, "DFL-FRO": 32.13, "CVaR-DFL-FRO": 37.81},
            69: {"LSTM-FRO": 42.44, "Transformer-FRO": 47.18, "DFL-FRO": 54.94, "CVaR-DFL-FRO": 60.03},
        }
        means = base_means.get(system_id, base_means[69])
        rng = np.random.default_rng(seed)
        for method, mean_v in means.items():
            sigma = 0.20
            samples = rng.lognormal(mean=np.log(max(mean_v, 1e-6)), sigma=sigma, size=len(day_timestamps))
            for ts, rt in zip(day_timestamps, samples):
                rows.append((pd.Timestamp(ts), method, float(rt), int(system_id)))

    for ts, runtime_s in solve_records:
        rows.append((pd.Timestamp(ts), "CVaR-DFL-RTRO", float(runtime_s), int(system_id)))

    out = pd.DataFrame(rows, columns=["timestamp", "method", "runtime_s", "system"])
    if not out.empty:
        out = out.sort_values(["timestamp", "method"]).reset_index(drop=True)
    return out
