# RADFL Artifacts

This directory mixes three kinds of assets:

## Inputs
- `typical_forecast_task_L14d_2022-10-13_to_2022-10-27.csv`
- `extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv`
- `pred_quantiles_extreme.csv`
- `dispatch_placeholder_24h_15min_tunedCaps_v6.csv`

## Placeholder And Historical Reference Files
- `fig_iv03_parts/placeholder_rtro_trigger_day_extreme.csv`
- `fig_iv03_parts/placeholder_solver_runtime_all.csv`

These are paper-style placeholders and historical references, not direct solver outputs from the current machine.

## Demo Outputs
- `dispatch_pyomo_demo.csv`
- `dispatch_pyomo_rtro_demo.csv`
- `fig_iv03_parts/rtro_trigger_day_pyomo_demo.csv`
- `fig_iv03_parts/solver_runtime_day_pyomo_demo.csv`

These are solver demo artifacts kept for traceability.

## Approximate Paper Reproduction Outputs
- `paper_repro/dispatch_cvxpy_paper_repro.csv`
- `paper_repro/dispatch_cvxpy_paper_repro_rtro.csv`
- `paper_repro/rtro_trigger_day_cvxpy_paper_repro.csv`
- `paper_repro/solver_runtime_day_cvxpy_paper_repro.csv`
- `paper_repro/cvxpy_paper_repro_metadata.json`

## Compatibility Files
- `rtro_trigger_day.csv`
- `solver_runtime_all.csv`

These two files stay at the top level because `RTRO.py --use-real-logs` expects them in this location.