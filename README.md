# CVaR-DFL-RTRO

This repository contains inference-only artifacts for forecasting case studies and optimization-demo scripts for robust microgrid dispatch.

## Included Scripts

- `load_inference_combined.py` (typical + extreme load inference)
- `typical_load.py` (typical-only inference)
- `extreme_load.py` (extreme-only inference)
- `Dispatch.py` (converted from `RADFL_online_dispatch.ipynb`, plotting)
- `RTRO.py` (converted from `ARH.ipynb`, plotting + placeholder logs)
- `optimization_dispatch_demo.py` (new: TSRO dispatch demo with Gurobi)
- `optimization_rtro_demo.py` (new: RTRO online re-optimization demo with Gurobi)
- `optimization_pyomo_demo.py` (new: Pyomo TSRO/RTRO demo; solver selectable: appsi_highs/cplex/gurobi)
- `optimization_core.py` (new: shared optimization core module)

Training code is intentionally removed from standalone scripts where applicable. Scripts directly load checkpoint/data files and run inference/plotting/optimization demos.

## Included Data and Artifacts

- `darts_logs/typical/*` checkpoint files
- `darts_logs/extreme/*` checkpoint files
- `datasets/RADFL/typical_forecast_task_L14d_2022-10-13_to_2022-10-27.csv`
- `datasets/RADFL/extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv`
- `datasets/RADFL/pred_quantiles_extreme.csv`
- `datasets/RADFL/dispatch_placeholder_24h_15min_tunedCaps_v6.csv`
- `datasets/RADFL/fig_iv03_parts/placeholder_rtro_trigger_day_extreme.csv`
- `datasets/RADFL/fig_iv03_parts/placeholder_solver_runtime_all.csv`
- `datasets/RADFL/dispatch_pyomo_demo.csv`
- `datasets/RADFL/dispatch_pyomo_rtro_demo.csv`
- `datasets/RADFL/fig_iv03_parts/rtro_trigger_day_pyomo_demo.csv`
- `datasets/RADFL/fig_iv03_parts/solver_runtime_day_pyomo_demo.csv`
- `requirements/*` copied from `darts-master/requirements`
- `darts_definition_files/*` selected source definition files from `paper_FEDQR/darts-master/darts`

## Darts Definition Mapping

The scripts import these Darts APIs, with definitions mapped to the following files in this repo:

- `from darts import TimeSeries`
  - `darts_definition_files/darts/timeseries.py` (`class TimeSeries`)
- `from darts.dataprocessing.transformers import Scaler`
  - `darts_definition_files/darts/dataprocessing/transformers/scaler.py` (`class Scaler`)
- `from darts.metrics import rmse, smape, r2_score`
  - `darts_definition_files/darts/metrics/metrics.py` (`def rmse`, `def smape`, `def r2_score`)
- `from darts.models import TFTModel`
  - `darts_definition_files/darts/models/forecasting/tft_model.py` (`class TFTModel`)
- `TFTModel.load_from_checkpoint(...)` (inherited implementation)
  - `darts_definition_files/darts/models/forecasting/torch_forecasting_model.py` (`def load_from_checkpoint`)

Entry-point import files are also included for traceability:
- `darts_definition_files/darts/__init__.py`
- `darts_definition_files/darts/dataprocessing/transformers/__init__.py`
- `darts_definition_files/darts/metrics/__init__.py`
- `darts_definition_files/darts/models/__init__.py`

Note: `darts_definition_files/` is a source-reference snapshot. Runtime still uses installed `darts` package.

## Repository Layout

```text
CVaR-DFL-RTRO/
  load_inference_combined.py
  typical_load.py
  extreme_load.py
  Dispatch.py
  RTRO.py
  optimization_core.py
  optimization_dispatch_demo.py
  optimization_rtro_demo.py
  optimization_pyomo_demo.py
  darts_logs/
    typical/
    extreme/
  datasets/
    RADFL/
      typical_forecast_task_L14d_2022-10-13_to_2022-10-27.csv
      extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv
      pred_quantiles_extreme.csv
      dispatch_placeholder_24h_15min_tunedCaps_v6.csv
      fig_iv03_parts/
        placeholder_rtro_trigger_day_extreme.csv
        placeholder_solver_runtime_all.csv
  requirements/
    core.txt
    torch.txt
    release.txt
    dev.txt
    dev-all.txt
  darts_definition_files/
    darts/
      __init__.py
      timeseries.py
      dataprocessing/transformers/__init__.py
      dataprocessing/transformers/scaler.py
      metrics/__init__.py
      metrics/metrics.py
      models/__init__.py
      models/forecasting/tft_model.py
      models/forecasting/torch_forecasting_model.py
```

## Environment

Recommended: Python 3.8 (same as the original FEDM runtime).

Minimal install example:

```bash
pip install -r requirements/core.txt -r requirements/torch.txt
```

Optional requirement sets:
- `requirements/release.txt`
- `requirements/dev.txt`
- `requirements/dev-all.txt`

For optimization demos, install Gurobi Python API in the active environment:

```bash
pip install gurobipy pyomo highspy
```

You also need a valid Gurobi license (`grbgetkey` or existing `gurobi.lic`).

## Run

Combined load inference:

```bash
python load_inference_combined.py --no-show
```

Typical-only load inference:

```bash
python typical_load.py --no-show
```

Extreme-only load inference:

```bash
python extreme_load.py --no-show
```

Dispatch plotting (from RADFL_online_dispatch):

```bash
python Dispatch.py
```

RTRO plotting (from ARH notebook, placeholder mode by default):

```bash
python RTRO.py
```

RTRO plotting with real logs:

```bash
python RTRO.py --use-real-logs
```

### Optimization Demo: TSRO Dispatch (Gurobi)

```bash
python optimization_dispatch_demo.py \
  --dataset-dir ./datasets/RADFL \
  --day 2022-04-29
```

Optional:

```bash
python optimization_dispatch_demo.py --solver-output
```

Default output:

- `datasets/RADFL/dispatch_tsro_demo.csv`

### Optimization Demo: RTRO Online Re-Optimization (Gurobi)

```bash
python optimization_rtro_demo.py \
  --dataset-dir ./datasets/RADFL \
  --day 2022-04-29
```

Optional threshold/window override:

```bash
python optimization_rtro_demo.py \
  --xi-g 1e-3 --xi-c 0.05 --delta-min 1 --delta-max 8
```

Default outputs:

- `datasets/RADFL/dispatch_rtro_demo.csv`
- `datasets/RADFL/fig_iv03_parts/rtro_trigger_day_demo.csv`
- `datasets/RADFL/fig_iv03_parts/solver_runtime_day_demo.csv`


### Optimization Demo: Pyomo (HiGHS/CPLEX/Gurobi)

```bash
python optimization_pyomo_demo.py --mode dispatch --solver auto --day 2022-04-29
python optimization_pyomo_demo.py --mode rtro --solver auto --day 2022-04-29
```

Notes:
- `--solver auto` currently resolves to `appsi_highs` in FEDM.
- If your machine has CPLEX, use `--solver cplex` directly.
- In current HiGHS runtime, RTRO demo keeps trigger evaluation and CSV export stable (no repeated re-solve).

### Notes on the Optimization Demos

- The optimization model follows the paper's two-stage robust structure: stage-1 schedule + stage-2 recourse under PI uncertainty (`q05/q50/q95`).
- RTRO trigger follows the paper logic with normalized risk indicator and min/max trigger windows.
- Dispatch output columns are aligned with `dispatch_placeholder_24h_15min_tunedCaps_v6.csv`.
- Trigger/runtime output columns are aligned with `placeholder_rtro_trigger_day_extreme.csv` and `placeholder_solver_runtime_all.csv`.

## Outputs

`typical_load.py` exports:
- `typical_day_forecast.pdf/png`
- `pred_quantiles.csv`
- `pred_ci_*.csv`
- terminal metrics: `RMSE`, `SMAPE`, `R^2`

`extreme_load.py` exports:
- `extreme_day_forecast.pdf/png`
- `pred_quantiles_extreme.csv`
- `pred_ci_*_extreme.csv`
- terminal metrics: `RMSE`, `SMAPE`, `R^2`

`Dispatch.py` exports:
- `datasets/RADFL/dispatch_parts/dispatch.png`
- `datasets/RADFL/dispatch_parts/dispatch.pdf`

`RTRO.py` exports (default placeholder mode):
- `datasets/RADFL/fig_iv03_parts/a_trigger_timeline.pdf/png`
- `datasets/RADFL/fig_iv03_parts/b_runtime_per_solve.pdf/png`
- `datasets/RADFL/fig_iv03_parts/c_cpu_time_day.pdf/png`
- `datasets/RADFL/fig_iv03_parts/placeholder_rtro_trigger_day_extreme.csv`
- `datasets/RADFL/fig_iv03_parts/placeholder_solver_runtime_all.csv`

`optimization_dispatch_demo.py` exports:
- `datasets/RADFL/dispatch_tsro_demo.csv` (default)

`optimization_rtro_demo.py` exports:
- `datasets/RADFL/dispatch_rtro_demo.csv` (default)
- `datasets/RADFL/fig_iv03_parts/rtro_trigger_day_demo.csv` (default)
- `datasets/RADFL/fig_iv03_parts/solver_runtime_day_demo.csv` (default)


`optimization_pyomo_demo.py` exports:
- `datasets/RADFL/dispatch_pyomo_demo.csv` (dispatch mode default)
- `datasets/RADFL/dispatch_pyomo_rtro_demo.csv` (rtro mode default)
- `datasets/RADFL/fig_iv03_parts/rtro_trigger_day_pyomo_demo.csv` (rtro mode default)
- `datasets/RADFL/fig_iv03_parts/solver_runtime_day_pyomo_demo.csv` (rtro mode default)

## Notes

- Plot styles are aligned with original notebook output styles.
- This repository provides inference/dispatch assets, optimization demo scripts, and checkpoints, not end-to-end training pipelines.
