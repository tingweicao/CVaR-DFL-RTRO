# CVaR-DFL-RTRO

This repository contains inference-only artifacts for multiple forecasting/dispatch case studies.

## Included Scripts

- `load_inference_combined.py` (typical + extreme load inference)
- `typical_load.py` (typical-only inference)
- `extreme_load.py` (extreme-only inference)
- `Dispatch.py` (converted from `RADFL_online_dispatch.ipynb`)
- `RTRO.py` (converted from `ARH.ipynb`)

Training code is intentionally removed from standalone scripts where applicable. Scripts directly load checkpoint/data files and run inference/plotting.

## Included Data and Artifacts

- `darts_logs/typical/*` checkpoint files
- `darts_logs/extreme/*` checkpoint files
- `datasets/RADFL/typical_forecast_task_L14d_2022-10-13_to_2022-10-27.csv`
- `datasets/RADFL/extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv`
- `datasets/RADFL/pred_quantiles_extreme.csv`
- `datasets/RADFL/dispatch_placeholder_24h_15min_tunedCaps_v6.csv`
- `datasets/RADFL/fig_iv03_parts/placeholder_rtro_trigger_day_extreme.csv`
- `datasets/RADFL/fig_iv03_parts/placeholder_solver_runtime_all.csv`
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

Dispatch (from RADFL_online_dispatch):

```bash
python Dispatch.py
```

RTRO (from ARH notebook, placeholder mode by default):

```bash
python RTRO.py
```

RTRO with real logs:

```bash
python RTRO.py --use-real-logs
```

Common optional arguments are supported in scripts (e.g., `--dataset-dir`, `--work-dir`, `--output-dir`, `--out-dir`) to override default paths.

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

## Notes

- Plot styles are aligned with original notebook output styles.
- This repository provides inference/dispatch assets and checkpoints, not end-to-end training pipelines.
