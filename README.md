# CVaR-DFL-RTRO

This repository contains inference-only artifacts for two load-forecasting case studies:
- typical load day
- extreme load day

Training code is intentionally removed from standalone scripts. All scripts directly load checkpoints and run inference.

## Included Files

- `load_inference_combined.py` (runs typical + extreme in one command)
- `typical_load.py` (typical-only script, no training)
- `extreme_load.py` (extreme-only script, no training)
- `darts_logs/typical/*` checkpoint files
- `darts_logs/extreme/*` checkpoint files
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
  darts_logs/
    typical/
    extreme/
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

## Required Datasets

Place these CSV files in a RADFL directory:
- `typical_forecast_task_L14d_2022-10-13_to_2022-10-27.csv`
- `extreme_forecast_task_L14d_2022-04-15_to_2022-04-29.csv`

Default dataset location expected by scripts:
- `./datasets/RADFL`

## Run

Combined run (both cases):

```bash
python load_inference_combined.py --dataset-dir "<path-to-RADFL>" --no-show
```

Typical-only run:

```bash
python typical_load.py --dataset-dir "<path-to-RADFL>" --no-show
```

Extreme-only run:

```bash
python extreme_load.py --dataset-dir "<path-to-RADFL>" --no-show
```

Common arguments:
- `--no-show`: save figures only, do not open plot windows
- `--dataset-dir`: RADFL CSV directory
- `--work-dir`: checkpoint root directory (default `./darts_logs`)
- `--output-dir`: output directory for figures and CSVs (default script directory)

## Outputs

Typical script exports:
- `typical_day_forecast.pdf/png`
- `pred_quantiles.csv`
- `pred_ci_*.csv`
- metrics in terminal: `RMSE`, `SMAPE`, `R^2`

Extreme script exports:
- `extreme_day_forecast.pdf/png`
- `pred_quantiles_extreme.csv`
- `pred_ci_*_extreme.csv`
- metrics in terminal: `RMSE`, `SMAPE`, `R^2`

## Notes

- Plot style is aligned with original notebook output style.
- This repository provides inference assets and checkpoints, not training pipelines.
