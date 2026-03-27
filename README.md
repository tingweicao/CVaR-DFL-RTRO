# CVaR-DFL-RTRO

This repository contains inference-only artifacts for the two load-forecasting case studies:
- typical load day
- extreme load day

The script reproduces plotting and CSV export from the original notebooks, but skips training and directly loads checkpoints.

## Included Files

- `load_inference_combined.py`
- `darts_logs/typical/*` checkpoint files
- `darts_logs/extreme/*` checkpoint files
- `requirements/*` copied from `darts-master/requirements`

## Repository Layout

```text
CVaR-DFL-RTRO/
  load_inference_combined.py
  darts_logs/
    typical/
      _model.pth.tar
      checkpoints/
        best-epoch=99-val_loss=0.17.ckpt
        last-epoch=199.ckpt
    extreme/
      _model.pth.tar
      checkpoints/
        best-epoch=26-val_loss=0.82.ckpt
        last-epoch=499.ckpt
  requirements/
    core.txt
    torch.txt
    release.txt
    dev.txt
    dev-all.txt
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

Default dataset location expected by the script:
- `./datasets/RADFL`

## Run

From repository root:

```bash
python load_inference_combined.py --dataset-dir "<path-to-RADFL>" --no-show
```

Arguments:
- `--no-show`: save figures only, do not open plot windows
- `--dataset-dir`: RADFL CSV directory
- `--work-dir`: checkpoint root directory (default `./darts_logs`)
- `--output-dir`: output directory for figures and CSVs (default script directory)

## Outputs

For both `typical` and `extreme`, the script exports:
- Figures: `typical_day_forecast.pdf/png`, `extreme_day_forecast.pdf/png`
- Quantiles: `pred_quantiles.csv`, `pred_quantiles_extreme.csv`
- Prediction intervals: `pred_ci_*.csv`, `pred_ci_*_extreme.csv`
- Metrics printed to terminal: `RMSE`, `SMAPE`, `R^2`

## Notes

- Plot style is aligned with the original notebook outputs.
- This repository provides inference assets and checkpoints, not training pipelines.
