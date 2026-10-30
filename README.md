# SoilCastNet – soil-moisture forecasting for smart irrigation (benchmark + proposed model)

Compares 9 existing forecasters (Persistence, Ridge, MLP, LSTM, GRU, 1D-CNN, CNN-LSTM, TCN, Transformer, DLinear)
with the proposed **SoilCastNet** on real sensor data; adds ablations, conformal-calibrated uncertainty,
robustness to sensor faults, per-sensor results, significance tests and an irrigation-trigger proxy.

## 1. Setup
    pip install -r requirements.txt          # GPU strongly recommended (Colab/Kaggle notebook is fine)

## 2. Data (real)
Download from Kaggle (needs your Kaggle login; the CLI works too: `kaggle datasets download -d sathyanarayanrao89/soil-moisture-data-from-field-scale-sensor-network --unzip -p data/`).
Then look at the columns first - this tells you the arguments to use:

    python run_experiments.py --inspect --data data/            # folder of per-sensor files, or a single csv

## 3. Run
Single CSV with a sensor-id column:
    python run_experiments.py --data data/soil.csv --time-col timestamp --target-col vwc --group-col sensor_id --freq h

Folder with one file per sensor (file name = sensor id):
    python run_experiments.py --data data/Hourly/ --time-col <time column> --target-col <VWC column> --freq h

Useful options: `--split spatial` (unseen sensors), `--lookback 48`, `--horizons 1,6,24`, `--seeds 5`,
`--epochs 30`, `--target-range 0,0.7` (drop impossible readings), `--feature-cols col1,col2` (e.g. rainfall, air temp),
`--max-train 200000` (cap windows for speed), `--no-ablation`, `--no-robustness`.
Run it twice (`--split temporal` and `--split spatial`, different `--out`) - both go in the paper.

`python run_experiments.py --demo --quick` runs the whole pipeline on a SYNTHETIC stand-in (pipeline test only; never report it).

## 4. Output (results/)
tables/*.csv + *.tex (MAE, RMSE, NSE, KGE, event-MAE, skill vs persistence, intervals, ablation, significance,
robustness, feature importance, irrigation proxy) and figures/*.png|pdf (fig01-fig14).

## Notes
* Only soil moisture (+ whatever numeric columns you provide) is used. If your data has no rainfall/irrigation column the
  model must infer wetting events from the moisture history - say so in the paper. Adding rainfall via `--feature-cols` is the single best improvement.
* The irrigation-trigger metrics are a PROXY (deficit below each sensor's 20th percentile); no real irrigation labels exist in soil-moisture datasets. Do not claim water savings.
* Conformal calibration (CQR) guarantees coverage only if validation and test are exchangeable; coverage can drift under seasonal shift - report it honestly.
* Compare against persistence first: soil moisture is highly autocorrelated, so beating it at +1 step is hard; gains show at longer horizons and on wetting events (`MAE_event`).
