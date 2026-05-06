# Adaptive Multi-Horizon Volatility Forecasting

Karim Nabilsi and Atulya Madhavan<br>
STAT UN 3106: Applied Machine Learning, Columbia University, Spring 2026

## Overview

This project forecasts forward realized volatility for the S&P 500. The main target is annualized 22-day realized volatility, which is roughly a one-month horizon. We start with a standard HAR-RV baseline, add several feature pipelines, and then test whether the forecast should become shorter-horizon during market stress.

The final model uses an XGBoost ensemble with a simple adaptive horizon rule. In calm periods it behaves like a normal 22-day volatility model. When the market is shocked, it blends toward a 5-day forecast so the prediction can react faster.

The best version reduced MSE by 8.5% versus the HAR-RV baseline and improved AUC from 0.8242 to 0.8467.

## Data and Targets

The raw market feature file begins in 2008, but after rolling windows, cached model outputs, and the 5-day target are joined, the adaptive experiment uses 3,862 trading days from 2010-07-16 to 2025-11-19.

The split is time-based:

| Split | Dates | Rows |
| --- | --- | ---: |
| Train | before 2020-01-01 | 2,382 |
| Test | 2020-01-01 onward | 1,480 |

The regression target is 22-day forward realized volatility. For classification-style evaluation, spike days are labeled when forward volatility is above the 90th percentile of the training distribution. The threshold is computed only on training data to avoid leakage.

## Modeling Approach

We used HAR-RV as the first model because it is a simple and well-known volatility benchmark. It predicts future volatility from lagged realized volatility at 1-day, 5-day, and 22-day horizons.

After that, we added feature pipelines that each try to capture a different type of information:

| Component | Input | Output | Purpose |
| --- | --- | --- | --- |
| Market features | Daily price and vol data | 24 features | Core volatility signals |
| HAR-RV | 1d, 5d, 22d RV | 1 prediction | Linear baseline |
| VAE | 30-day windows | 13 features | Latent regime features |
| Transformer | 60-day windows | 32 features | Sequential market patterns |
| FinBERT | NYT/FOMC text | 10 features | Financial sentiment |
| GAT | Cross-asset data | 9 features | Contagion/correlation structure |
| XGBoost | Joined feature table | 1 prediction | Final tabular model |

The full joined model has 88 input features. The neural models are intentionally small because the usable training sample is only a few thousand trading days.

Approximate model scale:

| Model | Size |
| --- | ---: |
| VAE | about 50K parameters |
| Transformer | about 30K parameters |
| GAT | about 5K parameters |
| XGBoost | 500 trees, depth 4 |

## Adaptive Horizon

The adaptive horizon model trains both a 22-day forecaster and a 5-day forecaster. A backward-looking shock score then decides how much to blend between them.

The shock score uses only information available at the prediction date:

- large recent absolute returns
- VIX jumps
- high VIX levels
- short-term realized volatility spikes

The final prediction is:

```text
prediction = (1 - shock_score) * ensemble_22d + shock_score * ensemble_5d
```

The shock score decays with a 5-day half-life. This avoids switching too abruptly after a one-day spike.

## Training and Tuning

The neural pipelines use AdamW with small learning rates and dropout. XGBoost uses a learning rate of 0.05, max depth 4, column subsampling, and L2 regularization.

We tested learning rates from 0.005 to 0.3. The sensitivity curve showed the expected pattern: too small learns slowly, too large becomes unstable, and 0.05 was near the best value.

The project was run on Google Colab free tier with a T4 GPU. A full run takes about 15-20 minutes. The cached evaluation scripts run much faster, usually around a minute.

## Results

| Model | MSE | AUC | MSE vs HAR-RV |
| --- | ---: | ---: | ---: |
| HAR-RV baseline | 0.010660 | 0.8242 | baseline |
| XGBoost ensemble | 0.010451 | 0.8439 | -2.0% |
| Adaptive 2-horizon model | 0.009757 | 0.8467 | -8.5% |

The adaptive model helped most during higher-stress regimes:

| Regime | MSE improvement |
| --- | ---: |
| Calm | 6.9% |
| Normal | 7.2% |
| Stress | 11.7% |
| Crisis | 8.7% |

The Diebold-Mariano test gives p = 0.13. That is not significant at the 5% level, but it is close to the 10% level and is directionally consistent with the error improvements.

## What We Learned

The full system is useful as an experiment, but not every added pipeline improved the model. SHAP and ablation both showed that the Transformer and market features carried most of the predictive signal.

SHAP pipeline contribution:

| Pipeline | Share |
| --- | ---: |
| Transformer | 63.1% |
| Market features | 14.7% |
| FinBERT | 7.6% |
| VAE | 6.8% |
| GAT | 6.8% |
| HAR-RV | 1.0% |

Removing VAE, FinBERT, or GAT slightly improved AUC in the ablation study. That was one of the more important findings: a more complicated model is not automatically better. The final adaptive horizon idea helped because it changed the forecasting horizon during shocks, not because it simply added more features.

## Reproducibility

The project uses a fixed seed of 42. We also tested five seeds and got AUC = 0.8435 +/- 0.0034, so the main result is not driven by one lucky random seed.

Other leakage controls:

- spike threshold is computed from training data only
- crisis centroid features exclude test-period crisis years
- shock detection is backward-looking only
- cached pipeline outputs are saved as CSVs
- FinBERT runs locally instead of using a paid API

## Important Files

| File | Purpose |
| --- | --- |
| `config.py` | shared settings and hyperparameters |
| `run.py` | runs the full cached pipeline |
| `features.py` | builds base market features |
| `p1_vae.py` | VAE regime features |
| `p2_transformer.py` | Transformer sequence features |
| `p3_finbert.py` | text sentiment features |
| `p4_gat.py` | cross-asset graph features |
| `p5_har_rv.py` | HAR-RV baseline |
| `meta_model.py` | XGBoost ensemble |
| `adaptive_horizon.py` | adaptive 22-day / 5-day experiment |
| `evaluation_report_adaptive.py` | report figures and robustness checks |

## How to Run

From the project directory:

```bash
pip install -r requirements.txt
python run.py --skip-download
python evaluation_report_adaptive.py
```

The `--skip-download` flag uses the cached data in `data/raw` and `data/processed`. This is the recommended mode for reproducing the submitted results.
