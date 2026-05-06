"""
Feature engineering: builds all 24 features + targets.
Uses IBKR intraday data for more accurate RV when available.
"""
import numpy as np
import pandas as pd
from config import *


def compute_intraday_rv(intraday_df, daily_index, horizons=[1, 5, 22]):
    log_ret = np.log(intraday_df["Close"] / intraday_df["Close"].shift(1)).dropna()
    daily_rv = log_ret.groupby(log_ret.index.date).apply(
        lambda x: np.sqrt(np.sum(x**2) * 252)
    )
    daily_rv.index = pd.to_datetime(daily_rv.index)
    daily_rv = daily_rv.reindex(daily_index)
    rv_features = pd.DataFrame(index=daily_index)
    for h in horizons:
        rv_features[f"RV_{h}d"] = daily_rv.rolling(h).mean()
    return rv_features


def compute_daily_rv(squared_returns, horizons=[1, 5, 22]):
    rv = pd.DataFrame(index=squared_returns.index)
    for h in horizons:
        rv[f"RV_{h}d"] = np.sqrt(squared_returns.rolling(h).sum() * (252 / h))
    return rv


def build_features(spx, vix_raw, spx_intraday=None, options_df=None, include_target=True):
    print("=" * 60)
    print("BUILDING FEATURES")
    print("=" * 60)

    close = spx["Close"].squeeze()
    high = spx["High"].squeeze()
    low = spx["Low"].squeeze()
    volume = spx["Volume"].squeeze()
    log_ret = np.log(close / close.shift(1))
    squared = log_ret ** 2

    feat = pd.DataFrame(index=close.index)

    if spx_intraday is not None and len(spx_intraday) > 1000:
        print("  Using IBKR intraday data for RV (5-min bars)")
        rv = compute_intraday_rv(spx_intraday, close.index)
        for col in rv.columns:
            feat[col] = rv[col]
    else:
        print("  Using daily close-to-close RV (Yahoo)")
        rv = compute_daily_rv(squared)
        for col in rv.columns:
            feat[col] = rv[col]

    feat["log_return"] = log_ret
    feat["abs_return"] = log_ret.abs()
    feat["return_5d"] = close.pct_change(5)
    feat["return_22d"] = close.pct_change(22)
    feat["vol_5d"] = log_ret.rolling(5).std() * np.sqrt(252)
    feat["vol_22d"] = log_ret.rolling(22).std() * np.sqrt(252)
    feat["parkinson_vol"] = np.sqrt((1/(4*np.log(2))) * (np.log(high/low)**2)) * np.sqrt(252)
    feat["volume_zscore"] = (volume - volume.rolling(22).mean()) / volume.rolling(22).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - (100 / (1 + gain / loss_s))

    feat["vix_level"] = vix_raw.get("VIX")
    if "VIX" in vix_raw.columns and "VIX3M" in vix_raw.columns:
        feat["slope_3m_spot"] = vix_raw["VIX3M"] - vix_raw["VIX"]
    if "VIX3M" in vix_raw.columns and "VIX6M" in vix_raw.columns:
        feat["slope_6m_3m"] = vix_raw["VIX6M"] - vix_raw["VIX3M"]
    if all(c in vix_raw.columns for c in ["VIX", "VIX3M", "VIX6M"]):
        feat["curvature"] = vix_raw["VIX"] - 2*vix_raw["VIX3M"] + vix_raw["VIX6M"]
    if "VIX" in vix_raw.columns and "VIX3M" in vix_raw.columns:
        feat["contango_flag"] = (vix_raw["VIX3M"] > vix_raw["VIX"]).astype(int)
    if "VIX" in vix_raw.columns:
        feat["vix_1d_change"] = vix_raw["VIX"].pct_change()
        feat["vix_zscore_20d"] = (
            (vix_raw["VIX"] - vix_raw["VIX"].rolling(20).mean()) / vix_raw["VIX"].rolling(20).std()
        )
        feat["vix_ma_ratio"] = vix_raw["VIX"] / vix_raw["VIX"].rolling(60).mean()

    if "vix_level" in feat.columns:
        feat["vix_momentum_5d"] = feat["vix_level"].pct_change(5)
        feat["vix_momentum_22d"] = feat["vix_level"].pct_change(22)
        feat["vix_acceleration"] = feat["vix_momentum_5d"].diff(5)
    if "slope_3m_spot" in feat.columns:
        feat["term_slope_change_5d"] = feat["slope_3m_spot"].diff(5)

    
    if options_df is not None:
        print("  Options data available (snapshot only — skipping for training)")

    
    if not include_target:
        feat = feat.dropna()
        target_cols = []
        feature_cols = list(feat.columns)
        print(f"\n  LIVE MODE: {len(feat)} rows, {len(feature_cols)} features (no target)")
        feat.to_csv(os.path.join(PROC_DIR, "features.csv"))
        return feat, feature_cols, target_cols, None

    if spx_intraday is not None and len(spx_intraday) > 1000:
        print("  Computing forward RV from intraday data")
        intra_ret = np.log(spx_intraday["Close"] / spx_intraday["Close"].shift(1)).dropna()
        daily_sq = intra_ret.groupby(intra_ret.index.date).apply(lambda x: np.sum(x**2))
        daily_sq.index = pd.to_datetime(daily_sq.index)
        daily_sq = daily_sq.reindex(close.index).fillna(method='ffill')
        sq_vals = daily_sq.values
    else:
        print("  Computing forward RV from daily closes")
        sq_vals = squared.values

    fwd_rv = []
    for i in range(len(sq_vals)):
        end = i + 1 + TARGET_WINDOW
        if end > len(sq_vals):
            fwd_rv.append(np.nan)
        else:
            fwd_rv.append(np.sqrt(np.sum(sq_vals[i+1:end]) * (252 / TARGET_WINDOW)))

    feat["fwd_rv_22d"] = fwd_rv
    feat = feat.dropna()

    train_mask = feat.index < SPLIT_DATE
    threshold = np.percentile(feat.loc[train_mask, "fwd_rv_22d"], SPIKE_PERCENTILE)
    feat["spike_label"] = (feat["fwd_rv_22d"] > threshold).astype(int)

    target_cols = ["fwd_rv_22d", "spike_label"]
    feature_cols = [c for c in feat.columns if c not in target_cols]

    print(f"\n  Dataset: {len(feat)} rows, {len(feature_cols)} features")
    print(f"  Spike threshold (p{SPIKE_PERCENTILE}): {threshold:.4f}")
    print(f"  Spikes: {feat['spike_label'].sum()} / {len(feat)}")

    feat.to_csv(os.path.join(PROC_DIR, "features.csv"))
    return feat, feature_cols, target_cols, threshold