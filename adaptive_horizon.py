"""
Adaptive Horizon Volatility Forecasting
=========================================
Switches between 22-day and 5-day prediction horizons based on
real-time shock detection. The economic logic: during normal markets,
monthly vol forecasts are appropriate for options trading. During
sudden dislocations, the next week matters more than the next month
because options delta-hedging accelerates and gamma exposure dominates.
"""

import pandas as pd, numpy as np, gc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROC_DIR = "data/processed"
SPLIT = "2020-01-01"
SEED = 42

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=0.3, reg_lambda=3.0,
    random_state=SEED, verbosity=0,
)


def load_and_prepare():
    """Load cached data and compute both 5-day and 22-day forward RV targets."""
    feat = pd.read_csv(f"{PROC_DIR}/features.csv", index_col=0, parse_dates=True)
    p1 = pd.read_csv(f"{PROC_DIR}/p1_latent.csv", index_col=0, parse_dates=True)
    p2 = pd.read_csv(f"{PROC_DIR}/p2_transformer.csv", index_col=0, parse_dates=True)
    p3 = pd.read_csv(f"{PROC_DIR}/p3_finbert.csv", index_col=0, parse_dates=True)
    p4 = pd.read_csv(f"{PROC_DIR}/p4_gat.csv", index_col=0, parse_dates=True)

    target_cols = ["fwd_rv_22d", "spike_label"]
    feature_cols = [c for c in feat.columns if c not in target_cols]

    
    log_ret = feat["log_return"] if "log_return" in feat.columns else np.log(1 + feat.get("return_1d", 0))
    squared = log_ret ** 2

    fwd_rv_5d = []
    for i in range(len(squared)):
        end = i + 1 + 5
        if end > len(squared):
            fwd_rv_5d.append(np.nan)
        else:
            fwd_rv_5d.append(np.sqrt(np.sum(squared.values[i+1:end]) * (252 / 5)))
    feat["fwd_rv_5d"] = fwd_rv_5d

   
    meta = feat[feature_cols].copy()
    meta = meta.join(p1, how="inner").join(p3, how="inner")
    meta = meta.join(p2, how="inner").join(p4, how="inner")
    meta["fwd_rv_22d"] = feat["fwd_rv_22d"]
    meta["fwd_rv_5d"] = feat["fwd_rv_5d"]
    meta["spike_label"] = feat["spike_label"]
    meta = meta.dropna()

    del p1, p2, p3, p4, feat
    gc.collect()

    meta_feat_cols = [c for c in meta.columns if c not in ["fwd_rv_22d", "fwd_rv_5d", "spike_label"]]
    return meta, meta_feat_cols


def detect_shock(meta, i, meta_feat_cols):
    
    row = meta.iloc[i]
    shock_score = 0.0

   
    if "abs_return" in meta.columns:
        abs_ret = row["abs_return"]
    elif "log_return" in meta.columns:
        abs_ret = abs(row["log_return"])
    else:
        abs_ret = abs(row.get("return_1d", 0))

    
    start = max(0, i - 60)
    if "abs_return" in meta.columns:
        trailing = meta["abs_return"].iloc[start:i]
    elif "log_return" in meta.columns:
        trailing = meta["log_return"].iloc[start:i].abs()
    else:
        trailing = meta.get("return_1d", pd.Series(dtype=float)).iloc[start:i].abs()

    if len(trailing) > 10:
        trail_mean = trailing.mean()
        trail_std = trailing.std()
        if trail_std > 0:
            z_score = (abs_ret - trail_mean) / trail_std
            if z_score > 3.0:
                shock_score = max(shock_score, min(1.0, (z_score - 3.0) / 3.0 + 0.5))
            elif z_score > 2.0:
                shock_score = max(shock_score, (z_score - 2.0) / 2.0 * 0.3)

    
    if "vix_level" in meta.columns and i > 0:
        vix_today = row["vix_level"]
        vix_yesterday = meta["vix_level"].iloc[i - 1]
        vix_jump = vix_today - vix_yesterday
        if vix_jump > 6:
            shock_score = max(shock_score, min(1.0, (vix_jump - 6) / 10 + 0.6))
        elif vix_jump > 4:
            shock_score = max(shock_score, (vix_jump - 4) / 4 * 0.4)

    
    if "vix_level" in meta.columns:
        vix = row["vix_level"]
        if vix > 40:
            shock_score = max(shock_score, 0.8)
        elif vix > 30:
            shock_score = max(shock_score, 0.4)

    
    if "vol_5d" in meta.columns:
        vol_5d = row["vol_5d"]
        if i > 60:
            trailing_vol5d = meta["vol_5d"].iloc[start:i]
            if len(trailing_vol5d) > 10:
                vol_z = (vol_5d - trailing_vol5d.mean()) / max(trailing_vol5d.std(), 1e-8)
                if vol_z > 3.0:
                    shock_score = max(shock_score, min(1.0, (vol_z - 3.0) / 3.0 + 0.4))

    return np.clip(shock_score, 0, 1)


def shock_decay(shock_scores, half_life=5):
    decayed = np.zeros_like(shock_scores)
    peak = 0.0
    for i in range(len(shock_scores)):
        peak = max(shock_scores[i], peak * np.exp(-np.log(2) / half_life))
        decayed[i] = peak
    return decayed


def get_har_weight(vix):
    if vix > 35: return 0.70
    elif vix > 25: return 0.60
    elif vix > 18: return 0.40
    else: return 0.30


def run_experiment():
    print("=" * 60)
    print("ADAPTIVE HORIZON VOL FORECASTING")
    print("=" * 60)

    meta, meta_feat_cols = load_and_prepare()
    print(f"  Data: {meta.shape[0]} rows, {len(meta_feat_cols)} features")
    print(f"  Has fwd_rv_5d: {meta['fwd_rv_5d'].notna().sum()} rows")
    print(f"  Has fwd_rv_22d: {meta['fwd_rv_22d'].notna().sum()} rows")

   
    train = meta[meta.index < SPLIT]
    test = meta[meta.index >= SPLIT]
    print(f"  Train: {len(train)}, Test: {len(test)}")

    

    
    har_22 = LinearRegression().fit(
        train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_22d"])
    har_22_pred = har_22.predict(test[["RV_1d", "RV_5d", "RV_22d"]])

    
    xgb_22 = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_22.fit(train[meta_feat_cols], train["fwd_rv_22d"])
    xgb_22_pred = xgb_22.predict(test[meta_feat_cols])

    
    xgb_5 = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_5.fit(train[meta_feat_cols], train["fwd_rv_5d"])
    xgb_5_pred = xgb_5.predict(test[meta_feat_cols])

    
    har_5 = LinearRegression().fit(
        train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_5d"])
    har_5_pred = har_5.predict(test[["RV_1d", "RV_5d", "RV_22d"]])

    print("\n  Models trained successfully")

    
    print("  Computing shock detection scores...")
    raw_shocks = []
    test_start_idx = list(meta.index).index(test.index[0])
    for i in range(len(test)):
        global_idx = test_start_idx + i
        score = detect_shock(meta, global_idx, meta_feat_cols)
        raw_shocks.append(score)

    raw_shocks = np.array(raw_shocks)
    shock_scores = shock_decay(raw_shocks, half_life=5)

    shock_days = np.sum(shock_scores > 0.3)
    print(f"  Shock days (score > 0.3): {shock_days}/{len(test)} ({100*shock_days/len(test):.1f}%)")
    print(f"  Mean shock score: {shock_scores.mean():.3f}")
    print(f"  Max shock score: {shock_scores.max():.3f}")

    
    vix = test["vix_level"].values
    har_w = np.array([get_har_weight(v) for v in vix])
    standard_ensemble = har_w * har_22_pred + (1 - har_w) * xgb_22_pred

    # Adaptive ensemble: blends 22-day and 5-day based on shock score
    # During calm: use 22-day ensemble 
    # During shock: transition toward 5-day predictions
    adaptive_22 = har_w * har_22_pred + (1 - har_w) * xgb_22_pred  # calm component
    adaptive_5 = har_w * har_5_pred + (1 - har_w) * xgb_5_pred      # shock component

    
    adaptive_pred = (1 - shock_scores) * adaptive_22 + shock_scores * adaptive_5

    # Evaluate against 22-day actual 
    actual_22 = test["fwd_rv_22d"].values
    actual_5 = test["fwd_rv_5d"].values
    spike = test["spike_label"].values

    print("\n" + "=" * 60)
    print("RESULTS: 22-DAY FORWARD VOL PREDICTION")
    print("=" * 60)
    print(f"\n  {'Model':<35s} {'MSE':>10s} {'AUC':>8s}")
    print(f"  {'-'*55}")

    models = {
        "HAR-RV (22d)": har_22_pred,
        "XGB (22d)": xgb_22_pred,
        "Standard Ensemble (22d)": standard_ensemble,
        "Adaptive Ensemble (22d+5d)": adaptive_pred,
    }

    for name, pred in models.items():
        mse = mean_squared_error(actual_22, pred)
        auc = roc_auc_score(spike, pred)
        print(f"  {name:<35s} {mse:>10.6f} {auc:>8.4f}")

   
    shock_mask = shock_scores > 0.3
    calm_mask = shock_scores <= 0.1

    if shock_mask.sum() > 20 and calm_mask.sum() > 20:
        print(f"\n" + "-" * 60)
        print(f"REGIME-SPECIFIC PERFORMANCE")
        print(f"-" * 60)

        print(f"\n  SHOCK PERIODS ({shock_mask.sum()} days):")
        print(f"  {'Model':<35s} {'MSE':>10s}")
        print(f"  {'-'*47}")
        for name, pred in models.items():
            mse = mean_squared_error(actual_22[shock_mask], pred[shock_mask])
            print(f"  {name:<35s} {mse:>10.6f}")

        print(f"\n  CALM PERIODS ({calm_mask.sum()} days):")
        print(f"  {'Model':<35s} {'MSE':>10s}")
        print(f"  {'-'*47}")
        for name, pred in models.items():
            mse = mean_squared_error(actual_22[calm_mask], pred[calm_mask])
            print(f"  {name:<35s} {mse:>10.6f}")

    
    print(f"\n" + "-" * 60)
    print(f"5-DAY VOL PREDICTION (shock periods only)")
    print(f"-" * 60)

    if shock_mask.sum() > 10:
        print(f"\n  {'Model':<35s} {'MSE (5d)':>10s}")
        print(f"  {'-'*47}")
        mse_har5 = mean_squared_error(actual_5[shock_mask], har_5_pred[shock_mask])
        mse_xgb5 = mean_squared_error(actual_5[shock_mask], xgb_5_pred[shock_mask])
        mse_adapt5 = mean_squared_error(actual_5[shock_mask], adaptive_5[shock_mask])
        print(f"  {'HAR-RV (5d)':<35s} {mse_har5:>10.6f}")
        print(f"  {'XGB (5d)':<35s} {mse_xgb5:>10.6f}")
        print(f"  {'Adaptive 5d component':<35s} {mse_adapt5:>10.6f}")

    
    print(f"\n" + "-" * 60)
    print(f"RESPONSE TIME ANALYSIS")
    print(f"-" * 60)
    print(f"  How quickly does each model respond to a vol spike?")

   
    vol_change = pd.Series(actual_22, index=test.index).pct_change(5)
    spike_onsets = vol_change[vol_change > 0.5].index

    if len(spike_onsets) > 0:
        print(f"  Found {len(spike_onsets)} spike onset periods")
        for onset in spike_onsets[:5]:
            idx = list(test.index).index(onset)
            if idx < 5 or idx + 10 > len(test):
                continue
            
            actual_before = actual_22[idx - 5]
            actual_after = actual_22[idx]
            std_before = standard_ensemble[idx - 5]
            std_after = standard_ensemble[idx]
            adp_before = adaptive_pred[idx - 5]
            adp_after = adaptive_pred[idx]

            actual_jump = (actual_after - actual_before) / max(actual_before, 0.01) * 100
            std_jump = (std_after - std_before) / max(std_before, 0.01) * 100
            adp_jump = (adp_after - adp_before) / max(adp_before, 0.01) * 100
            shock_at_onset = shock_scores[idx]

            print(f"\n  {onset.strftime('%Y-%m-%d')} (shock={shock_at_onset:.2f}):")
            print(f"    Actual vol change:    {actual_jump:+.1f}%")
            print(f"    Standard ensemble:    {std_jump:+.1f}%")
            print(f"    Adaptive ensemble:    {adp_jump:+.1f}%")

    
    print(f"\n  Generating plots...")

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Plot 1: Actual vs predictions
    ax = axes[0]
    ax.plot(test.index, actual_22, linewidth=1.0, color="black", alpha=0.8, label="Actual 22d RV")
    ax.plot(test.index, standard_ensemble, linewidth=0.8, color="#D4762C", alpha=0.75, linestyle="--", label="Standard Ensemble")
    ax.plot(test.index, adaptive_pred, linewidth=0.8, color="#2C5F8A", alpha=0.9, label="Adaptive Ensemble")
    ax.set_ylabel("22-Day Realized Vol", fontsize=10)
    ax.set_title("Adaptive vs Standard Ensemble Predictions", fontsize=13)
    ax.legend(fontsize=9)

    # Plot 2: Shock scores
    ax = axes[1]
    ax.fill_between(test.index, 0, shock_scores, alpha=0.6, color="#C44E52", label="Shock Score")
    ax.fill_between(test.index, 0, raw_shocks, alpha=0.3, color="#FF6B6B", label="Raw (pre-decay)")
    ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="Shock threshold")
    ax.set_ylabel("Shock Score", fontsize=10)
    ax.set_title("Real-Time Shock Detection (backward-looking only)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)

    # Plot 3: Horizon blend
    ax = axes[2]
    effective_horizon = 22 * (1 - shock_scores) + 5 * shock_scores
    ax.fill_between(test.index, 5, effective_horizon, alpha=0.4, color="#2C5F8A")
    ax.plot(test.index, effective_horizon, linewidth=0.7, color="#2C5F8A")
    ax.set_ylabel("Effective Horizon (days)", fontsize=10)
    ax.set_title("Adaptive Prediction Horizon", fontsize=13)
    ax.set_ylim(4, 23)
    ax.axhline(y=22, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Plot 4: Prediction error comparison
    ax = axes[3]
    std_error = (standard_ensemble - actual_22) ** 2
    adp_error = (adaptive_pred - actual_22) ** 2
    std_rolling = pd.Series(std_error, index=test.index).rolling(22).mean()
    adp_rolling = pd.Series(adp_error, index=test.index).rolling(22).mean()
    ax.plot(test.index, std_rolling, linewidth=0.9, color="#D4762C", linestyle="--", label="Standard MSE (22d avg)")
    ax.plot(test.index, adp_rolling, linewidth=0.9, color="#2C5F8A", label="Adaptive MSE (22d avg)")
    ax.set_ylabel("Rolling MSE", fontsize=10)
    ax.set_title("Rolling Prediction Error: Adaptive vs Standard", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlabel("Date", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{PROC_DIR}/report_adaptive_horizon.png", dpi=150)
    plt.close()
    print(f"  Plot saved to {PROC_DIR}/report_adaptive_horizon.png")

  
    std_mse = mean_squared_error(actual_22, standard_ensemble)
    adp_mse = mean_squared_error(actual_22, adaptive_pred)
    std_auc = roc_auc_score(spike, standard_ensemble)
    adp_auc = roc_auc_score(spike, adaptive_pred)

    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"\n  Standard Ensemble:  MSE={std_mse:.6f}  AUC={std_auc:.4f}")
    print(f"  Adaptive Ensemble:  MSE={adp_mse:.6f}  AUC={adp_auc:.4f}")
    print(f"  MSE change: {(adp_mse - std_mse)/std_mse*100:+.2f}%")
    print(f"  AUC change: {(adp_auc - std_auc):+.4f}")

    if adp_mse < std_mse:
        print(f"\n  CONCLUSION: Adaptive horizon IMPROVES vol forecasting.")
        print(f"  Shortening the horizon during shocks reduces prediction error")
        print(f"  because the model adapts faster to extreme market conditions.")
    else:
        print(f"\n  CONCLUSION: Adaptive horizon does NOT improve 22-day forecasting.")
        print(f"  The shock-driven 5-day predictions may be valuable for short-term")
        print(f"  trading but don't improve the 22-day target that the system was")
        print(f"  designed to predict.")

    print(f"\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETE")
    print(f"=" * 60)


if __name__ == "__main__":
    run_experiment()
