"""Meta-model: XGBoost + HAR-RV regime-conditional ensemble."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb
from config import *


def get_regime_weight(vix_level):
    """Return HAR-RV weight based on VIX regime."""
    if vix_level > 35:
        return ENSEMBLE_HAR_WEIGHT_CRISIS
    elif vix_level > 25:
        return ENSEMBLE_HAR_WEIGHT_STRESS
    elif vix_level > 18:
        return ENSEMBLE_HAR_WEIGHT_NORMAL
    else:
        return ENSEMBLE_HAR_WEIGHT_CALM


def train_meta(feat, feature_cols, latent_df, trans_df, text_df, gat_df, har_pred):
    print("\n" + "=" * 60)
    print("META-MODEL: XGBoost + HAR-RV Regime-Conditional Ensemble")
    print("=" * 60)

    target_cols = ["fwd_rv_22d", "spike_label"]
    base_cols = [c for c in feature_cols if c not in target_cols]

    # Merge all pipeline outputs
    meta = feat[base_cols].copy()
    meta = meta.join(latent_df, how="inner")
    meta = meta.join(text_df, how="inner")
    meta = meta.join(trans_df, how="inner")
    meta = meta.join(gat_df, how="inner")
    meta = meta.join(feat[target_cols], how="inner").dropna()

    meta_feat_cols = [c for c in meta.columns if c not in target_cols]
    print(f"  Meta dataset: {meta.shape[0]} rows, {len(meta_feat_cols)} features")

    
    p5_cols = ["RV_1d", "RV_5d", "RV_22d"]
    p1_cols = [c for c in meta_feat_cols if c.startswith("latent") or c in
               ["recon_error", "vae_anomaly", "crisis_distance", "latent_velocity", "latent_acceleration"]]
    p2t_cols = [c for c in meta_feat_cols if c.startswith("trans_")]
    p3_cols = [c for c in meta_feat_cols if c.startswith("finbert")]
    p3_cols = [c for c in p3_cols if c in meta_feat_cols]
    p4_cols = [c for c in meta_feat_cols if c.startswith("gat") or c == "mean_abs_corr"]
    p2m_cols = [c for c in meta_feat_cols if c not in p5_cols + p1_cols + p2t_cols + p3_cols + p4_cols]

    dims = {"HAR-RV": len(p5_cols), "Market": len(p2m_cols), "Transformer": len(p2t_cols),
            "VAE": len(p1_cols), "FinBERT": len(p3_cols), "GAT": len(p4_cols)}
    print(f"  Pipeline dimensions: {dims}")

    # Split
    m_train = meta[meta.index < SPLIT_DATE]
    m_test = meta[meta.index >= SPLIT_DATE]

    # HAR-RV baseline
    har = LinearRegression().fit(m_train[["RV_1d", "RV_5d", "RV_22d"]], m_train["fwd_rv_22d"])
    hp = har.predict(m_test[["RV_1d", "RV_5d", "RV_22d"]])

    # XGBoost on all pipeline features
    xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_m.fit(m_train[meta_feat_cols], m_train["fwd_rv_22d"])
    xp = xgb_m.predict(m_test[meta_feat_cols])

    # Regime-conditional blending
    vix_test = m_test["vix_level"].values if "vix_level" in m_test.columns else np.full(len(m_test), 18)
    har_weights = np.array([get_regime_weight(v) for v in vix_test])
    ensemble = har_weights * hp + (1 - har_weights) * xp

    actual = m_test["fwd_rv_22d"].values
    spike = m_test["spike_label"].values

    print(f"\n{'=' * 60}")
    print("RESULTS (test: 2020+)")
    print(f"{'=' * 60}")
    print(f"  HAR-RV                 MSE={mean_squared_error(actual, hp):.6f}  AUC={roc_auc_score(spike, hp):.4f}")
    print(f"  XGBoost (all features) MSE={mean_squared_error(actual, xp):.6f}  AUC={roc_auc_score(spike, xp):.4f}")
    print(f"  Ensemble (regime-cond) MSE={mean_squared_error(actual, ensemble):.6f}  AUC={roc_auc_score(spike, ensemble):.4f}")

    print(f"\n  Regime weights:")
    print(f"    Crisis (VIX>35): HAR={ENSEMBLE_HAR_WEIGHT_CRISIS:.0%}")
    print(f"    Stress (VIX>25): HAR={ENSEMBLE_HAR_WEIGHT_STRESS:.0%}")
    print(f"    Normal (VIX>18): HAR={ENSEMBLE_HAR_WEIGHT_NORMAL:.0%}")
    print(f"    Calm   (VIX<=18): HAR={ENSEMBLE_HAR_WEIGHT_CALM:.0%}")

    return {
        "m_train": m_train, "m_test": m_test,
        "meta_feat_cols": meta_feat_cols, "meta": meta,
        "predictions": {"har": hp, "xgb": xp, "ensemble": ensemble},
        "xgb_model": xgb_m,
    }
