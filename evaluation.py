"""Evaluation: Diebold-Mariano, SHAP analysis, walk-forward validation."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb
import shap
from config import *


def diebold_mariano(actual, p1, p2, h=22):
    """
    Diebold-Mariano test for equal predictive accuracy.
    Tests whether squared forecast errors of two models differ
    significantly, accounting for serial correlation from
    overlapping 22-day prediction windows.
    """
    e1 = (actual - p1)**2
    e2 = (actual - p2)**2
    d = e1 - e2
    n = len(d)
    d_mean = d.mean()
    gamma = [np.cov(d[:-k] if k > 0 else d, d[k:] if k > 0 else d)[0, 1] for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return 0, 1.0
    dm = d_mean / np.sqrt(var_d)
    return dm, 2 * stats.t.sf(abs(dm), df=n - 1)


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


def run_evaluation(results):
    """Evaluate the regime-conditional XGBoost/HAR ensemble."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    m_test = results["m_test"]
    meta_feat_cols = results["meta_feat_cols"]
    preds = results["predictions"]
    actual = m_test["fwd_rv_22d"].values
    spike = m_test["spike_label"].values

    # Diebold-Mariano tests
    print("\nDIEBOLD-MARIANO TESTS")
    for n1, p1, n2, p2 in [
        ("HAR-RV", preds["har"], "Ensemble", preds["ensemble"]),
        ("HAR-RV", preds["har"], "XGBoost", preds["xgb"]),
    ]:
        dm, pval = diebold_mariano(actual, p1, p2)
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {n1} vs {n2}: DM={dm:.3f}, p={pval:.4f} {sig}")

    
    print("\nSHAP ANALYSIS")
    try:
        xgb_m = results["xgb_model"]
        explainer = shap.TreeExplainer(xgb_m)
        sv = explainer.shap_values(m_test[meta_feat_cols])

        plt.figure(figsize=(10, 14))
        shap.summary_plot(sv, m_test[meta_feat_cols], plot_type="bar", show=False, max_display=30)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PROC_DIR, "shap_overall.png"), dpi=150)
        plt.close()
        print(f"  SHAP plot saved to {os.path.join(PROC_DIR, 'shap_overall.png')}")

        fi = pd.Series(np.abs(sv).mean(axis=0), index=meta_feat_cols).sort_values(ascending=False)
        p5 = ["RV_1d", "RV_5d", "RV_22d"]
        p1 = [c for c in meta_feat_cols if c.startswith("latent") or c in
              ["recon_error", "vae_anomaly", "crisis_distance", "latent_velocity", "latent_acceleration"]]
        p2t = [c for c in meta_feat_cols if c.startswith("trans_")]
        p3 = [c for c in meta_feat_cols if c.startswith("finbert")]
        p3 = [c for c in p3 if c in fi.index]
        p4 = [c for c in meta_feat_cols if c.startswith("gat") or c == "mean_abs_corr"]
        p2m = [c for c in meta_feat_cols if c not in p5 + p1 + p2t + p3 + p4]

        total = fi.sum()
        print("\nPIPELINE CONTRIBUTIONS")
        for name, cols in [("P5 HAR-RV", p5), ("Market features", p2m), ("P2 Transformer", p2t),
                            ("P1 VAE", p1), ("P3 FinBERT", p3), ("P4 GAT", p4)]:
            val = fi[[c for c in cols if c in fi.index]].sum()
            print(f"  {name:20s}  SHAP={val:.4f}  ({100 * val / total:.1f}%)")
    except Exception as e:
        print(f"  SHAP failed (non-critical): {e}")

    return {}


def run_walk_forward(meta, meta_feat_cols):
    """Walk-forward CV using the regime-conditional XGBoost/HAR ensemble."""
    print("\n" + "=" * 60)
    print("WALK-FORWARD CV (Regime-Conditional Ensemble)")
    print("=" * 60)

    test_years = [y for y in range(2016, 2027) if meta[meta.index.year == y].shape[0] > 50]
    wf = []

    for year in test_years:
        tr = meta[meta.index.year < year]
        te = meta[meta.index.year == year]
        if len(tr) < 500 or len(te) < 30:
            continue

        h = LinearRegression().fit(tr[["RV_1d", "RV_5d", "RV_22d"]], tr["fwd_rv_22d"])
        hp = h.predict(te[["RV_1d", "RV_5d", "RV_22d"]])

        xg = xgb.XGBRegressor(**XGB_PARAMS)
        xg.fit(tr[meta_feat_cols], tr["fwd_rv_22d"])
        xp = xg.predict(te[meta_feat_cols])

        vix_vals = te["vix_level"].values if "vix_level" in te.columns else np.full(len(te), 18)
        har_w = np.array([get_regime_weight(v) for v in vix_vals])
        ens = har_w * hp + (1 - har_w) * xp

        h_auc = roc_auc_score(te["spike_label"], hp) if te["spike_label"].nunique() > 1 else np.nan
        x_auc = roc_auc_score(te["spike_label"], xp) if te["spike_label"].nunique() > 1 else np.nan
        e_auc = roc_auc_score(te["spike_label"], ens) if te["spike_label"].nunique() > 1 else np.nan

        wf.append({
            "Year": year,
            "HAR_MSE": mean_squared_error(te["fwd_rv_22d"], hp), "HAR_AUC": h_auc,
            "XGB_MSE": mean_squared_error(te["fwd_rv_22d"], xp), "XGB_AUC": x_auc,
            "Ens_MSE": mean_squared_error(te["fwd_rv_22d"], ens), "Ens_AUC": e_auc,
        })

        if not np.isnan(h_auc):
            avg_w = np.mean(har_w)
            print(f"  {year}: HAR={h_auc:.3f} | XGB={x_auc:.3f} | Ens={e_auc:.3f} (avg HAR weight: {avg_w:.0%})")
        else:
            print(f"  {year}: no spikes")

    wf_df = pd.DataFrame(wf).set_index("Year")
    valid = wf_df.dropna(subset=["HAR_AUC"])
    print(f"\nAverages:")
    print(f"  HAR-RV:   MSE={valid['HAR_MSE'].mean():.6f}  AUC={valid['HAR_AUC'].mean():.3f}")
    print(f"  XGBoost:  MSE={valid['XGB_MSE'].mean():.6f}  AUC={valid['XGB_AUC'].mean():.3f}")
    print(f"  Ensemble: MSE={valid['Ens_MSE'].mean():.6f}  AUC={valid['Ens_AUC'].mean():.3f}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for col, label in [("HAR_MSE", "HAR-RV"), ("XGB_MSE", "XGBoost"), ("Ens_MSE", "Ensemble")]:
            axes[0].plot(wf_df.index, wf_df[col], 'o-', label=label, markersize=4)
        axes[0].set_title("MSE by Year"); axes[0].legend()
        for col, label in [("HAR_AUC", "HAR-RV"), ("XGB_AUC", "XGBoost"), ("Ens_AUC", "Ensemble")]:
            d = wf_df[col].dropna()
            axes[1].plot(d.index, d.values, 'o-', label=label, markersize=4)
        axes[1].set_title("AUC by Year"); axes[1].legend()
        plt.suptitle("Walk-Forward CV (Regime-Conditional Ensemble)")
        plt.tight_layout()
        plt.savefig(os.path.join(PROC_DIR, "walk_forward.png"), dpi=150)
        plt.close()
        print(f"  Walk-forward plot saved")
    except Exception as e:
        print(f"  Walk-forward plotting failed: {e}")

    return wf_df
