"""

Adaptive 2-Horizon (22d+5d) model as the primary ensemble.

"""

import numpy as np, pandas as pd, os, gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy import stats
import xgboost as xgb

PROC_DIR = "data/processed"
SPLIT = "2020-01-01"
SEED = 42

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=0.3, reg_lambda=3.0,
    random_state=SEED, verbosity=0,
)


def load_data():
    feat = pd.read_csv(f"{PROC_DIR}/features.csv", index_col=0, parse_dates=True)
    p1 = pd.read_csv(f"{PROC_DIR}/p1_latent.csv", index_col=0, parse_dates=True)
    p2 = pd.read_csv(f"{PROC_DIR}/p2_transformer.csv", index_col=0, parse_dates=True)
    p3 = pd.read_csv(f"{PROC_DIR}/p3_finbert.csv", index_col=0, parse_dates=True)
    p4 = pd.read_csv(f"{PROC_DIR}/p4_gat.csv", index_col=0, parse_dates=True)

    target_cols = ["fwd_rv_22d", "spike_label"]
    feature_cols = [c for c in feat.columns if c not in target_cols]

    # Compute 5-day forward RV
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


def diebold_mariano(actual, p1, p2, h=22):
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


def get_har_weight(vix):
    if vix > 35: return 0.70
    elif vix > 25: return 0.60
    elif vix > 18: return 0.40
    else: return 0.30


def detect_shock(meta, i):
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
            if z_score > 4.0:
                shock_score = max(shock_score, min(1.0, (z_score - 3.0) / 4.0 + 0.6))
            elif z_score > 3.0:
                shock_score = max(shock_score, min(1.0, (z_score - 3.0) / 3.0 + 0.5))
            elif z_score > 2.0:
                shock_score = max(shock_score, (z_score - 2.0) / 2.0 * 0.3)

    if "vix_level" in meta.columns and i > 0:
        vix_today = row["vix_level"]
        vix_yesterday = meta["vix_level"].iloc[i - 1]
        vix_jump = vix_today - vix_yesterday
        if vix_jump > 10:
            shock_score = max(shock_score, min(1.0, (vix_jump - 6) / 10 + 0.7))
        elif vix_jump > 6:
            shock_score = max(shock_score, min(1.0, (vix_jump - 6) / 10 + 0.6))
        elif vix_jump > 4:
            shock_score = max(shock_score, (vix_jump - 4) / 4 * 0.4)

    if "vix_level" in meta.columns:
        vix = row["vix_level"]
        if vix > 50:
            shock_score = max(shock_score, 0.95)
        elif vix > 40:
            shock_score = max(shock_score, 0.8)
        elif vix > 30:
            shock_score = max(shock_score, 0.4)

    if "vol_5d" in meta.columns and i > 60:
        vol_5d = row["vol_5d"]
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


def run_adaptive_seed(meta, meta_feat_cols, seed):
    """Run full adaptive 2-horizon model for a single seed."""
    train = meta[meta.index < SPLIT]
    test = meta[meta.index >= SPLIT]

    # HAR-RV models
    har_22 = LinearRegression().fit(train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_22d"])
    har_5 = LinearRegression().fit(train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_5d"])
    har_22_pred = har_22.predict(test[["RV_1d", "RV_5d", "RV_22d"]])
    har_5_pred = har_5.predict(test[["RV_1d", "RV_5d", "RV_22d"]])

    # XGBoost models
    params = {**XGB_PARAMS, "random_state": seed}
    xgb_22 = xgb.XGBRegressor(**params)
    xgb_22.fit(train[meta_feat_cols], train["fwd_rv_22d"])
    xgb_22_pred = xgb_22.predict(test[meta_feat_cols])

    xgb_5 = xgb.XGBRegressor(**params)
    xgb_5.fit(train[meta_feat_cols], train["fwd_rv_5d"])
    xgb_5_pred = xgb_5.predict(test[meta_feat_cols])

    
    vix = test["vix_level"].values
    har_w = np.array([get_har_weight(v) for v in vix])

    # Standard ensemble (22d only)
    standard = har_w * har_22_pred + (1 - har_w) * xgb_22_pred

    
    test_start_idx = list(meta.index).index(test.index[0])
    raw_shocks = []
    for i in range(len(test)):
        raw_shocks.append(detect_shock(meta, test_start_idx + i))
    shock_scores = shock_decay(np.array(raw_shocks), half_life=5)

    # Adaptive 2-horizon
    ens_22 = har_w * har_22_pred + (1 - har_w) * xgb_22_pred
    ens_5 = har_w * har_5_pred + (1 - har_w) * xgb_5_pred
    adaptive = (1 - shock_scores) * ens_22 + shock_scores * ens_5

    actual = test["fwd_rv_22d"].values
    spike = test["spike_label"].values

    return {
        "har_mse": mean_squared_error(actual, har_22_pred),
        "har_auc": roc_auc_score(spike, har_22_pred),
        "std_mse": mean_squared_error(actual, standard),
        "std_auc": roc_auc_score(spike, standard),
        "adp_mse": mean_squared_error(actual, adaptive),
        "adp_auc": roc_auc_score(spike, adaptive),
        "har_pred": har_22_pred, "std_pred": standard, "adaptive": adaptive,
        "actual": actual, "spike": spike, "test": test,
        "xgb_22": xgb_22, "shock_scores": shock_scores,
        "meta_feat_cols": meta_feat_cols,
    }


# Figures

def fig_walk_forward(meta, meta_feat_cols):
    print("\n[1/10] Walk-Forward AUC by Year...")
    years = [y for y in range(2016, 2027) if meta[meta.index.year == y].shape[0] > 50]
    results = []

    for year in years:
        tr = meta[meta.index.year < year]
        te = meta[meta.index.year == year]
        if len(tr) < 500 or len(te) < 30 or te["spike_label"].nunique() < 2:
            continue

        h22 = LinearRegression().fit(tr[["RV_1d", "RV_5d", "RV_22d"]], tr["fwd_rv_22d"])
        h5 = LinearRegression().fit(tr[["RV_1d", "RV_5d", "RV_22d"]], tr["fwd_rv_5d"])
        hp22 = h22.predict(te[["RV_1d", "RV_5d", "RV_22d"]])
        hp5 = h5.predict(te[["RV_1d", "RV_5d", "RV_22d"]])

        xg22 = xgb.XGBRegressor(**XGB_PARAMS); xg22.fit(tr[meta_feat_cols], tr["fwd_rv_22d"])
        xg5 = xgb.XGBRegressor(**XGB_PARAMS); xg5.fit(tr[meta_feat_cols], tr["fwd_rv_5d"])
        xp22 = xg22.predict(te[meta_feat_cols])
        xp5 = xg5.predict(te[meta_feat_cols])

        vix = te["vix_level"].values
        har_w = np.array([get_har_weight(v) for v in vix])
        std_ens = har_w * hp22 + (1 - har_w) * xp22

        # Shock detection for this fold
        te_start = list(meta.index).index(te.index[0])
        shocks = shock_decay(np.array([detect_shock(meta, te_start + i) for i in range(len(te))]))
        e22 = har_w * hp22 + (1 - har_w) * xp22
        e5 = har_w * hp5 + (1 - har_w) * xp5
        adp = (1 - shocks) * e22 + shocks * e5

        results.append({
            "Year": year,
            "HAR-RV": roc_auc_score(te["spike_label"], hp22),
            "Standard": roc_auc_score(te["spike_label"], std_ens),
            "Adaptive": roc_auc_score(te["spike_label"], adp),
        })

    df = pd.DataFrame(results).set_index("Year")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df)); w = 0.25
    ax.bar(x - w, df["HAR-RV"], w, label="HAR-RV", color="#8B9DAF")
    ax.bar(x, df["Standard"], w, label="Standard Ensemble", color="#5B7AA5")
    ax.bar(x + w, df["Adaptive"], w, label="Adaptive 2-Horizon", color="#2C5F8A")
    ax.set_xticks(x); ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Walk-Forward Spike Detection AUC by Year", fontsize=14)
    ax.legend(fontsize=11); ax.set_ylim(0.4, 1.0)
    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_walk_forward.png", dpi=150); plt.close()
    print(f"  Avg HAR-RV: {df['HAR-RV'].mean():.3f}, Avg Adaptive: {df['Adaptive'].mean():.3f}")
    return df


def fig_shap_pipeline(meta_feat_cols, xgb_model, test):
    print("\n[2/10] SHAP Pipeline Contributions...")
    try:
        import shap
    except ImportError:
        print("  SHAP not installed"); return

    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(test[meta_feat_cols])
    fi = pd.Series(np.abs(sv).mean(axis=0), index=meta_feat_cols)
    total = fi.sum()

    p5 = ["RV_1d", "RV_5d", "RV_22d"]
    p1 = [c for c in meta_feat_cols if c.startswith("latent") or c in ["recon_error", "vae_anomaly", "crisis_distance", "latent_velocity", "latent_acceleration"]]
    p2t = [c for c in meta_feat_cols if c.startswith("trans_")]
    p3 = [c for c in meta_feat_cols if c.startswith("finbert")]; p3 = [c for c in p3 if c in meta_feat_cols]
    p4 = [c for c in meta_feat_cols if c.startswith("gat") or c == "mean_abs_corr"]
    p2m = [c for c in meta_feat_cols if c not in p5 + p1 + p2t + p3 + p4]

    pipeline_shap = {}
    for name, cols in [("HAR-RV\n(P5)", p5), ("Market\nFeatures", p2m), ("Transformer\n(P2)", p2t),
                        ("VAE\n(P1)", p1), ("FinBERT\n(P3)", p3), ("GAT\n(P4)", p4)]:
        pipeline_shap[name] = fi[[c for c in cols if c in fi.index]].sum()

    names = list(pipeline_shap.keys()); values = list(pipeline_shap.values())
    pcts = [v / total * 100 for v in values]
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]; values = [values[i] for i in order]; pcts = [pcts[i] for i in order]
    colors = ["#2C5F8A", "#5B7AA5", "#8B9DAF", "#A8BFD0", "#C5D5E3", "#DDE7EF"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.barh(range(len(names)), values, color=colors)
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel("Mean |SHAP|"); ax1.set_title("Pipeline SHAP Contributions"); ax1.invert_yaxis()
    for i, (v, p) in enumerate(zip(values, pcts)):
        ax1.text(v + total * 0.01, i, f"{p:.1f}%", va="center", fontsize=10)
    ax2.pie(values, labels=names, autopct="%1.1f%%", colors=colors, textprops={"fontsize": 9})
    ax2.set_title("Pipeline Share of Total SHAP")
    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_shap_pipelines.png", dpi=150); plt.close()
    for n, p in zip(names, pcts): print(f"    {n.replace(chr(10), ' '):20s}  {p:.1f}%")


def fig_seed_robustness(meta, meta_feat_cols):
    print("\n[3/10] Seed Robustness (5 seeds)...")
    seeds = [42, 43, 44, 45, 46]
    results = []
    for s in seeds:
        r = run_adaptive_seed(meta, meta_feat_cols, s)
        results.append({"Seed": s, "HAR AUC": r["har_auc"], "Std AUC": r["std_auc"], "Adp AUC": r["adp_auc"],
                         "HAR MSE": r["har_mse"], "Std MSE": r["std_mse"], "Adp MSE": r["adp_mse"]})
    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(seeds)); w = 0.25
    ax1.bar(x - w, df["HAR AUC"], w, label="HAR-RV", color="#8B9DAF")
    ax1.bar(x, df["Std AUC"], w, label="Standard", color="#5B7AA5")
    ax1.bar(x + w, df["Adp AUC"], w, label="Adaptive", color="#2C5F8A")
    ax1.set_xticks(x); ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.set_ylabel("AUC"); ax1.set_title("Spike Detection AUC Across Seeds"); ax1.legend(); ax1.set_ylim(0.75, 0.90)
    ax2.bar(x - w, df["HAR MSE"], w, label="HAR-RV", color="#8B9DAF")
    ax2.bar(x, df["Std MSE"], w, label="Standard", color="#5B7AA5")
    ax2.bar(x + w, df["Adp MSE"], w, label="Adaptive", color="#2C5F8A")
    ax2.set_xticks(x); ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylabel("MSE"); ax2.set_title("Regression MSE Across Seeds"); ax2.legend()
    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_seed_robustness.png", dpi=150); plt.close()
    print(f"  Adaptive AUC: {df['Adp AUC'].mean():.4f} +/- {df['Adp AUC'].std():.4f}")
    print(f"  Adaptive MSE: {df['Adp MSE'].mean():.6f} +/- {df['Adp MSE'].std():.6f}")
    return df


def fig_predictions_vs_actual(results):
    print("\n[4/10] Predictions vs Actual...")
    test = results["test"]; actual = results["actual"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test.index, actual, linewidth=1.0, color="black", alpha=0.8, label="Actual RV")
    ax.plot(test.index, results["adaptive"], linewidth=0.8, color="#2C5F8A", alpha=0.85, label="Adaptive 2-Horizon")
    ax.plot(test.index, results["har_pred"], linewidth=0.8, color="#D4762C", alpha=0.7, linestyle="--", label="HAR-RV")
    spike_mask = results["spike"] == 1
    for i in range(len(spike_mask)):
        if spike_mask[i]:
            ax.axvspan(test.index[i], test.index[min(i + 1, len(test) - 1)], alpha=0.08, color="red")
    ax.set_ylabel("22-Day Realized Volatility"); ax.set_title("Adaptive 2-Horizon Predictions vs Actual (2020-2026)")
    ax.legend(); plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_predictions_ts.png", dpi=150); plt.close()


def fig_calibration(results):
    print("\n[5/10] Regression Calibration...")
    actual = results["actual"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pred, color in [("HAR-RV", results["har_pred"], "#8B9DAF"),
                               ("Standard", results["std_pred"], "#5B7AA5"),
                               ("Adaptive", results["adaptive"], "#2C5F8A")]:
        bins = np.quantile(pred, np.linspace(0, 1, 11)); bm, am = [], []
        for j in range(len(bins) - 1):
            mask = (pred >= bins[j]) & (pred < bins[j + 1])
            if mask.sum() > 0: bm.append(pred[mask].mean()); am.append(actual[mask].mean())
        ax.plot(bm, am, "o-", label=name, color=color, markersize=5)
    ax.plot([0, 0.6], [0, 0.6], "k--", alpha=0.3, label="Perfect")
    ax.set_xlabel("Predicted RV"); ax.set_ylabel("Actual RV"); ax.set_title("Regression Calibration")
    ax.legend(); plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_calibration.png", dpi=150); plt.close()


def fig_top_features(meta_feat_cols, xgb_model, test):
    print("\n[6/10] Top 15 Features by SHAP...")
    try:
        import shap
    except ImportError: return
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(test[meta_feat_cols])
    fi = pd.Series(np.abs(sv).mean(axis=0), index=meta_feat_cols).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(fi)), fi.values, color="#2C5F8A")
    ax.set_yticks(range(len(fi))); ax.set_yticklabels(fi.index, fontsize=9)
    ax.set_xlabel("Mean |SHAP|"); ax.set_title("Top 15 Features by SHAP Importance")
    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_top_features.png", dpi=150); plt.close()
    print(f"  Top feature: {fi.index[-1]} ({fi.values[-1]:.4f})")


def fig_regime_analysis(results):
    print("\n[7/10] Regime Analysis...")
    test = results["test"]; actual = results["actual"]
    vix = test["vix_level"].values
    regimes = {"Calm (VIX<=18)": vix <= 18, "Normal (18<VIX<=25)": (vix > 18) & (vix <= 25),
               "Stress (25<VIX<=35)": (vix > 25) & (vix <= 35), "Crisis (VIX>35)": vix > 35}
    regime_data = []
    for name, mask in regimes.items():
        if mask.sum() < 10: continue
        har_mse = mean_squared_error(actual[mask], results["har_pred"][mask])
        adp_mse = mean_squared_error(actual[mask], results["adaptive"][mask])
        regime_data.append({"Regime": name, "Days": mask.sum(), "HAR MSE": har_mse, "Adp MSE": adp_mse,
                            "Improvement": (har_mse - adp_mse) / har_mse * 100})
    df = pd.DataFrame(regime_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(df)); w = 0.3
    ax1.bar(x - w/2, df["HAR MSE"], w, label="HAR-RV", color="#8B9DAF")
    ax1.bar(x + w/2, df["Adp MSE"], w, label="Adaptive", color="#2C5F8A")
    ax1.set_xticks(x); ax1.set_xticklabels(df["Regime"], fontsize=9)
    ax1.set_ylabel("MSE"); ax1.set_title("MSE by VIX Regime"); ax1.legend()
    colors = ["#2C5F8A" if v > 0 else "#C44E52" for v in df["Improvement"]]
    ax2.bar(range(len(df)), df["Improvement"], color=colors)
    ax2.set_xticks(range(len(df))); ax2.set_xticklabels(df["Regime"], fontsize=9)
    ax2.set_ylabel("MSE Improvement (%)"); ax2.set_title("Adaptive Improvement Over HAR-RV"); ax2.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_regime_analysis.png", dpi=150); plt.close()
    for _, row in df.iterrows():
        print(f"  {row['Regime']:25s}  {row['Days']:4.0f} days  Improvement={row['Improvement']:+.1f}%")


def fig_ablation(meta, meta_feat_cols):
    print("\n[8/10] Pipeline Ablation Study...")
    p1_cols = [c for c in meta_feat_cols if c.startswith("latent") or c in ["recon_error", "vae_anomaly", "crisis_distance", "latent_velocity", "latent_acceleration"]]
    p2t_cols = [c for c in meta_feat_cols if c.startswith("trans_")]
    p3_cols = [c for c in meta_feat_cols if c.startswith("finbert")]; p3_cols = [c for c in p3_cols if c in meta_feat_cols]
    p4_cols = [c for c in meta_feat_cols if c.startswith("gat") or c == "mean_abs_corr"]

    pipelines = {"Full (all pipelines)": [], "Without VAE (P1)": p1_cols, "Without Transformer (P2)": p2t_cols,
                 "Without FinBERT (P3)": p3_cols, "Without GAT (P4)": p4_cols}

    train = meta[meta.index < SPLIT]; test = meta[meta.index >= SPLIT]
    actual = test["fwd_rv_22d"].values; spike = test["spike_label"].values
    har = LinearRegression().fit(train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_22d"])
    hp = har.predict(test[["RV_1d", "RV_5d", "RV_22d"]])
    har5 = LinearRegression().fit(train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_5d"])
    hp5 = har5.predict(test[["RV_1d", "RV_5d", "RV_22d"]])

    test_start = list(meta.index).index(test.index[0])
    shocks = shock_decay(np.array([detect_shock(meta, test_start + i) for i in range(len(test))]))
    vix = test["vix_level"].values; har_w = np.array([get_har_weight(v) for v in vix])

    ablation_results = []
    for name, remove_cols in pipelines.items():
        keep = [c for c in meta_feat_cols if c not in remove_cols]
        xg22 = xgb.XGBRegressor(**XGB_PARAMS); xg22.fit(train[keep], train["fwd_rv_22d"])
        xg5 = xgb.XGBRegressor(**XGB_PARAMS); xg5.fit(train[keep], train["fwd_rv_5d"])
        e22 = har_w * hp + (1 - har_w) * xg22.predict(test[keep])
        e5 = har_w * hp5 + (1 - har_w) * xg5.predict(test[keep])
        adp = (1 - shocks) * e22 + shocks * e5
        auc = roc_auc_score(spike, adp); mse = mean_squared_error(actual, adp)
        ablation_results.append({"Config": name, "AUC": auc, "MSE": mse})

    df = pd.DataFrame(ablation_results); full_auc = df.iloc[0]["AUC"]
    df["AUC Drop"] = full_auc - df["AUC"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2C5F8A"] + ["#C44E52" if d > 0 else "#4CAF50" for d in df["AUC Drop"].iloc[1:]]
    ax1.barh(range(len(df)), df["AUC"], color=colors)
    ax1.set_yticks(range(len(df))); ax1.set_yticklabels(df["Config"], fontsize=10)
    ax1.set_xlabel("Adaptive Ensemble AUC"); ax1.set_title("Pipeline Ablation"); ax1.axvline(x=full_auc, color="gray", linestyle="--", alpha=0.5)
    for i, (auc, drop) in enumerate(zip(df["AUC"], df["AUC Drop"])):
        label = f"{auc:.4f}" if i == 0 else f"{auc:.4f} ({drop:+.4f})"
        ax1.text(auc + 0.001, i, label, va="center", fontsize=9)
    ax1.invert_yaxis()
    drop_df = df.iloc[1:]; drop_colors = ["#C44E52" if d > 0 else "#4CAF50" for d in drop_df["AUC Drop"]]
    ax2.bar(range(len(drop_df)), drop_df["AUC Drop"], color=drop_colors)
    ax2.set_xticks(range(len(drop_df)))
    ax2.set_xticklabels([c.replace("Without ", "").replace(" (P1)", "\n(P1)").replace(" (P2)", "\n(P2)").replace(" (P3)", "\n(P3)").replace(" (P4)", "\n(P4)") for c in drop_df["Config"]], fontsize=9)
    ax2.set_ylabel("AUC Drop"); ax2.set_title("AUC Impact of Removing Each Pipeline"); ax2.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_ablation.png", dpi=150); plt.close()
    for _, row in df.iterrows(): print(f"    {row['Config']:30s}  AUC={row['AUC']:.4f}")


def fig_lr_sensitivity(meta, meta_feat_cols):
    print("\n[9/10] Learning Rate Sensitivity...")
    train = meta[meta.index < SPLIT]; test = meta[meta.index >= SPLIT]
    actual = test["fwd_rv_22d"].values; spike = test["spike_label"].values
    har = LinearRegression().fit(train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_22d"])
    hp = har.predict(test[["RV_1d", "RV_5d", "RV_22d"]])
    har5 = LinearRegression().fit(train[["RV_1d", "RV_5d", "RV_22d"]], train["fwd_rv_5d"])
    hp5 = har5.predict(test[["RV_1d", "RV_5d", "RV_22d"]])
    test_start = list(meta.index).index(test.index[0])
    shocks = shock_decay(np.array([detect_shock(meta, test_start + i) for i in range(len(test))]))
    vix = test["vix_level"].values; har_w = np.array([get_har_weight(v) for v in vix])

    lrs = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]; lr_results = []
    for lr in lrs:
        params = {**XGB_PARAMS, "learning_rate": lr}
        xg22 = xgb.XGBRegressor(**params); xg22.fit(train[meta_feat_cols], train["fwd_rv_22d"])
        xg5 = xgb.XGBRegressor(**params); xg5.fit(train[meta_feat_cols], train["fwd_rv_5d"])
        e22 = har_w * hp + (1 - har_w) * xg22.predict(test[meta_feat_cols])
        e5 = har_w * hp5 + (1 - har_w) * xg5.predict(test[meta_feat_cols])
        adp = (1 - shocks) * e22 + shocks * e5
        lr_results.append({"lr": lr, "AUC": roc_auc_score(spike, adp), "MSE": mean_squared_error(actual, adp)})

    df = pd.DataFrame(lr_results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(df["lr"], df["AUC"], "o-", color="#2C5F8A"); ax1.set_xscale("log")
    ax1.axhline(y=roc_auc_score(spike, hp), color="#8B9DAF", linestyle="--", label="HAR-RV")
    best_lr = df.loc[df["AUC"].idxmax(), "lr"]
    ax1.axvline(x=best_lr, color="green", linestyle=":", alpha=0.5, label=f"Best lr={best_lr}")
    ax1.set_xlabel("Learning Rate"); ax1.set_ylabel("Adaptive AUC"); ax1.set_title("AUC vs Learning Rate"); ax1.legend()
    ax2.plot(df["lr"], df["MSE"], "o-", color="#2C5F8A"); ax2.set_xscale("log")
    ax2.axhline(y=mean_squared_error(actual, hp), color="#8B9DAF", linestyle="--", label="HAR-RV")
    ax2.set_xlabel("Learning Rate"); ax2.set_ylabel("Adaptive MSE"); ax2.set_title("MSE vs Learning Rate"); ax2.legend()
    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_lr_sensitivity.png", dpi=150); plt.close()
    print(f"  Best lr: {best_lr} (AUC={df.loc[df['AUC'].idxmax(), 'AUC']:.4f})")


def fig_architecture(meta_feat_cols):
    print("\n[10/10] Architecture Diagram...")
    p5 = ["RV_1d", "RV_5d", "RV_22d"]
    p1 = [c for c in meta_feat_cols if c.startswith("latent") or c in ["recon_error", "vae_anomaly", "crisis_distance", "latent_velocity", "latent_acceleration"]]
    p2t = [c for c in meta_feat_cols if c.startswith("trans_")]
    p3 = [c for c in meta_feat_cols if c.startswith("finbert")]; p3 = [c for c in p3 if c in meta_feat_cols]
    p4 = [c for c in meta_feat_cols if c.startswith("gat") or c == "mean_abs_corr"]
    p2m = [c for c in meta_feat_cols if c not in p5 + p1 + p2t + p3 + p4]

    fig, ax = plt.subplots(figsize=(14, 9)); ax.set_xlim(0, 14); ax.set_ylim(0, 11); ax.axis("off")
    ax.text(7, 10.5, "Adaptive 2-Horizon System Architecture", ha="center", fontsize=16, fontweight="bold")

    
    ax.add_patch(plt.Rectangle((0.5, 8.5), 3, 1, facecolor="#DDE7EF", edgecolor="#2C5F8A", linewidth=1.5))
    ax.text(2, 9.2, "Raw Data", ha="center", fontsize=11, fontweight="bold")
    ax.text(2, 8.8, "SPX, VIX, Cross-Assets, NYT, FOMC", ha="center", fontsize=8)

    
    ax.add_patch(plt.Rectangle((0.5, 6.8), 3, 1, facecolor="#C5D5E3", edgecolor="#2C5F8A", linewidth=1.5))
    ax.text(2, 7.5, "Feature Engineering", ha="center", fontsize=11, fontweight="bold")
    ax.text(2, 7.0, f"N x 24 base features", ha="center", fontsize=9)
    ax.annotate("", xy=(2, 6.8), xytext=(2, 8.5), arrowprops=dict(arrowstyle="->", color="#2C5F8A"))

    
    pipelines = [("P5: HAR-RV", f"3 features", 0.3, "#8B9DAF"), ("P1: VAE", f"{len(p1)} features", 2.8, "#7BA5C4"),
                 ("P2: Transformer", f"{len(p2t)} features", 5.3, "#5B7AA5"), ("P3: FinBERT", f"{len(p3)} features", 7.8, "#4A8EBF"),
                 ("P4: GAT", f"{len(p4)} features", 10.3, "#3A7AB5")]
    for name, desc, x_pos, color in pipelines:
        ax.add_patch(plt.Rectangle((x_pos, 4.5), 2.2, 1.8, facecolor=color, edgecolor="#1A3A5C", linewidth=1.2, alpha=0.7))
        ax.text(x_pos + 1.1, 5.7, name, ha="center", fontsize=9, fontweight="bold", color="white")
        ax.text(x_pos + 1.1, 5.0, desc, ha="center", fontsize=8, color="white")
        ax.annotate("", xy=(x_pos + 1.1, 6.3), xytext=(2, 6.8), arrowprops=dict(arrowstyle="->", color="#2C5F8A", alpha=0.5))

    # Dual XGBoost
    ax.add_patch(plt.Rectangle((1.5, 2.5), 4.5, 1.5, facecolor="#2C5F8A", edgecolor="#1A3A5C", linewidth=2))
    ax.text(3.75, 3.5, "XGBoost 22-day", ha="center", fontsize=10, fontweight="bold", color="white")
    ax.text(3.75, 2.9, f"N x {len(meta_feat_cols)} -> predicted 22d vol", ha="center", fontsize=8, color="white")

    ax.add_patch(plt.Rectangle((7.5, 2.5), 4.5, 1.5, facecolor="#5B7AA5", edgecolor="#1A3A5C", linewidth=2))
    ax.text(9.75, 3.5, "XGBoost 5-day", ha="center", fontsize=10, fontweight="bold", color="white")
    ax.text(9.75, 2.9, f"N x {len(meta_feat_cols)} -> predicted 5d vol", ha="center", fontsize=8, color="white")

    for name, desc, x_pos, color in pipelines:
        ax.annotate("", xy=(3.75, 4.0), xytext=(x_pos + 1.1, 4.5), arrowprops=dict(arrowstyle="->", color="#2C5F8A", alpha=0.3))
        ax.annotate("", xy=(9.75, 4.0), xytext=(x_pos + 1.1, 4.5), arrowprops=dict(arrowstyle="->", color="#5B7AA5", alpha=0.3))

    # Shock detector + blender
    ax.add_patch(plt.Rectangle((3, 0.8), 8, 1.2, facecolor="#C44E52", edgecolor="#1A3A5C", linewidth=2, alpha=0.8))
    ax.text(7, 1.7, "Shock Detector + Adaptive Blender", ha="center", fontsize=11, fontweight="bold", color="white")
    ax.text(7, 1.1, "Calm: 100% 22d  |  Shock: blend toward 5d  |  HAR-RV regime weights", ha="center", fontsize=8, color="white")
    ax.annotate("", xy=(5, 2.0), xytext=(3.75, 2.5), arrowprops=dict(arrowstyle="->", color="#2C5F8A"))
    ax.annotate("", xy=(9, 2.0), xytext=(9.75, 2.5), arrowprops=dict(arrowstyle="->", color="#5B7AA5"))

    # Output
    ax.add_patch(plt.Rectangle((4.5, 0), 5, 0.6, facecolor="#DDE7EF", edgecolor="#2C5F8A", linewidth=1.5))
    ax.text(7, 0.3, "Output: Adaptive Vol Forecast + Spike Probability", ha="center", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(7, 0.6), xytext=(7, 0.8), arrowprops=dict(arrowstyle="->", color="#2C5F8A"))

    plt.tight_layout(); plt.savefig(f"{PROC_DIR}/report_architecture.png", dpi=150); plt.close()
    print(f"  Total features: {len(meta_feat_cols)}")


def print_summary(results, wf_df, seed_df):
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS TABLE")
    print("=" * 60)
    actual = results["actual"]; spike = results["spike"]

    dm_har_adp, p_har_adp = diebold_mariano(actual, results["har_pred"], results["adaptive"])
    dm_har_std, p_har_std = diebold_mariano(actual, results["har_pred"], results["std_pred"])
    dm_std_adp, p_std_adp = diebold_mariano(actual, results["std_pred"], results["adaptive"])

    print(f"\n  {'Metric':<30s} {'HAR-RV':>10s} {'Standard':>10s} {'Adaptive':>10s}")
    print(f"  {'-'*62}")
    print(f"  {'Test AUC (2020+)':<30s} {results['har_auc']:>10.4f} {results['std_auc']:>10.4f} {results['adp_auc']:>10.4f}")
    print(f"  {'Test MSE (2020+)':<30s} {results['har_mse']:>10.6f} {results['std_mse']:>10.6f} {results['adp_mse']:>10.6f}")
    print(f"  {'Walk-Forward Avg AUC':<30s} {wf_df['HAR-RV'].mean():>10.3f} {wf_df['Standard'].mean():>10.3f} {wf_df['Adaptive'].mean():>10.3f}")

    print(f"\n  Seed Robustness (n=5):")
    print(f"  {'Adaptive AUC':<30s} {seed_df['Adp AUC'].mean():.4f} +/- {seed_df['Adp AUC'].std():.4f}")
    print(f"  {'Adaptive MSE':<30s} {seed_df['Adp MSE'].mean():.6f} +/- {seed_df['Adp MSE'].std():.6f}")

    print(f"\n  Diebold-Mariano Tests:")
    for name, dm, p in [("HAR-RV vs Adaptive", dm_har_adp, p_har_adp),
                         ("HAR-RV vs Standard", dm_har_std, p_har_std),
                         ("Standard vs Adaptive", dm_std_adp, p_std_adp)]:
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {name:<30s} DM={dm:.3f}  p={p:.4f} {sig}")

    print(f"\n  Improvement over HAR-RV:")
    print(f"  {'Test MSE':<30s} {(results['har_mse']-results['adp_mse'])/results['har_mse']*100:+.1f}%")
    print(f"  {'Test AUC':<30s} +{(results['adp_auc']-results['har_auc']):.4f}")
    print(f"  {'vs Standard MSE':<30s} {(results['std_mse']-results['adp_mse'])/results['std_mse']*100:+.1f}%")


def main():
    print("=" * 60)
    print("ADAPTIVE 2-HORIZON EVALUATION REPORT")
    print("=" * 60)

    meta, meta_feat_cols = load_data()
    print(f"  Data: {meta.shape[0]} rows, {len(meta_feat_cols)} features")

    results = run_adaptive_seed(meta, meta_feat_cols, SEED)
    print(f"\n  Adaptive AUC: {results['adp_auc']:.4f}")
    print(f"  Adaptive MSE: {results['adp_mse']:.6f}")
    print(f"  Standard AUC: {results['std_auc']:.4f}")
    print(f"  HAR-RV AUC:   {results['har_auc']:.4f}")

    wf_df = fig_walk_forward(meta, meta_feat_cols)
    fig_shap_pipeline(meta_feat_cols, results["xgb_22"], results["test"])
    seed_df = fig_seed_robustness(meta, meta_feat_cols)
    fig_predictions_vs_actual(results)
    fig_calibration(results)
    fig_top_features(meta_feat_cols, results["xgb_22"], results["test"])
    fig_regime_analysis(results)
    fig_ablation(meta, meta_feat_cols)
    fig_lr_sensitivity(meta, meta_feat_cols)
    fig_architecture(meta_feat_cols)

    print_summary(results, wf_df, seed_df)

    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to {PROC_DIR}/report_*.png:")
    for f in sorted(os.listdir(PROC_DIR)):
        if f.startswith("report_"): print(f"  {f}")


if __name__ == "__main__":
    main()
