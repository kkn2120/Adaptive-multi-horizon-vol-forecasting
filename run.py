import sys
import gc
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

from config import *


def load_spx_csv(path):
    """Load SPX CSV handling both yfinance formats."""
    import pandas as pd
    try:
        df = pd.read_csv(path, header=[0,1,2], index_col=0, parse_dates=True).droplevel([1,2], axis=1)
        return df
    except:
        return pd.read_csv(path, index_col=0, parse_dates=True)


def main():
    skip_download = "--skip-download" in sys.argv
    import os, json, pandas as pd

    
    if not skip_download:
        from data_download import download_all
        data = download_all()
        if isinstance(data["spx"].columns, pd.MultiIndex):
            data["spx"].columns = data["spx"].columns.get_level_values(0)
    else:
        print("Loading cached data...")
        data = {
            "spx": load_spx_csv(os.path.join(RAW_DIR, "spx_daily.csv")),
            "vix": pd.read_csv(os.path.join(RAW_DIR, "vix_term_structure.csv"), index_col=0, parse_dates=True),
            "cross_assets": pd.read_csv(os.path.join(RAW_DIR, "cross_assets.csv"), index_col=0, parse_dates=True),
            "spx_intraday": None,
            "options": None,
        }
        intra_path = os.path.join(RAW_DIR, "spx_intraday.csv")
        if os.path.exists(intra_path):
            data["spx_intraday"] = pd.read_csv(intra_path, index_col=0, parse_dates=True)
        opts_path = os.path.join(RAW_DIR, "spx_options.csv")
        if os.path.exists(opts_path):
            data["options"] = pd.read_csv(opts_path, index_col=0)
        news_path = os.path.join(RAW_DIR, "news_headlines.json")
        if os.path.exists(news_path):
            with open(news_path) as f:
                data["news"] = json.load(f)
        else:
            data["news"] = {}
        print("Cached data loaded.")

    
    from features import build_features
    feat, feature_cols, target_cols, threshold = build_features(
        data["spx"], data["vix"], data.get("spx_intraday"), data.get("options")
    )

    
    from p5_har_rv import train_har_rv
    har_model, har_pred = train_har_rv(feat)
    gc.collect()

    
    from p1_vae import train_vae
    latent_df, vae_model = train_vae(feat, feature_cols)
    del vae_model; gc.collect()
    print("  [memory freed: P1 model released]")

    
    from p2_transformer import train_transformer
    trans_df, trans_model = train_transformer(feat, feature_cols)
    del trans_model; gc.collect()
    print("  [memory freed: P2 model released]")

    
    from p3_finbert import train_finbert
    text_df = train_finbert(feat, data.get("news", {}))
    gc.collect()
    print("  [memory freed: P3 released]")

    
    from p4_gat import train_gat
    gat_df, gat_model = train_gat(feat, data["cross_assets"])
    del gat_model, data; gc.collect()
    print("  [memory freed: P4 model + raw data released]")

    
    from meta_model import train_meta
    gc.collect()
    results = train_meta(feat, feature_cols, latent_df, trans_df, text_df, gat_df, har_pred)

    
    from evaluation import run_evaluation, run_walk_forward
    eval_results = run_evaluation(results)
    wf_df = run_walk_forward(results["meta"], results["meta_feat_cols"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {PROC_DIR}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
