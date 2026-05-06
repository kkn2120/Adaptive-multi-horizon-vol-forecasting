"""P5: HAR-RV linear baseline."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from config import *


def train_har_rv(feat):
    """Train and evaluate HAR-RV. Returns model and predictions."""
    print("\n" + "=" * 60)
    print("P5: HAR-RV BASELINE")
    print("=" * 60)

    har_cols = ["RV_1d", "RV_5d", "RV_22d"]
    train = feat[feat.index < SPLIT_DATE]
    test = feat[feat.index >= SPLIT_DATE]

    har = LinearRegression().fit(train[har_cols], train["fwd_rv_22d"])
    har_pred = har.predict(test[har_cols])

    mse = mean_squared_error(test["fwd_rv_22d"], har_pred)
    auc = roc_auc_score(test["spike_label"], har_pred)

    print(f"  Coefs: RV_1d={har.coef_[0]:.4f}, RV_5d={har.coef_[1]:.4f}, RV_22d={har.coef_[2]:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  AUC: {auc:.4f}")

    return har, har_pred
