from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def regression_metrics(target, prediction) -> dict[str, float]:
    y = np.asarray(target)
    p = np.asarray(prediction)
    error = p - y
    mape_mask = np.abs(y) >= 1.0
    mape = float(np.mean(np.abs(error[mape_mask]) / np.abs(y[mape_mask])) * 100) if np.any(mape_mask) else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": mape,
        "mape_valid_fraction": float(np.mean(mape_mask)),
    }


def classification_metrics(target, probability) -> dict[str, float]:
    y = np.asarray(target).reshape(-1)
    p = np.asarray(probability).reshape(-1)
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p))}
