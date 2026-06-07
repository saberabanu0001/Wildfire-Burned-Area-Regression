"""Evaluation metrics for burned-area regression."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_log_target(y_log: np.ndarray) -> np.ndarray:
    """Convert log1p predictions back to hectares."""
    return np.expm1(y_log)


def compute_metrics(y_true_ha: np.ndarray, y_pred_ha: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R² on original hectare scale."""
    y_true_ha = np.asarray(y_true_ha, dtype=float).ravel()
    y_pred_ha = np.asarray(y_pred_ha, dtype=float).ravel()
    y_pred_ha = np.clip(y_pred_ha, 0, None)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_ha, y_pred_ha))),
        "mae": float(mean_absolute_error(y_true_ha, y_pred_ha)),
        "r2": float(r2_score(y_true_ha, y_pred_ha)),
    }


def metrics_from_log_predictions(
    y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict:
    """Compute metrics after inverse log1p transform."""
    return compute_metrics(
        inverse_log_target(y_true_log),
        inverse_log_target(y_pred_log),
    )
