"""SHAP-based feature selection using a tree ensemble explainer."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor

from src.data import subset_features
from src.train import train_model


def run_shap_selection(splits: dict, drop_n: int | None = None) -> dict:
    """Rank features with SHAP, drop least important, retrain MLP."""
    config = splits["config"]
    drop_n = drop_n if drop_n is not None else config["shap_drop_n"]
    feature_cols = splits["feature_cols"]

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]

    tree_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=config["seed"],
    )
    tree_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_val)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = sorted(
        zip(feature_cols, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )
    kept = [name for name, _ in importance[:-drop_n]]
    dropped = [name for name, _ in importance[-drop_n:]]

    artifacts = Path(config["paths"]["artifacts"])
    plots_dir = artifacts / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_val, columns=feature_cols),
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar chart of mean |SHAP|
    names = [n for n, _ in importance]
    scores = [s for _, s in importance]
    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], scores[::-1], color="#457B9D")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Feature Importance (GradientBoosting + SHAP)")
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    feature_list = {
        "kept_features": kept,
        "dropped_features": dropped,
        "importance_ranking": [{"feature": n, "mean_abs_shap": float(s)} for n, s in importance],
    }
    with open(artifacts / "feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)

    reduced_splits = subset_features(splits, kept)
    baseline_cfg = config["baseline"]
    train_result = train_model(
        reduced_splits,
        hidden1=baseline_cfg["hidden1"],
        hidden2=baseline_cfg["hidden2"],
        dropout=baseline_cfg["dropout"],
        lr=baseline_cfg["lr"],
        batch_size=baseline_cfg["batch_size"],
        max_epochs=baseline_cfg["max_epochs"],
        patience=baseline_cfg["patience"],
        model_name="shap_mlp",
    )

    return {
        "kept_features": kept,
        "dropped_features": dropped,
        "importance": importance,
        "train_result": train_result,
        "reduced_splits": reduced_splits,
    }
