#!/usr/bin/env python3
"""Run the full wildfire ML pipeline end-to-end."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_config, load_raw_data, prepare_data
from src.metrics import inverse_log_target
from src.pso_tune import run_pso_tuning
from src.shap_selection import run_shap_selection
from src.train import train_model


def run_eda(config: dict) -> None:
    """Generate EDA plots (Phase 1)."""
    df = load_raw_data(config)
    plots_dir = Path(config["paths"]["artifacts"]) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    area = df["area"]
    print(f"Rows: {len(df)}, zero-area: {(area == 0).sum()} ({100 * (area == 0).mean():.1f}%)")
    print(f"Area skew: {area.skew():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(area, bins=50, color="#FF6B35", edgecolor="white")
    axes[0].set_title("Burned Area (ha) — Raw")
    axes[0].set_xlabel("area (ha)")

    log_area = np.log1p(area)
    axes[1].hist(log_area, bins=50, color="#457B9D", edgecolor="white")
    axes[1].set_title("log1p(area)")
    axes[1].set_xlabel("log1p(area)")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_area_distribution.png", dpi=150)
    plt.close()

    num_cols = config["numerical_cols"] + [config["target_col"]]
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    import seaborn as sns

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_correlation.png", dpi=150)
    plt.close()


def plot_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    y_true = inverse_log_target(y_true_log)
    y_pred = inverse_log_target(y_pred_log)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="#2D6A4F", edgecolors="none")
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--", lw=1)
    plt.xlabel("Actual Area (ha)")
    plt.ylabel("Predicted Area (ha)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    config = load_config()
    artifacts = Path(config["paths"]["artifacts"])
    plots_dir = artifacts / "plots"
    results_dir = artifacts / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: EDA ===")
    run_eda(config)

    print("=== Phase 2: Preprocessing ===")
    splits = prepare_data()
    print(f"Features: {len(splits['feature_cols'])}")

    print("=== Phase 3: Baseline MLP ===")
    baseline_cfg = config["baseline"]
    baseline = train_model(
        splits,
        hidden1=baseline_cfg["hidden1"],
        hidden2=baseline_cfg["hidden2"],
        dropout=baseline_cfg["dropout"],
        lr=baseline_cfg["lr"],
        batch_size=baseline_cfg["batch_size"],
        max_epochs=baseline_cfg["max_epochs"],
        patience=baseline_cfg["patience"],
        model_name="baseline_mlp",
    )
    print("Baseline test metrics:", baseline["test_metrics"])
    plot_predictions(
        splits["y_test"],
        baseline["test_predictions"],
        "Baseline MLP — Actual vs Predicted",
        plots_dir / "baseline_predictions.png",
    )

    print("=== Phase 4: SHAP Feature Selection ===")
    shap_out = run_shap_selection(splits)
    print("Dropped:", shap_out["dropped_features"])
    print("SHAP MLP test metrics:", shap_out["train_result"]["test_metrics"])
    plot_predictions(
        shap_out["reduced_splits"]["y_test"],
        shap_out["train_result"]["test_predictions"],
        "SHAP-reduced MLP — Actual vs Predicted",
        plots_dir / "shap_predictions.png",
    )

    print("=== Phase 5: PSO Hyperparameter Tuning ===")
    pso_out = run_pso_tuning(splits, feature_splits=shap_out["reduced_splits"])
    print("PSO best:", pso_out["pso_result"]["best_hyperparameters"])
    print("PSO MLP test metrics:", pso_out["train_result"]["test_metrics"])
    plot_predictions(
        splits["y_test"],
        pso_out["train_result"]["test_predictions"],
        "PSO-tuned MLP — Actual vs Predicted",
        plots_dir / "pso_predictions.png",
    )

    print("=== Phase 6: Evaluation ===")
    comparison = pd.DataFrame(
        [
            {
                "model": "Baseline MLP",
                **baseline["test_metrics"],
            },
            {
                "model": "SHAP-reduced MLP",
                **shap_out["train_result"]["test_metrics"],
            },
            {
                "model": "PSO-tuned MLP (SHAP features)",
                **pso_out["train_result"]["test_metrics"],
            },
        ]
    )
    comparison.to_csv(results_dir / "comparison.csv", index=False)
    print(comparison.to_string(index=False))

    summary = {
        "baseline": baseline["test_metrics"],
        "shap_mlp": shap_out["train_result"]["test_metrics"],
        "pso_mlp": pso_out["train_result"]["test_metrics"],
        "dropped_features": shap_out["dropped_features"],
        "pso_hyperparameters": pso_out["pso_result"]["best_hyperparameters"],
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Artifacts saved to", artifacts)


if __name__ == "__main__":
    main()
