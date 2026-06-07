"""PSO hyperparameter tuning with pyswarms."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps

from src.train import train_model


def run_pso_tuning(splits: dict, feature_splits: dict | None = None) -> dict:
    """Tune lr, hidden size, and dropout with PSO."""
    data = feature_splits or splits
    config = splits["config"]
    pso_cfg = config["pso"]
    baseline_cfg = config["baseline"]
    seed = config["seed"]

    bounds = (
        np.array(pso_cfg["lr_bounds"]),
        np.array(pso_cfg["hidden_bounds"], dtype=float),
        np.array(pso_cfg["dropout_bounds"]),
    )
    lower = np.array([bounds[0][0], bounds[1][0], bounds[2][0]])
    upper = np.array([bounds[0][1], bounds[1][1], bounds[2][1]])

    history_costs = []

    def objective(particles: np.ndarray) -> np.ndarray:
        costs = []
        for particle in particles:
            lr = float(particle[0])
            hidden = int(round(particle[1]))
            dropout = float(particle[2])
            result = train_model(
                data,
                hidden1=hidden,
                hidden2=max(hidden // 2, 16),
                dropout=dropout,
                lr=lr,
                batch_size=baseline_cfg["batch_size"],
                max_epochs=pso_cfg["inner_epochs"],
                patience=10,
                model_name="pso_candidate",
                save=False,
            )
            costs.append(result["val_metrics"]["rmse"])
        history_costs.append(costs)
        return np.array(costs)

    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=pso_cfg["n_particles"],
        dimensions=3,
        options=options,
        bounds=(lower, upper),
    )

    np.random.seed(seed)
    best_cost, best_pos = optimizer.optimize(
        objective,
        iters=pso_cfg["n_iterations"],
        verbose=False,
    )

    best_lr = float(best_pos[0])
    best_hidden = int(round(best_pos[1]))
    best_dropout = float(best_pos[2])

    final_result = train_model(
        data,
        hidden1=best_hidden,
        hidden2=max(best_hidden // 2, 16),
        dropout=best_dropout,
        lr=best_lr,
        batch_size=baseline_cfg["batch_size"],
        max_epochs=baseline_cfg["max_epochs"],
        patience=baseline_cfg["patience"],
        model_name="pso_mlp",
    )

    artifacts = Path(config["paths"]["artifacts"])
    plots_dir = artifacts / "plots"
    results_dir = artifacts / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convergence: best cost per iteration
    iter_best = [min(costs) for costs in history_costs]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(iter_best) + 1), iter_best, marker="o", color="#E63946")
    plt.xlabel("PSO Iteration")
    plt.ylabel("Best Validation RMSE (ha)")
    plt.title("PSO Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pso_convergence.png", dpi=150)
    plt.close()

    pso_result = {
        "best_cost": float(best_cost),
        "best_hyperparameters": {
            "lr": best_lr,
            "hidden1": best_hidden,
            "hidden2": max(best_hidden // 2, 16),
            "dropout": best_dropout,
        },
        "convergence": iter_best,
    }
    with open(results_dir / "pso_best.json", "w") as f:
        json.dump(pso_result, f, indent=2)

    return {
        "pso_result": pso_result,
        "train_result": final_result,
    }
