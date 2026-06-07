"""Training and evaluation loop for the MLP."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import metrics_from_log_predictions
from src.models import BurnedAreaMLP


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    splits: dict,
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 15,
    model_name: str = "model",
    save: bool = True,
) -> dict:
    """Train MLP with early stopping; return model and metrics."""
    config = splits["config"]
    set_seed(config["seed"])

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    input_dim = X_train.shape[1]
    model = BurnedAreaMLP(input_dim, hidden1, hidden2, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)
        history["train_loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_preds, torch.tensor(y_val, dtype=torch.float32)).item()
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    val_metrics = metrics_from_log_predictions(y_val, val_preds)
    test_metrics = metrics_from_log_predictions(y_test, test_preds)

    result = {
        "model_name": model_name,
        "hyperparameters": {
            "hidden1": hidden1,
            "hidden2": hidden2,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs_trained": len(history["train_loss"]),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "model": model,
        "val_predictions": val_preds,
        "test_predictions": test_preds,
    }

    if save:
        artifacts = Path(config["paths"]["artifacts"])
        models_dir = artifacts / "models"
        results_dir = artifacts / "results"
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "dropout": dropout,
                "feature_cols": splits["feature_cols"],
            },
            models_dir / f"{model_name}.pt",
        )

        with open(results_dir / f"{model_name}.json", "w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "hyperparameters": result["hyperparameters"],
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
            )

    return result
