"""Data loading, preprocessing, and splitting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    return pd.read_csv(config["paths"]["raw_data"])


def preprocess_dataframe(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals; keep target in original hectares."""
    df = df.copy()
    categorical_cols = config["categorical_cols"]
    numerical_cols = config["numerical_cols"]
    target_col = config["target_col"]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=float)
    feature_cols = [c for c in df_encoded.columns if c != target_col]
    # Stable ordering: numerical first, then one-hot columns
    one_hot_cols = sorted(c for c in feature_cols if c not in numerical_cols)
    feature_cols = numerical_cols + one_hot_cols

    return df_encoded[feature_cols + [target_col]], feature_cols


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
) -> dict:
    """Train/val/test split with stratification-friendly random split."""
    target_col = config["target_col"]
    seed = config["seed"]
    split_cfg = config["split"]

    X = df[feature_cols].values
    y = df[target_col].values
    if config.get("log_target", True):
        y_model = np.log1p(y)
    else:
        y_model = y.copy()

    X_train, X_temp, y_train, y_temp, y_train_ha, y_temp_ha = _split_with_targets(
        X, y_model, y, test_size=(1 - split_cfg["train"]), random_state=seed
    )

    val_ratio = split_cfg["val"] / (split_cfg["val"] + split_cfg["test"])
    X_val, X_test, y_val, y_test, y_val_ha, y_test_ha = _split_with_targets(
        X_temp, y_temp, y_temp_ha, test_size=(1 - val_ratio), random_state=seed
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_ha": y_train_ha,
        "y_val_ha": y_val_ha,
        "y_test_ha": y_test_ha,
        "feature_cols": feature_cols,
    }


def _split_with_targets(X, y_model, y_ha, test_size, random_state):
    X_a, X_b, y_a, y_b, y_ha_a, y_ha_b = train_test_split(
        X, y_model, y_ha, test_size=test_size, random_state=random_state
    )
    return X_a, X_b, y_a, y_b, y_ha_a, y_ha_b


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(scaler: MinMaxScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def save_processed_splits(
    splits: dict,
    scaler: MinMaxScaler,
    config: dict,
    feature_cols: list[str] | None = None,
) -> Path:
    """Save processed CSVs, scaler, and feature metadata."""
    artifacts = Path(config["paths"]["artifacts"])
    processed_dir = Path(config["paths"]["processed_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_cols or splits["feature_cols"]
    joblib.dump(scaler, artifacts / "scaler.joblib")

    for name in ("train", "val", "test"):
        X = splits[f"X_{name}"]
        y_ha = splits[f"y_{name}_ha"]
        y_log = splits[f"y_{name}"]
        part = pd.DataFrame(X, columns=cols)
        part["area_ha"] = y_ha
        part["area_log1p"] = y_log
        part.to_csv(processed_dir / f"{name}.csv", index=False)

    metadata = {
        "feature_cols": cols,
        "n_features": len(cols),
        "log_target": config.get("log_target", True),
    }
    import json

    with open(artifacts / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return artifacts


def prepare_data(config_path: str = "config.yaml") -> dict:
    """Full preprocessing pipeline."""
    config = load_config(config_path)
    df = load_raw_data(config)
    df_processed, feature_cols = preprocess_dataframe(df, config)
    splits = split_data(df_processed, feature_cols, config)
    scaler = fit_scaler(splits["X_train"])

    splits["X_train"] = transform_features(scaler, splits["X_train"])
    splits["X_val"] = transform_features(scaler, splits["X_val"])
    splits["X_test"] = transform_features(scaler, splits["X_test"])

    save_processed_splits(splits, scaler, config, feature_cols)
    splits["scaler"] = scaler
    splits["config"] = config
    return splits


def subset_features(splits: dict, feature_cols: list[str]) -> dict:
    """Return a copy of splits restricted to selected feature columns."""
    all_cols = splits["feature_cols"]
    indices = [all_cols.index(c) for c in feature_cols]
    out = dict(splits)
    out["feature_cols"] = feature_cols
    for split in ("train", "val", "test"):
        out[f"X_{split}"] = splits[f"X_{split}"][:, indices]
    return out
