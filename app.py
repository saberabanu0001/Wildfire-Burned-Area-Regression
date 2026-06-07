"""Optional Gradio demo for live burned-area predictions."""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.models import BurnedAreaMLP


def load_artifacts():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    with open("artifacts/feature_metadata.json") as f:
        metadata = json.load(f)

    scaler = joblib.load("artifacts/scaler.joblib")
    checkpoint = torch.load("artifacts/models/pso_mlp.pt", weights_only=False)

    model = BurnedAreaMLP(
        input_dim=checkpoint["input_dim"],
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_feature_cols = metadata["feature_cols"]
    model_feature_cols = checkpoint["feature_cols"]
    feature_indices = [all_feature_cols.index(c) for c in model_feature_cols]

    return config, scaler, model, all_feature_cols, feature_indices


CONFIG, SCALER, MODEL, ALL_FEATURE_COLS, FEATURE_INDICES = load_artifacts()


def risk_label(hectares: float) -> str:
    if hectares < 1:
        return "Low risk"
    if hectares <= 10:
        return "Moderate risk"
    return "High risk"


def predict(
    x, y, ffmc, dmc, dc, isi, temp, rh, wind, rain, month, day
):
    row = {
        "X": x,
        "Y": y,
        "FFMC": ffmc,
        "DMC": dmc,
        "DC": dc,
        "ISI": isi,
        "temp": temp,
        "RH": rh,
        "wind": wind,
        "rain": rain,
        "month": month,
        "day": day,
    }
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=CONFIG["categorical_cols"], dtype=float)

    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[ALL_FEATURE_COLS]

    X_scaled = SCALER.transform(df.values)
    X_model = X_scaled[:, FEATURE_INDICES]
    with torch.no_grad():
        pred_log = MODEL(torch.tensor(X_model, dtype=torch.float32)).item()
    pred_ha = float(np.expm1(pred_log))
    risk = risk_label(pred_ha)
    return f"**{pred_ha:.2f} hectares**\n\n{risk}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(1, 9, value=7, label="X (grid)"),
        gr.Slider(1, 9, value=5, label="Y (grid)"),
        gr.Slider(0, 100, value=90, label="FFMC"),
        gr.Slider(0, 300, value=50, label="DMC"),
        gr.Slider(0, 900, value=200, label="DC"),
        gr.Slider(0, 60, value=10, label="ISI"),
        gr.Slider(0, 35, value=15, label="Temp (°C)"),
        gr.Slider(0, 100, value=50, label="RH (%)"),
        gr.Slider(0, 10, value=2, label="Wind"),
        gr.Slider(0, 5, value=0, label="Rain"),
        gr.Dropdown(
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            value="aug",
            label="Month",
        ),
        gr.Dropdown(
            ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            value="sun",
            label="Day",
        ),
    ],
    outputs=gr.Markdown(label="Predicted Burned Area"),
    title="Wildfire Burned Area Predictor",
    description="Predicts hectares burned using the PSO-tuned MLP. Run `python run_pipeline.py` first.",
)

if __name__ == "__main__":
    demo.launch()
