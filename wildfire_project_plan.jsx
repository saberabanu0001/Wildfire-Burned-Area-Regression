import { useState } from "react";

const phases = [
  {
    id: 1,
    emoji: "📦",
    title: "Data Collection & Setup",
    duration: "Week 1",
    color: "#FF6B35",
    tasks: [
      "Download Forest Fires Dataset (UCI ML Repository) — tabular data only, no satellite imagery needed",
      "Set up Google Colab notebook + GitHub repo",
      "Install libraries: PyTorch, Scikit-learn, SHAP, pyswarms",
      "Explore dataset: 13 features (temp, wind, humidity, FFMC, DMC, etc.), target = burned area (ha)",
    ],
    note: "⚡ Start with UCI dataset — no satellite imagery complexity. Simpler and well-documented.",
    output: "Clean repo + EDA notebook committed to GitHub",
  },
  {
    id: 2,
    emoji: "🧹",
    title: "Data Preprocessing",
    duration: "Week 2",
    color: "#F7C59F",
    tasks: [
      "Handle extreme skewness in burned area → apply log1p transformation",
      "Encode categorical features (month, day) with one-hot encoding",
      "Normalize numerical features using MinMaxScaler",
      "Split into train/val/test (70/15/15)",
    ],
    note: "🔑 log1p(area) is the single most important step — skewed targets destroy regression models.",
    output: "Preprocessed dataset saved as .csv + scaler saved with joblib",
  },
  {
    id: 3,
    emoji: "🧠",
    title: "Build Baseline MLP",
    duration: "Week 3",
    color: "#A8DADC",
    tasks: [
      "Build a simple 3-layer MLP in PyTorch (input → 64 → 32 → 1)",
      "Train with Adam optimizer, MSE loss",
      "Evaluate with RMSE and MAE on test set",
      "This is your baseline — record the numbers carefully",
    ],
    note: "📌 Don't skip this. A baseline gives you something to beat in later phases.",
    output: "Baseline RMSE/MAE scores logged + model saved",
  },
  {
    id: 4,
    emoji: "🔍",
    title: "Feature Selection with SHAP",
    duration: "Week 4",
    color: "#457B9D",
    tasks: [
      "Train a LightGBM model quickly as SHAP explainer backbone",
      "Generate SHAP summary plot — visualize feature importance",
      "Drop bottom 2–3 features by SHAP score",
      "Retrain MLP on reduced feature set and compare RMSE",
    ],
    note: "🎯 This is your 'wrapper-based feature selection' — SHAP tells you what matters.",
    output: "SHAP plot saved + reduced feature list documented",
  },
  {
    id: 5,
    emoji: "⚙️",
    title: "PSO Hyperparameter Tuning",
    duration: "Week 5",
    color: "#E63946",
    tasks: [
      "Use pyswarms library (simple PSO, beginner-friendly)",
      "Optimize 3 hyperparameters only: learning rate, hidden layer size, dropout rate",
      "Run PSO for 20 iterations with 10 particles (light compute on Colab)",
      "Train final MLP with PSO-found hyperparameters",
    ],
    note: "🐝 Keep PSO search space small. 3 params × 10 particles × 20 iters is enough to show improvement.",
    output: "PSO convergence curve + best hyperparameters logged",
  },
  {
    id: 6,
    emoji: "📊",
    title: "Evaluation & Results",
    duration: "Week 6",
    color: "#2D6A4F",
    tasks: [
      "Compare 3 models: Baseline MLP, SHAP-reduced MLP, PSO-tuned MLP",
      "Metrics: RMSE, MAE, R² — table format",
      "Plot: Actual vs Predicted burned area (scatter)",
      "Final SHAP bar chart on best model for explainability",
    ],
    note: "📈 Your research story: PSO + SHAP selection beats baseline → that's your contribution.",
    output: "Results table + all plots saved as .png",
  },
  {
    id: 7,
    emoji: "📝",
    title: "Report & Submission",
    duration: "Week 7",
    color: "#6D6875",
    tasks: [
      "Write report sections: Introduction, Methodology, Results, Discussion",
      "Include all plots inline",
      "Push final notebook + README to GitHub",
      "Optional: add Gradio demo for live predictions in Colab",
    ],
    note: "🚀 A Gradio demo takes 10 lines of code and makes your submission stand out.",
    output: "Final report (.pdf or .docx) + public GitHub repo",
  },
];

const stack = [
  { label: "Language", value: "Python 3.10+", icon: "🐍" },
  { label: "DL Framework", value: "PyTorch", icon: "🔥" },
  { label: "ML Utilities", value: "Scikit-learn", icon: "⚙️" },
  { label: "PSO Library", value: "pyswarms", icon: "🐝" },
  { label: "Explainability", value: "SHAP + LightGBM", icon: "🔍" },
  { label: "Environment", value: "Google Colab + Jupyter", icon: "☁️" },
  { label: "Version Control", value: "GitHub", icon: "🐙" },
  { label: "Dataset", value: "UCI Forest Fires", icon: "📦" },
];

export default function WildfirePlan() {
  const [active, setActive] = useState(null);

  return (
    <div style={{
      fontFamily: "'Georgia', serif",
      background: "linear-gradient(135deg, #0d0d0d 0%, #1a0a00 50%, #0d0d0d 100%)",
      minHeight: "100vh",
      color: "#f5e6d0",
      padding: "40px 20px",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 48 }}>
        <div style={{ fontSize: 56, marginBottom: 12 }}>🔥</div>
        <h1 style={{
          fontFamily: "'Georgia', serif",
          fontSize: "clamp(22px, 4vw, 36px)",
          fontWeight: 700,
          color: "#FF6B35",
          margin: "0 0 8px",
          letterSpacing: "-0.5px",
        }}>
          Wildfire Burned Area Regression
        </h1>
        <p style={{ color: "#c9a882", fontSize: 15, margin: 0 }}>
          Metaheuristic-Optimized Hybrid Deep Learning · 7-Week Beginner Plan
        </p>
      </div>

      {/* Stack */}
      <div style={{
        maxWidth: 820,
        margin: "0 auto 48px",
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
        gap: 12,
      }}>
        {stack.map(s => (
          <div key={s.label} style={{
            background: "rgba(255,107,53,0.08)",
            border: "1px solid rgba(255,107,53,0.2)",
            borderRadius: 10,
            padding: "12px 16px",
          }}>
            <div style={{ fontSize: 20, marginBottom: 4 }}>{s.icon}</div>
            <div style={{ fontSize: 11, color: "#c9a882", textTransform: "uppercase", letterSpacing: 1 }}>{s.label}</div>
            <div style={{ fontSize: 13, color: "#f5e6d0", fontWeight: 600 }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Phases */}
      <div style={{ maxWidth: 820, margin: "0 auto" }}>
        {phases.map((phase, i) => {
          const isOpen = active === phase.id;
          return (
            <div key={phase.id} style={{ marginBottom: 16 }}>
              {/* Connector line */}
              {i > 0 && (
                <div style={{
                  width: 2, height: 16, background: "rgba(255,107,53,0.2)",
                  margin: "-8px auto 0", marginBottom: 0,
                  width: 2,
                }} />
              )}
              <div
                onClick={() => setActive(isOpen ? null : phase.id)}
                style={{
                  background: isOpen
                    ? "rgba(255,107,53,0.12)"
                    : "rgba(255,255,255,0.03)",
                  border: `1px solid ${isOpen ? phase.color : "rgba(255,255,255,0.08)"}`,
                  borderRadius: 12,
                  padding: "18px 24px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  transition: "all 0.2s ease",
                }}
              >
                <div style={{
                  width: 44, height: 44, borderRadius: "50%",
                  background: phase.color + "22",
                  border: `2px solid ${phase.color}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 20, flexShrink: 0,
                }}>
                  {phase.emoji}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                    <span style={{ color: phase.color, fontSize: 12, fontFamily: "monospace" }}>
                      Phase {phase.id}
                    </span>
                    <span style={{ color: "#c9a882", fontSize: 12 }}>{phase.duration}</span>
                  </div>
                  <div style={{ fontWeight: 700, fontSize: 16, color: "#f5e6d0", marginTop: 2 }}>
                    {phase.title}
                  </div>
                </div>
                <div style={{
                  color: "#c9a882", fontSize: 18,
                  transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
                  transition: "transform 0.2s",
                }}>›</div>
              </div>

              {/* Expanded */}
              {isOpen && (
                <div style={{
                  background: "rgba(255,107,53,0.05)",
                  border: `1px solid ${phase.color}33`,
                  borderTop: "none",
                  borderRadius: "0 0 12px 12px",
                  padding: "20px 24px",
                }}>
                  {/* Key note */}
                  <div style={{
                    background: "rgba(255,107,53,0.1)",
                    border: `1px solid ${phase.color}55`,
                    borderRadius: 8,
                    padding: "10px 14px",
                    fontSize: 13,
                    color: "#f5e6d0",
                    marginBottom: 16,
                  }}>
                    {phase.note}
                  </div>

                  {/* Tasks */}
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, color: "#c9a882", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Tasks</div>
                    {phase.tasks.map((t, ti) => (
                      <div key={ti} style={{ display: "flex", gap: 10, marginBottom: 8, alignItems: "flex-start" }}>
                        <div style={{
                          width: 20, height: 20, borderRadius: "50%",
                          background: phase.color + "33",
                          border: `1px solid ${phase.color}`,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 10, color: phase.color, flexShrink: 0, marginTop: 1,
                          fontFamily: "monospace", fontWeight: 700,
                        }}>{ti + 1}</div>
                        <div style={{ fontSize: 14, color: "#e8d5bc", lineHeight: 1.5 }}>{t}</div>
                      </div>
                    ))}
                  </div>

                  {/* Output */}
                  <div style={{
                    display: "flex", gap: 8, alignItems: "flex-start",
                    padding: "10px 14px",
                    background: "rgba(0,0,0,0.2)",
                    borderRadius: 8,
                    borderLeft: `3px solid ${phase.color}`,
                  }}>
                    <span style={{ fontSize: 14 }}>✅</span>
                    <span style={{ fontSize: 13, color: "#c9a882" }}><strong style={{ color: "#f5e6d0" }}>Deliverable:</strong> {phase.output}</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Research Story */}
      <div style={{
        maxWidth: 820, margin: "40px auto 0",
        background: "rgba(255,107,53,0.07)",
        border: "1px solid rgba(255,107,53,0.25)",
        borderRadius: 16,
        padding: "28px 32px",
      }}>
        <h3 style={{ color: "#FF6B35", margin: "0 0 16px", fontSize: 18 }}>🎯 Your Research Narrative</h3>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: 16,
        }}>
          {[
            { step: "Problem", text: "Burned area is extremely skewed — most fires are small, a few are huge. Standard models fail." },
            { step: "Solution 1", text: "log1p transform + SHAP feature selection removes noise and compresses skewness." },
            { step: "Solution 2", text: "PSO replaces manual grid search and finds better hyperparameters automatically." },
            { step: "Contribution", text: "Combining SHAP + PSO on this dataset beats standard MLP — that's your result." },
          ].map(r => (
            <div key={r.step} style={{
              background: "rgba(0,0,0,0.2)", borderRadius: 10, padding: "14px 16px"
            }}>
              <div style={{ fontSize: 11, color: "#FF6B35", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{r.step}</div>
              <div style={{ fontSize: 13, color: "#e8d5bc", lineHeight: 1.6 }}>{r.text}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ textAlign: "center", marginTop: 40, color: "#c9a882", fontSize: 12 }}>
        Click each phase to expand · 7 weeks · Beginner-friendly
      </div>
    </div>
  );
}
