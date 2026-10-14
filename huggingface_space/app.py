import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

FEATURE_COLS = [
    "mean_consumption", "std_consumption", "max_consumption", "min_consumption",
    "cv_consumption", "peak_offpeak_ratio", "weekend_weekday_ratio",
    "mean_to_peak_ratio", "load_factor", "sudden_change_count",
    "rolling_mean_7d", "rolling_std_7d"
]

model  = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(mean_c, std_c, max_c, min_c, cv_c, peak_op,
            wk_ratio, mtp, lf, sc_count, roll_m, roll_s):

    features = {
        "mean_consumption":      mean_c,
        "std_consumption":       std_c,
        "max_consumption":       max_c,
        "min_consumption":       min_c,
        "cv_consumption":        cv_c,
        "peak_offpeak_ratio":    peak_op,
        "weekend_weekday_ratio": wk_ratio,
        "mean_to_peak_ratio":    mtp,
        "load_factor":           lf,
        "sudden_change_count":   sc_count,
        "rolling_mean_7d":       roll_m,
        "rolling_std_7d":        roll_s,
    }

    X    = pd.DataFrame([features])[FEATURE_COLS]
    X_sc = scaler.transform(X)
    proba = model.predict_proba(X_sc)[0][1]
    risk_score = round(proba * 100, 1)

    if proba < 0.4:   risk_level = "Low"
    elif proba < 0.7: risk_level = "Medium"
    else:             risk_level = "High"

    label = "Potential Electricity Theft" if proba >= 0.5 else "Normal Consumer"
    top3  = [FEATURE_COLS[i] for i in np.argsort(model.feature_importances_)[::-1][:3]]

    # Risk gauge plot
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.barh([""], [40], color="#27ae60", height=0.5)
    ax.barh([""], [30], color="#e67e22", height=0.5, left=40)
    ax.barh([""], [30], color="#e74c3c", height=0.5, left=70)
    ax.axvline(risk_score, color="white", linewidth=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk Score", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3748")
    ax.set_title(f"Risk Score: {risk_score}/100", color="white", fontsize=11)
    plt.tight_layout()

    result = f"""
## {'🚨' if risk_level=='High' else '⚠️' if risk_level=='Medium' else '✅'} {label}

| Metric | Value |
|--------|-------|
| Probability | {proba*100:.1f}% |
| Risk Score | {risk_score} / 100 |
| Risk Level | {risk_level} |
| Top Feature 1 | {top3[0]} |
| Top Feature 2 | {top3[1]} |
| Top Feature 3 | {top3[2]} |
"""
    return result, fig

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Electricity Theft Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ⚡ Electricity Theft Detection System
    **Explainable AI-powered detection using Ensemble Learning + SHAP | SGCC Dataset**

    Adjust the consumer features below and click **Predict** to detect potential electricity theft.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Consumption Statistics")
            mean_c   = gr.Number(label="Mean Daily Consumption (kWh)", value=7.0)
            std_c    = gr.Number(label="Std of Consumption",           value=4.0)
            max_c    = gr.Number(label="Max Daily Consumption (kWh)",  value=28.0)
            min_c    = gr.Number(label="Min Daily Consumption (kWh)",  value=0.5)

            gr.Markdown("### Derived Ratios")
            cv_c     = gr.Number(label="CV of Consumption",            value=1.46)
            peak_op  = gr.Number(label="Peak / Off-Peak Ratio",        value=15.0)
            wk_ratio = gr.Number(label="Weekend / Weekday Ratio",      value=1.3)
            mtp      = gr.Number(label="Mean-to-Peak Ratio (0-1)",     value=0.25)
            lf       = gr.Number(label="Load Factor (0-1)",            value=0.25)

            gr.Markdown("### Temporal Features")
            sc_count = gr.Number(label="Sudden Change Count",          value=115.0)
            roll_m   = gr.Number(label="Rolling Mean 7d (kWh)",        value=6.5)
            roll_s   = gr.Number(label="Rolling Std 7d",               value=1.3)

            btn = gr.Button("Run Prediction", variant="primary")

        with gr.Column():
            gr.Markdown("### Prediction Result")
            result_md  = gr.Markdown("*Enter features and click Run Prediction*")
            gauge_plot = gr.Plot(label="Risk Gauge")

    btn.click(
        fn=predict,
        inputs=[mean_c, std_c, max_c, min_c, cv_c, peak_op,
                wk_ratio, mtp, lf, sc_count, roll_m, roll_s],
        outputs=[result_md, gauge_plot]
    )

    gr.Markdown("---")

    with gr.Accordion("View EDA & Model Plots", open=False):
        figures_dir = "results/figures"
        if os.path.exists(figures_dir):
            plot_files = sorted([f for f in os.listdir(figures_dir) if f.endswith(".png")])
            for i in range(0, len(plot_files), 2):
                with gr.Row():
                    gr.Image(os.path.join(figures_dir, plot_files[i]),
                             label=plot_files[i].replace(".png","").replace("_"," "))
                    if i+1 < len(plot_files):
                        gr.Image(os.path.join(figures_dir, plot_files[i+1]),
                                 label=plot_files[i+1].replace(".png","").replace("_"," "))

    gr.Markdown("""
    ---
    *B III M.Sc. AI & ML — Advanced Machine Learning Lab Project*
    """)

if __name__ == "__main__":
    # HuggingFace Spaces (Docker SDK) requires server_port=7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
