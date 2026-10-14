"""
run_pipeline.py — End-to-End Electricity Theft Detection Pipeline
=================================================================
Orchestrates all pipeline stages in the correct order:
  1. Data Preprocessing
  2. Feature Engineering
  3. EDA & Visualizations (10+ plots + Jupyter notebook)
  4. Model Training with GridSearchCV (baseline: LR, RF, XGBoost)
  5. Advanced Optuna Hyperparameter Optimization (LightGBM, XGBoost, CatBoost)
  6. Model Evaluation & Benchmarking
     - Threshold tuning (optimise F1 per model)
     - Rigorous error analysis (FPR, FNR, TP/TN/FP/FN)
     - Comparison vs published literature
  7. SHAP Explainability (global + local)

Usage:
    python run_pipeline.py               # Full pipeline
    python run_pipeline.py --skip-optuna # Skip Optuna (faster, GridSearch only)
    python run_pipeline.py --only-eval   # Only run evaluation & explain
"""

import sys
import os
import time
import argparse
import traceback

# ── Ensure project root is in path ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def banner(text):
    """Print a formatted section banner."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def run_stage(stage_name, func, *args, **kwargs):
    """Run a pipeline stage with timing and error handling."""
    banner(f"STAGE: {stage_name}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n  ✓ {stage_name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ✗ {stage_name} FAILED after {elapsed:.1f}s")
        print(f"    Error: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Electricity Theft Detection Pipeline")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="Skip Optuna tuning (use baseline GridSearch only)")
    parser.add_argument("--only-eval", action="store_true",
                        help="Only run evaluation and explainability")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training stages (use existing models)")
    args = parser.parse_args()

    pipeline_start = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        ELECTRICITY THEFT DETECTION — ADVANCED ML PIPELINE           ║
║        SGCC Dataset | XGBoost + LightGBM + CatBoost + SHAP          ║
║        MLflow Experiment Tracking | Optuna Hyperparameter Opt.       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    if not args.only_eval and not args.skip_train:
        # ── Stage 1: Data Preprocessing ──────────────────────────────────────
        from src.data_preprocessing import run_preprocessing
        run_stage("Data Preprocessing", run_preprocessing)

        # ── Stage 2: Feature Engineering ─────────────────────────────────────
        from src.feature_engineering import run_feature_engineering
        run_stage("Feature Engineering", run_feature_engineering)

        # ── Stage 3: EDA & Visualizations ────────────────────────────────────
        from notebooks.eda import run_eda
        run_stage("EDA & Visualizations (10 plots)", run_eda)

        # ── Stage 4: Baseline Training (GridSearchCV) ─────────────────────────
        from src.train import run_training
        run_stage("Baseline Training (LR + RF + XGBoost + GridSearchCV)", run_training)

        # ── Stage 5: Optuna Optimization ─────────────────────────────────────
        if not args.skip_optuna:
            from src.train_optuna import run_optuna_training
            run_stage("Advanced Optuna Optimization (LightGBM + XGBoost + CatBoost)",
                      run_optuna_training)
        else:
            print("\n  [Skipping Optuna — use --skip-optuna=False to enable]")

    # ── Stage 6: Evaluation & Comparison ─────────────────────────────────────
    from src.evaluate import run_evaluation
    run_stage("Model Evaluation & Benchmarking", run_evaluation)

    # ── Stage 7: SHAP Explainability ─────────────────────────────────────────
    from src.explain import run_explanation
    run_stage("SHAP Explainability", run_explanation)

    # ── Done ─────────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    banner(f"PIPELINE COMPLETE — Total time: {total/60:.1f} minutes")

    print("""
  Output Summary:
  ─────────────────────────────────────────────
  Figures       → results/figures/   (17+ plots incl. threshold curves & error analysis)
  Metrics       → results/metrics/   (model_comparison.csv, benchmark_comparison.csv, error_analysis.csv)
  Models        → models/            (LR, RF, XGBoost, LightGBM, CatBoost .pkl files)
  Notebook      → notebooks/eda.ipynb  (interactive EDA with SMOTE + PCA scree)
  MLflow Runs   → mlruns/            (run: mlflow ui → http://localhost:5000)

  Key Improvements Over Baseline (threshold tuning):
  ─────────────────────────────────────────────────
  • Each model's decision threshold is tuned to maximize F1
  • Error analysis saved: FPR, FNR, TP/TN/FP/FN per model
  • Optuna-tuned LightGBM, XGBoost, CatBoost included in comparison

  Next Steps:
  ─────────────────────────────────────────────
  1. View MLflow:   mlflow ui
  2. Start API:     uvicorn api.main:app --reload --port 8000
  3. Start UI:      python app/app.py  (Gradio)
  4. Docker:        docker-compose up --build
  5. Deploy HF:     python deploy_hf.py
    """)


if __name__ == "__main__":
    main()
