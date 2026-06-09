# Hybrid ARIMA-Transformer Intrusion Detection System (IDS)

This repository contains the official codebase for the IEEE research paper proposing a novel **Hybrid ARIMA-Transformer Architecture** with **Particle Swarm Optimization (PSO)** and **Explainable AI (SHAP)** for robust network anomaly detection.

## 📌 Architecture Overview
Modern cyber attacks present highly complex, non-linear patterns that evade traditional linear models, while pure deep learning models often lack interpretability. Our hybrid approach solves this:
1. **Linear Modeling (ARIMA):** Captures and filters out predictable, linear time-series trends from network traffic.
2. **Non-Linear Modeling (PyTorch Transformer):** A Multi-Head Self-Attention Transformer encoder learns the complex, non-linear attack signatures from the ARIMA residuals.
3. **Adaptive Thresholding (PSO):** Particle Swarm Optimization dynamically adjusts the anomaly detection sensitivity threshold (optimizing for F2-Score).
4. **Explainable AI (SHAP):** Provides a transparent, feature-level explanation of why the Transformer flagged specific traffic as an attack.

## 🚀 Key Results
Evaluated on benchmark network traffic datasets (KDD Cup 99 / Kaggle Real), the model achieves state-of-the-art anomaly detection metrics:
*   **ROC AUC Score:** `0.9586`
*   **Precision:** `99.18%`
*   **Recall:** `23.21%`
*   **False Positive Rate (FPR):** `0.71%` (Extremely low false alarm rate)

*Note: All performance graphs (ROC Curve, Precision-Recall Curve, Confusion Matrix, and SHAP Summary) are formatted for IEEE two-column layout and are available in the `results/` directory.*

## 📂 Project Structure
*   `src/data_prep.py`: Data loading, scaling, and sequence generation.
*   `src/model_arima.py`: Baseline linear forecasting.
*   `src/model_transformer.py`: PyTorch Transformer encoder training.
*   `src/hybrid_detector.py`: PSO optimization and anomaly thresholding.
*   `src/xai_explainer.py`: SHAP integration for Explainable AI.
*   `src/evaluate.py`: End-to-end pipeline execution and IEEE metric/graph generation.
*   `results/`: Automatically generated IEEE-formatted PNG charts.

## ⚙️ Setup & Execution
1. Install requirements:
   ```bash
   pip install pandas numpy scikit-learn statsmodels torch pyswarm shap matplotlib
   ```
2. Run the end-to-end evaluation pipeline:
   ```bash
   python src/evaluate.py
   ```
This script will preprocess the data, train both the ARIMA and Transformer models, optimize the threshold using PSO, calculate SHAP values, and output the final IEEE metrics and graphs into the `results/` folder.
