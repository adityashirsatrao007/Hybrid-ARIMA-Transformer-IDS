import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score, confusion_matrix
from data_prep import synthesize_dummy_data, preprocess_data
from model_arima import fit_arima_and_extract_residuals
from model_transformer import train_transformer, create_sequences
from hybrid_detector import generate_hybrid_forecast, detect_anomalies
from xai_explainer import explain_transformer

def run_pipeline():
    print("--- STARTING IEEE ARIMA-TRANSFORMER HYBRID PIPELINE ---")
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Data Prep
    if not os.path.exists("data/network_traffic.csv"):
        synthesize_dummy_data()
    scaled_values, scaler, original_df = preprocess_data()
    
    # Train-test split (80-20)
    train_size = int(len(scaled_values) * 0.8)
    train_data = scaled_values[:train_size]
    test_data = scaled_values[train_size:]
    test_labels = original_df["Is_Attack"].values[train_size:]
    
    # 2. ARIMA Linear Baseline
    print("\n--- PHASE 1: LINEAR MODELING (ARIMA) ---")
    arima_model, train_residuals, _ = fit_arima_and_extract_residuals(train_data)
    
    # 3. Transformer Non-Linear Modeling
    print("\n--- PHASE 2: NON-LINEAR MODELING (TRANSFORMER) ---")
    seq_length = 10
    transformer_model, _ = train_transformer(train_residuals, seq_length=seq_length, epochs=5)
    
    # 4. Hybrid Forecasting
    print("\n--- PHASE 3: HYBRID FORECASTING & ANOMALY DETECTION ---")
    # For a real pipeline, we'd do step-by-step rolling forecast. 
    # For this proof-of-concept, we forecast the test set block.
    final_actual, hybrid_forecast, final_arima = generate_hybrid_forecast(
        test_data, arima_model, transformer_model, seq_length
    )
    
    # Align labels with the truncated test data (due to sequence length)
    aligned_labels = test_labels[seq_length+1:]
    
    # Detect Anomalies adaptively
    detected_anomalies, adaptive_thresholds, errors = detect_anomalies(
        final_actual, hybrid_forecast, window=50, k=3
    )
    
    # 5. Explainable AI (SHAP)
    print("\n--- PHASE 4: EXPLAINABLE AI (SHAP) ---")
    # Take a small sample of training residuals for the SHAP background
    bg_seqs, _ = create_sequences(train_residuals[:150], seq_length)
    # Take a small sample of test residuals to explain
    test_res_sample = test_data[:50] - arima_model.forecast(steps=50).values.reshape(-1, 1) if hasattr(arima_model.forecast(steps=50), "values") else test_data[:50] - arima_model.forecast(steps=50).reshape(-1, 1)
    test_seqs, _ = create_sequences(test_res_sample, seq_length)
    if len(test_seqs) > 0:
        explain_transformer(transformer_model, bg_seqs, test_seqs)
    
    # 6. Evaluation Metrics
    print("\n--- PHASE 5: IEEE PAPER METRICS ---")
    
    # Forecasting metrics
    rmse_hybrid = np.sqrt(mean_squared_error(final_actual, hybrid_forecast))
    mae_hybrid = mean_absolute_error(final_actual, hybrid_forecast)
    print(f"Hybrid Model RMSE: {rmse_hybrid:.6f}")
    print(f"Hybrid Model MAE:  {mae_hybrid:.6f}")
    
    # Anomaly metrics
    if sum(aligned_labels) > 0:
        f1 = f1_score(aligned_labels, detected_anomalies, zero_division=0)
        precision = precision_score(aligned_labels, detected_anomalies, zero_division=0)
        recall = recall_score(aligned_labels, detected_anomalies, zero_division=0)
        cm = confusion_matrix(aligned_labels, detected_anomalies)
        
        # False Positive Rate = FP / (FP + TN)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"Cyber Attack Detection F1-Score: {f1:.4f}")
        print(f"Cyber Attack Detection Precision: {precision:.4f}")
        print(f"Cyber Attack Detection Recall: {recall:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
    else:
        print("No attacks in the test set to evaluate F1-score.")
        
    # Generate Visualizations
    print("\nGenerating IEEE Figures...")
    plt.figure(figsize=(15, 6))
    plt.plot(final_actual, label="Actual Network Traffic", color='blue', alpha=0.6)
    plt.plot(hybrid_forecast, label="Hybrid ARIMA-Transformer Forecast", color='orange', alpha=0.8)
    
    # Mark real attacks
    attack_indices = np.where(aligned_labels == 1)[0]
    if len(attack_indices) > 0:
        plt.scatter(attack_indices, final_actual[attack_indices], color='red', label="True DDoS Attack", marker='x', s=50, zorder=5)
        
    # Mark detected anomalies
    detected_indices = np.where(detected_anomalies == 1)[0]
    if len(detected_indices) > 0:
        # Move up slightly so they don't overlap perfectly with True attacks visually
        plt.scatter(detected_indices, final_actual[detected_indices]*1.05, color='purple', label="Model Detected Anomaly", marker='v', s=50, zorder=6)
        
    plt.title("Adaptive Hybrid ARIMA-Transformer Forecast & Anomaly Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Traffic Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/forecast_anomaly_detection.png", dpi=300)
    plt.close()
    
    print("\nPipeline finished successfully. Artifacts saved to 'results/' directory.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_pipeline()
