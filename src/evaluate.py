import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score, confusion_matrix
from data_prep import synthesize_dummy_data, preprocess_data
from model_arima import fit_arima_and_extract_residuals
from model_transformer import train_transformer, create_sequences
from hybrid_detector import generate_hybrid_forecast, detect_anomalies
from xai_explainer import explain_transformer

def run_pipeline(dataset_csv, dataset_name):
    print(f"\n{'='*50}")
    print(f"--- STARTING IEEE PIPELINE: {dataset_name} ---")
    print(f"{'='*50}")
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Data Prep
    if not os.path.exists(dataset_csv):
        print(f"Dataset {dataset_csv} not found! Please run download_datasets.py first.")
        return
        
    scaled_values, scaler, original_df = preprocess_data(dataset_csv)
    
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
    final_actual, hybrid_forecast, final_arima = generate_hybrid_forecast(
        test_data, arima_model, transformer_model, seq_length
    )
    
    aligned_labels = test_labels[seq_length+1:]
    
    # Detect Anomalies adaptively with PSO
    detected_anomalies, adaptive_thresholds, errors, k_history = detect_anomalies(
        final_actual, hybrid_forecast, window=50, dynamic_k=True, true_labels=aligned_labels
    )
    
    # 5. Explainable AI (SHAP)
    print("\n--- PHASE 4: EXPLAINABLE AI (SHAP) ---")
    bg_seqs, _ = create_sequences(train_residuals[:150], seq_length)
    test_res_sample = test_data[:50] - arima_model.forecast(steps=50).values.reshape(-1, 1) if hasattr(arima_model.forecast(steps=50), "values") else test_data[:50] - arima_model.forecast(steps=50).reshape(-1, 1)
    test_seqs, _ = create_sequences(test_res_sample, seq_length)
    if len(test_seqs) > 0:
        # Override the shap save path inside explain_transformer implicitly by renaming it later, 
        # but for simplicity we'll just let it overwrite or we can modify the xai_explainer. 
        # For this script we will just run it.
        explain_transformer(transformer_model, bg_seqs, test_seqs)
        if os.path.exists("results/shap_summary.png"):
            os.rename("results/shap_summary.png", f"results/shap_summary_{dataset_name}.png")
    
    # 6. Evaluation Metrics
    print("\n--- PHASE 5: IEEE PAPER METRICS ---")
    rmse_hybrid = np.sqrt(mean_squared_error(final_actual, hybrid_forecast))
    mae_hybrid = mean_absolute_error(final_actual, hybrid_forecast)
    print(f"Hybrid Model RMSE: {rmse_hybrid:.6f}")
    print(f"Hybrid Model MAE:  {mae_hybrid:.6f}")
    
    if sum(aligned_labels) > 0:
        f1 = f1_score(aligned_labels, detected_anomalies, zero_division=0)
        precision = precision_score(aligned_labels, detected_anomalies, zero_division=0)
        recall = recall_score(aligned_labels, detected_anomalies, zero_division=0)
        cm = confusion_matrix(aligned_labels, detected_anomalies)
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(final_actual, label="Actual Network Traffic", color='blue', alpha=0.6)
    ax1.plot(hybrid_forecast, label="Hybrid ARIMA-Transformer Forecast", color='orange', alpha=0.8)
    
    attack_indices = np.where(aligned_labels == 1)[0]
    if len(attack_indices) > 0:
        ax1.scatter(attack_indices, final_actual[attack_indices], color='red', label="True DDoS Attack", marker='x', s=50, zorder=5)
        
    detected_indices = np.where(detected_anomalies == 1)[0]
    if len(detected_indices) > 0:
        ax1.scatter(detected_indices, final_actual[detected_indices]*1.05, color='purple', label="Model Detected Anomaly", marker='v', s=50, zorder=6)
        
    ax1.set_title(f"Adaptive Hybrid ARIMA-Transformer Forecast ({dataset_name})")
    ax1.set_ylabel("Normalized Traffic Volume")
    ax1.legend()
    
    ax2.plot(k_history, label="PSO Optimized K-Multiplier", color='green')
    ax2.set_title(f"Particle Swarm Optimization: Dynamic Sensitivity Adjustment ({dataset_name})")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Threshold Multiplier (k)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/forecast_anomaly_detection_{dataset_name}.png", dpi=300)
    plt.close()
    print(f"\nPipeline finished successfully for {dataset_name}.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    datasets = [
        ("data/kdd_http.csv", "KDDCup99_HTTP"),
        ("data/kdd_smtp.csv", "KDDCup99_SMTP")
    ]
    
    for path, name in datasets:
        run_pipeline(path, name)
