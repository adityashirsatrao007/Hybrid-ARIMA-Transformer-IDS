import numpy as np
import torch
import pandas as pd

def detect_anomalies(actual, predictions, window=50, k=3):
    """
    Implements the Adaptive Anomaly Detector using a rolling mean/std threshold.
    """
    errors = np.abs(actual - predictions)
    
    anomalies = []
    thresholds = []
    
    # We need a rolling window to be "adaptive"
    for i in range(len(errors)):
        if i < window:
            # Not enough data for a rolling window yet
            thresholds.append(np.inf)
            anomalies.append(0)
            continue
            
        rolling_errors = errors[i-window:i]
        mean_err = np.mean(rolling_errors)
        std_err = np.std(rolling_errors)
        
        # Adaptive threshold formula
        threshold = mean_err + (k * std_err)
        thresholds.append(threshold)
        
        if errors[i] > threshold:
            anomalies.append(1) # Anomaly Detected!
        else:
            anomalies.append(0)
            
    return np.array(anomalies), np.array(thresholds), errors

def generate_hybrid_forecast(test_data, arima_model, transformer_model, seq_length):
    """
    Generates the final forecast: ARIMA (Linear) + Transformer (Non-Linear)
    """
    print("Generating Hybrid Forecast...")
    
    # 1. Get ARIMA prediction for the test set
    arima_pred = arima_model.forecast(steps=len(test_data))
    arima_pred_vals = arima_pred.values.reshape(-1, 1) if hasattr(arima_pred, "values") else arima_pred.reshape(-1, 1)
    
    # Calculate initial residuals (we won't know future actuals in production, 
    # but for testing against our test_data we do)
    test_residuals = test_data - arima_pred_vals
    
    # 2. Get Transformer prediction for the residuals
    # We need a sequence to predict the next step.
    # For a true rolling forecast, we'd predict 1 step, append, and repeat.
    # For simplicity in this demo, we'll use the create_sequences approach on the test_residuals.
    
    from model_transformer import create_sequences
    X_test, y_test_res = create_sequences(test_residuals, seq_length)
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    transformer_model.eval()
    with torch.no_grad():
        transformer_pred_res = transformer_model(X_tensor).numpy()
        
    # Align the arrays (since Transformer loses 'seq_length' steps at the start)
    final_actual = test_data[seq_length+1:]
    final_arima = arima_pred_vals[seq_length+1:]
    
    # Hybrid Forecast = Linear + Non-Linear
    hybrid_forecast = final_arima + transformer_pred_res
    
    print("Hybrid Forecasting complete.")
    return final_actual, hybrid_forecast, final_arima

if __name__ == "__main__":
    print("Hybrid Detector module loaded.")
