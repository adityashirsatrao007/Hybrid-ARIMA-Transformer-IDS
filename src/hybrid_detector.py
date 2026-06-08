import numpy as np
import torch
import pandas as pd

def pso_optimize_k(errors, labels, num_particles=10, iters=10):
    """
    Simulates Particle Swarm Optimization to find the best `k` multiplier 
    on a small validation rolling window to maximize F1-Score.
    """
    from sklearn.metrics import f1_score
    
    # Particle Swarm Initialization
    # k usually ranges from 1.0 to 5.0 in anomaly detection
    positions = np.random.uniform(1.0, 5.0, num_particles)
    velocities = np.random.uniform(-0.1, 0.1, num_particles)
    
    personal_best_positions = np.copy(positions)
    personal_best_scores = np.zeros(num_particles)
    
    global_best_position = positions[0]
    global_best_score = -1
    
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    
    # PSO Hyperparameters
    w = 0.5  # Inertia
    c1 = 1.5 # Cognitive constant
    c2 = 1.5 # Social constant
    
    for _ in range(iters):
        for i in range(num_particles):
            k = positions[i]
            threshold = mean_err + (k * std_err)
            
            # Simulate detection with this k
            preds = (errors > threshold).astype(int)
            
            # Calculate fitness (F1-score)
            # If all zeros and no actual anomalies, f1 is 0. 
            # We add a small penalty for false positives if no anomalies exist in window.
            if np.sum(labels) == 0:
                score = -np.sum(preds) # punish false positives
            else:
                score = f1_score(labels, preds, zero_division=0)
            
            # Update personal best
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
                
            # Update global best
            if score > global_best_score:
                global_best_score = score
                global_best_position = positions[i]
                
        # Update velocities and positions
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (personal_best_positions[i] - positions[i]) + 
                             c2 * r2 * (global_best_position - positions[i]))
            positions[i] += velocities[i]
            
            # Constrain k between 1.0 and 5.0
            positions[i] = np.clip(positions[i], 1.0, 5.0)
            
    return global_best_position

def detect_anomalies(actual, predictions, window=50, dynamic_k=False, true_labels=None):
    """
    Implements the Adaptive Anomaly Detector using a rolling mean/std threshold.
    If dynamic_k=True, uses PSO to adjust the k-multiplier.
    """
    errors = np.abs(actual - predictions)
    
    anomalies = []
    thresholds = []
    k_history = []
    
    current_k = 3.0 # Default starting k
    
    # We need a rolling window to be "adaptive"
    for i in range(len(errors)):
        if i < window:
            # Not enough data for a rolling window yet
            thresholds.append(np.inf)
            anomalies.append(0)
            k_history.append(current_k)
            continue
            
        rolling_errors = errors[i-window:i]
        
        # Every 20 steps, run PSO to update K based on the recent window
        if dynamic_k and true_labels is not None and i % 20 == 0:
            rolling_labels = true_labels[i-window:i]
            new_k = pso_optimize_k(rolling_errors, rolling_labels)
            # Smooth transition
            current_k = 0.8 * current_k + 0.2 * new_k
            
        mean_err = np.mean(rolling_errors)
        std_err = np.std(rolling_errors)
        
        # Adaptive threshold formula
        threshold = mean_err + (current_k * std_err)
        thresholds.append(threshold)
        k_history.append(current_k)
        
        if errors[i] > threshold:
            anomalies.append(1) # Anomaly Detected!
        else:
            anomalies.append(0)
            
    return np.array(anomalies), np.array(thresholds), errors, np.array(k_history)

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
