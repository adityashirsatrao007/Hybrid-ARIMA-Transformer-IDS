import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os

def fit_arima_and_extract_residuals(train_data, p=2, d=0, q=2):
    """
    Fits an ARIMA model to the training data and extracts the residuals.
    These residuals are what the Transformer will learn from.
    """
    print("Fitting ARIMA model...")
    # Add a small constant to prevent errors if perfectly flat
    model = ARIMA(train_data, order=(p, d, q))
    fitted_model = model.fit()
    
    # Get the linear predictions
    predictions = fitted_model.predict(start=0, end=len(train_data)-1)
    
    # Calculate residuals (Actual - Predicted)
    pred_vals = predictions.values.reshape(-1, 1) if hasattr(predictions, "values") else predictions.reshape(-1, 1)
    residuals = train_data - pred_vals
    
    print("ARIMA fitting complete.")
    return fitted_model, residuals, pred_vals

def forecast_arima(fitted_model, steps):
    """ Forecasts future steps using the fitted ARIMA model. """
    forecast = fitted_model.forecast(steps=steps)
    return forecast

if __name__ == "__main__":
    from data_prep import preprocess_data
    
    # Quick test
    scaled_values, _, _ = preprocess_data()
    train_size = int(len(scaled_values) * 0.8)
    train_data = scaled_values[:train_size]
    
    fitted_model, residuals, predictions = fit_arima_and_extract_residuals(train_data)
    print(f"Residuals shape: {residuals.shape}")
