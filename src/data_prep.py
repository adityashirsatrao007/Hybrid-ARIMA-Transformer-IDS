import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def synthesize_dummy_data(output_path="data/network_traffic.csv", rows=5000):
    """
    Since public cybersecurity datasets are massive (often >30GB), we will generate
    a mathematically representative 5% dummy dataset that mimics normal linear 
    traffic with sudden chaotic spikes (anomalies/DDoS attacks) to build and test 
    our pipeline locally. You can later point this pipeline to the real CICIoT2023 CSVs.
    """
    print(f"Synthesizing {rows} rows of network traffic data...")
    
    # 1. Base Linear Traffic (ARIMA loves this)
    time_index = np.arange(rows)
    base_traffic = 1000 + 50 * np.sin(time_index / 50) + 20 * np.cos(time_index / 10)
    
    # 2. Random Noise (Normal fluctuations)
    noise = np.random.normal(0, 15, rows)
    traffic = base_traffic + noise
    
    # 3. Inject Anomalies (DDoS Spikes)
    labels = np.zeros(rows) # 0 = Normal, 1 = Attack
    anomaly_indices = [1000, 2500, 4000]
    for idx in anomaly_indices:
        # A massive spike over 10-20 seconds
        duration = np.random.randint(10, 20)
        spike_magnitude = np.random.uniform(500, 1500)
        traffic[idx:idx+duration] += spike_magnitude
        labels[idx:idx+duration] = 1
        
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2024-01-01", periods=rows, freq="S"),
        "TotalBytes": traffic,
        "Is_Attack": labels
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return df

def preprocess_data(input_path="data/network_traffic.csv"):
    """
    Loads and scales the time-series data for the Neural Network.
    Returns: The scaled values and the original dataframe.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    # We will forecast 'TotalBytes'
    scaled_values = scaler.fit_transform(df[["TotalBytes"]])
    
    print("Preprocessing complete.")
    return scaled_values, scaler, df

if __name__ == "__main__":
    synthesize_dummy_data()
    preprocess_data()
