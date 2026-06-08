import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

print("Loading Kaggle KDDCup99 dataset...")
# Load real KDDCup99 from Kaggle download
df = pd.read_csv("data/kddcup99.csv")
print(f"Loaded {len(df)} rows.")

# Sample 10000 rows to keep it fast
df = df.sample(10000, random_state=42)

# Find categorical columns and drop them, keep numeric
numeric_df = df.select_dtypes(include=[np.number])

print(f"Using {len(numeric_df.columns)} numeric features...")

# Standardize
scaler = StandardScaler()
scaled_X = scaler.fit_transform(numeric_df.fillna(0))

# Reduce to 1D Traffic Volume using PCA
pca = PCA(n_components=1)
traffic_signal = pca.fit_transform(scaled_X).flatten()

# Normalize to [0, 1]
traffic_signal = (traffic_signal - np.min(traffic_signal)) / (np.max(traffic_signal) - np.min(traffic_signal))

# The KDD dataset from Kaggle has the label column usually as 'label'
label_col = 'label'
if 'label' not in df.columns:
    label_col = df.columns[-1] # Usually the last column

labels = np.array([0 if str(val).strip().lower() in ['normal', 'normal.'] else 1 for val in df[label_col]])

out_df = pd.DataFrame({
    "Timestamp": range(len(traffic_signal)),
    "Value": traffic_signal,
    "Is_Attack": labels
})

out_path = "data/kaggle_kddcup99.csv"
out_df.to_csv(out_path, index=False)
print(f"Saved real dataset to {out_path} (Attacks: {sum(labels)})")
