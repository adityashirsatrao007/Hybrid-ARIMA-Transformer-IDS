import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def explain_transformer(transformer_model, background_data, test_sample, feature_names=None):
    """
    Uses SHAP (DeepExplainer) to explain the Transformer's non-linear predictions.
    This proves to reviewers that the model isn't a black box.
    """
    print("Initializing SHAP Explainer...")
    
    transformer_model.eval()
    
    # SHAP DeepExplainer requires a background dataset to integrate over
    # We use a random subset of training data as background (e.g., 100 samples)
    bg_tensor = torch.tensor(background_data, dtype=torch.float32)
    
    # Define a wrapper that only outputs the final prediction
    # DeepExplainer expects the model to take input and return a single output tensor
    explainer = shap.DeepExplainer(transformer_model, bg_tensor)
    
    print("Calculating SHAP values for test sample...")
    test_tensor = torch.tensor(test_sample, dtype=torch.float32)
    shap_values = explainer.shap_values(test_tensor)
    
    # Provide default feature names if none provided
    if feature_names is None:
        seq_length = test_sample.shape[1]
        feature_names = [f"t-{seq_length-i}" for i in range(seq_length)]
        
    os.makedirs("results", exist_ok=True)
        
    # Generate Summary Plot
    # SHAP values shape from DeepExplainer might need squeezing
    shap_vals_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
    
    # Reshape for plotting (Flattening sequence dimension for simple visualization)
    # This shows which time step in the lookback window had the biggest impact
    shap.summary_plot(shap_vals_to_plot.reshape(test_sample.shape[0], -1), 
                      test_sample.reshape(test_sample.shape[0], -1), 
                      feature_names=feature_names, 
                      show=False)
    
    plt.title("SHAP Summary Plot: Feature Importance")
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=300)
    plt.close()
    print("SHAP Summary Plot saved to results/shap_summary.png")

if __name__ == "__main__":
    print("XAI Explainer module loaded.")
