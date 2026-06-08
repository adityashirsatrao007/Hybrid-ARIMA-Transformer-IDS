import torch
import torch.nn as nn
import numpy as np

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size=1, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, 
            nhead=1, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: [batch_size, sequence_length, feature_size]
        output = self.transformer_encoder(src)
        # We only care about the prediction for the last time step
        output = self.decoder(output[:, -1, :])
        return output

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_transformer(residuals, seq_length=10, epochs=10, lr=0.001):
    print("Preparing data for Transformer...")
    X, y = create_sequences(residuals, seq_length)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    model = TimeSeriesTransformer(feature_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training Transformer for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            
    print("Transformer training complete.")
    return model, seq_length

if __name__ == "__main__":
    # Quick test
    dummy_residuals = np.random.normal(0, 0.1, (1000, 1))
    model, seq_len = train_transformer(dummy_residuals, epochs=5)
