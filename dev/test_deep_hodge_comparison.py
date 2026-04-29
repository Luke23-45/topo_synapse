import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import sys

# Ensure synapse is in path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from synapse.synapse_core.proxy import DifferentiableHodgeProxy
from synapse.synapse_arch.deep_hodge import DeepHodgeTransformer

# --- Synthetic Data Generation ---
# Task: Distinguish a circular trajectory (loop) from a linear trajectory
def generate_synthetic_data(num_samples=1000, max_points=16, d_model=8):
    torch.manual_seed(42)
    X = torch.zeros(num_samples, max_points, d_model)
    y = torch.zeros(num_samples, dtype=torch.float32)
    
    for i in range(num_samples):
        is_loop = i % 2 == 0
        y[i] = 1.0 if is_loop else 0.0
        
        # Add a lot of noise so it's not trivial
        noise = torch.randn(max_points, d_model) * 0.3
        
        t = torch.linspace(0, 1, max_points)
        if is_loop:
            # Circle (Loop)
            X[i, :, 0] = torch.cos(2 * torch.pi * t)
            X[i, :, 1] = torch.sin(2 * torch.pi * t)
        else:
            # Line (No loop)
            X[i, :, 0] = t * 2 - 1
            X[i, :, 1] = 0.0
            
        X[i] += noise
        
    return X, y

# --- Baseline Model (Global Eigenvalue Proxy) ---
class BaselineProxyModel(nn.Module):
    def __init__(self, d_model=8, k_dim=2, num_scales=3, max_points=16):
        super().__init__()
        self.geom_proj = nn.Linear(d_model, k_dim)
        
        self.proxy = DifferentiableHodgeProxy(
            lift_dim=k_dim,
            hidden_dim=d_model,
            num_scales=num_scales,
            num_eigs=4,
            max_points=max_points,
            tau=1e-4
        )
        
        # Readout from proxy features
        self.readout = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        B, K, D = x.shape
        # Project to geometric space
        P = self.geom_proj(x)
        
        # Dummy proxy weights (assume all anchors selected)
        y_star = torch.ones(B, K, device=x.device)
        
        proxy_features = self.proxy(P, y_star)
        logits = self.readout(proxy_features).squeeze(-1)
        return logits

# --- New Model (Deep Hodge Transformer) ---
class DeepHodgeModel(nn.Module):
    def __init__(self, d_model=8, k_dim=2, num_scales=3, max_points=16, num_layers=2):
        super().__init__()
        # The transformer uses d_model and projects to k_dim internally
        self.transformer = DeepHodgeTransformer(
            num_layers=num_layers,
            d_model=d_model,
            k_dim=k_dim,
            num_scales=num_scales,
            max_points=max_points
        )
        
        self.readout = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        # Topology-routed sequence
        x_out = self.transformer(x)
        
        # Mean pooling
        pooled = x_out.mean(dim=1)
        logits = self.readout(pooled).squeeze(-1)
        return logits

def train_model(model, name, train_loader, test_loader, epochs=15):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        acc = correct / total * 100
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")
            
    print(f"Finished in {time.time() - start_time:.2f}s | Final Accuracy: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    X, y = generate_synthetic_data(num_samples=200, max_points=16, d_model=8)
    
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test")
    print("Task: Classify circular loop trajectory vs linear trajectory (noisy)")
    
    baseline = BaselineProxyModel()
    acc_baseline = train_model(baseline, "Baseline (Eigenvalue Proxy)", train_loader, test_loader, epochs=5)
    
    deep_hodge = DeepHodgeModel(num_layers=2)
    acc_hodge = train_model(deep_hodge, "Deep Hodge-Transformer", train_loader, test_loader, epochs=5)
    
    print(f"\n=== SUMMARY ===")
    print(f"Baseline Accuracy: {acc_baseline:.2f}%")
    print(f"Deep Hodge Accuracy: {acc_hodge:.2f}%")
    print(f"Absolute Improvement: {acc_hodge - acc_baseline:+.2f}%")
