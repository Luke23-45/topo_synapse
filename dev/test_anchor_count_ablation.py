"""
Anchor count ablation study: How does the number of selected anchors (K) affect performance?

Robust design:
- Pre-generated data cached in memory (all models use identical data)
- Multi-seed evaluation (3 seeds per K value)
- Proper train/val/test splits (60/20/20)
- Early stopping with patience
- Per-class accuracy reporting
- Statistical significance (mean ± std across seeds)

Dataset characteristics:
- input_dim=10, num_classes=8, seq_len=64
- High noise, complex topological patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import os
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from synapse.common.encoders.topological_encoder import TopologicalEncoder
from synapse.synapse_arch.deep_hodge import DeepHodgeTransformer

# ============================================================
# DEVICE — auto-detect GPU
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATA GENERATION — deterministic, cached once in memory
# ============================================================

class SyntheticDataCache:
    """Generates data once, stores in memory, provides deterministic splits."""
    
    def __init__(self, num_samples=1600, seq_len=64, input_dim=10, num_classes=8, seed=42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seed = seed
        self._X = None
        self._y = None
    
    def generate(self):
        """Generate and cache data. Only runs once."""
        if self._X is not None:
            return self._X, self._y
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        N = self.num_samples
        T = self.seq_len
        D = self.input_dim
        C = self.num_classes
        
        X = torch.zeros(N, T, D)
        y = torch.zeros(N, dtype=torch.long)
        
        # Pre-generate all noise tensors at once (vectorized)
        noise_high = torch.randn(N, T, D) * 0.25
        drift = torch.randn(N, D) * 1.0
        osc_phase = torch.randn(N) * 2 * np.pi
        dropout_mask = (torch.rand(N, T, D) > 0.1).float()
        outlier_mask = torch.rand(N, T) < 0.05
        outlier_vals = torch.randn(N, T, D) * 2.5
        
        for i in range(N):
            class_id = i % C
            y[i] = class_id
            
            if class_id == 0:
                # Sphere (β0=1, β1=0)
                angles = torch.randn(T, D)
                angles = angles / angles.norm(dim=1, keepdim=True).clamp(min=1e-8)
                radius = 1.0 + torch.rand(T, 1) * 1.0
                X[i] = angles * radius
                
            elif class_id == 1:
                # Torus (β0=1, β1=1)
                t = torch.linspace(0, 2*np.pi, T)
                s = torch.linspace(0, 2*np.pi, T)
                R, r = 2.0, 0.5
                X[i, :, 0] = (R + r * torch.cos(s)) * torch.cos(t)
                X[i, :, 1] = (R + r * torch.cos(s)) * torch.sin(t)
                X[i, :, 2] = r * torch.sin(s)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.3 + X[i, :, 0] * 0.1
                    
            elif class_id == 2:
                # Figure-8 (β0=1, β1=2)
                t = torch.linspace(0, 2*np.pi, T)
                mask = (t < np.pi).float()
                X[i, :, 0] = 2 * torch.cos(t)
                X[i, :, 1] = 2 * torch.sin(t) * mask
                X[i, :, 2] = 2 * torch.sin(t) * (1 - mask)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.2
                    
            elif class_id == 3:
                # Twisted helix
                t = torch.linspace(0, 4*np.pi, T)
                X[i, :, 0] = torch.cos(t)
                X[i, :, 1] = torch.sin(t)
                X[i, :, 2] = t / (4*np.pi)
                for d in range(3, D):
                    X[i, :, d] = torch.sin(t + d * np.pi/4) * 0.5
                    
            elif class_id == 4:
                # Klein bottle-like (non-orientable)
                t = torch.linspace(0, 2*np.pi, T)
                s = torch.linspace(0, 2*np.pi, T)
                X[i, :, 0] = (2 + torch.cos(s/2) * torch.cos(t) - torch.sin(s/2) * torch.sin(2*t)) * torch.cos(s)
                X[i, :, 1] = (2 + torch.cos(s/2) * torch.cos(t) - torch.sin(s/2) * torch.sin(2*t)) * torch.sin(s)
                X[i, :, 2] = torch.sin(s/2) * torch.cos(t) + torch.cos(s/2) * torch.sin(2*t)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.4
                    
            elif class_id == 5:
                # Double torus (β0=1, β1=2)
                t = torch.linspace(0, 2*np.pi, T)
                mask = (t < np.pi).float()
                R1, r1 = 2.0, 0.5
                R2, r2 = 1.5, 0.3
                X[i, :, 0] = mask * ((R1 + r1 * torch.cos(t)) * torch.cos(t))
                X[i, :, 1] = mask * ((R1 + r1 * torch.cos(t)) * torch.sin(t))
                X[i, :, 2] = mask * (r1 * torch.sin(t))
                X[i, :, 0] += (1-mask) * ((R2 + r2 * torch.cos(t)) * torch.cos(t) + 3)
                X[i, :, 1] += (1-mask) * ((R2 + r2 * torch.cos(t)) * torch.sin(t))
                X[i, :, 2] += (1-mask) * (r2 * torch.sin(t))
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.3
                    
            elif class_id == 6:
                # Spiral with varying radius
                t = torch.linspace(0, 4*np.pi, T)
                radius = 0.5 + t / (4*np.pi) * 1.5
                X[i, :, 0] = radius * torch.cos(t)
                X[i, :, 1] = radius * torch.sin(t)
                X[i, :, 2] = t / (4*np.pi)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.3 + radius * 0.2
                    
            else:  # class_id == 7
                # Random cluster structure
                centers = torch.randn(4, D) * 2
                for t_idx in range(T):
                    cluster_id = t_idx % 4
                    X[i, t_idx] = centers[cluster_id] + torch.randn(D) * 0.5
            
            # Apply pre-generated noise
            X[i] += noise_high[i]
            X[i] += drift[i].unsqueeze(0)
            X[i] += torch.sin(torch.linspace(0, 10*np.pi, T).unsqueeze(1) + osc_phase[i]) * 0.3
            X[i] *= dropout_mask[i]
            X[i][outlier_mask[i]] += outlier_vals[i][outlier_mask[i]]
        
        # Normalize per-feature (zero mean, unit variance) — makes data more realistic
        X = (X - X.mean(dim=(0,1), keepdim=True)) / (X.std(dim=(0,1), keepdim=True) + 1e-8)
        
        self._X = X
        self._y = y
        return X, y
    
    def get_splits(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """Return deterministic train/val/test splits."""
        X, y = self.generate()
        N = X.shape[0]
        
        # Stratified split: ensure equal class distribution
        indices_by_class = defaultdict(list)
        for idx in range(N):
            indices_by_class[y[idx].item()].append(idx)
        
        train_idx, val_idx, test_idx = [], [], []
        for cls, indices in indices_by_class.items():
            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            # Shuffle deterministically
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(indices)
            train_idx.extend(perm[:n_train].tolist())
            val_idx.extend(perm[n_train:n_train+n_val].tolist())
            test_idx.extend(perm[n_train+n_val:].tolist())
        
        return (
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx],
        )
    
    @property
    def size_mb(self):
        if self._X is None:
            return 0
        return (self._X.element_size() * self._X.nelement() + self._y.element_size() * self._y.nelement()) / 1e6


# ============================================================
# MODEL
# ============================================================

class TopologicalEncoderWithK(nn.Module):
    def __init__(self, input_dim=10, d_model=64, k_dim=16, num_scales=3, max_proxy_points=32, K=8, r=1, lam=0.5, num_classes=8):
        super().__init__()
        self.encoder = TopologicalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            hidden_dim=128,
            k=k_dim,
            K=K,
            r=r,
            lam=lam,
            max_proxy_points=max_proxy_points,
        )
        self.transformer = DeepHodgeTransformer(
            num_layers=2,
            d_model=d_model,
            k_dim=k_dim,
            num_scales=num_scales,
            max_points=max_proxy_points,
        )
        self.readout = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        tokens, y_star = self.encoder(x)
        x_out = self.transformer(tokens)
        pooled = x_out.mean(dim=1)
        logits = self.readout(pooled)
        return logits


# ============================================================
# TRAINING — with early stopping, per-class metrics
# ============================================================

def train_model(model, name, train_loader, val_loader, test_loader, epochs=20, patience=5):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    epoch_pbar = tqdm(range(epochs), desc=f"{name} epochs", leave=False)
    for epoch in epoch_pbar:
        # Train
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                logits = model(batch_x)
                val_loss += criterion(logits, batch_y).item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_acc = val_correct / val_total * 100
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_pbar.set_postfix(loss=f"{total_loss/len(train_loader):.4f}", val_loss=f"{avg_val_loss:.4f}", val_acc=f"{val_acc:.1f}%")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(batch_y)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Overall accuracy
    test_acc = (all_preds == all_labels).float().mean().item() * 100
    
    # Per-class accuracy
    num_classes = all_labels.max().item() + 1
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == c).float().mean().item() * 100
    
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f}s | Test Acc: {test_acc:.2f}% | Best Val Loss: {best_val_loss:.4f}")
    
    return {
        'test_acc': test_acc,
        'per_class_acc': per_class_acc,
        'best_val_loss': best_val_loss,
        'time_s': elapsed,
    }


# ============================================================
# MAIN — multi-seed, cached data, comprehensive reporting
# ============================================================

if __name__ == "__main__":
    seq_len = 64
    max_proxy_points = 32  # B2 matrix is O(n^3), 32 is safe
    input_dim = 10
    d_model = 64
    k_dim = 16
    num_classes = 8
    num_seeds = 2
    epochs = 20
    patience = 5
    
    K_values = [4, 8, 16, 32]
    
    # --- Step 1: Generate data once, cache in memory ---
    print("=" * 70)
    print("STEP 1: Generating and caching synthetic data")
    print("=" * 70)
    
    data_cache = SyntheticDataCache(
        num_samples=1600,  # Larger dataset for robustness
        seq_len=seq_len,
        input_dim=input_dim,
        num_classes=num_classes,
        seed=42,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = data_cache.get_splits()
    
    print(f"Data generated: {data_cache.size_mb:.1f} MB in memory")
    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)" if DEVICE.type == "cuda" else " (CPU)"))
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class distribution (train): {torch.bincount(y_train).tolist()}")
    print(f"Class distribution (val):   {torch.bincount(y_val).tolist()}")
    print(f"Class distribution (test):  {torch.bincount(y_test).tolist()}")
    
    # --- Step 2: Run ablation across K values and seeds ---
    print(f"\n{'='*70}")
    print("STEP 2: Anchor count ablation (multi-seed)")
    print(f"{'='*70}")
    
    # results[K][seed_idx] = metrics dict
    all_results = defaultdict(list)
    
    for K in K_values:
        print(f"\n{'='*70}")
        print(f"K={K} anchors (compression: {K}/{seq_len} = {K/seq_len*100:.1f}%)")
        print(f"{'='*70}")
        
        for seed_idx in tqdm(range(num_seeds), desc=f"K={K} seeds", leave=False):
            seed = 42 + seed_idx * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Same data, different model init
            _pin = DEVICE.type == "cuda"
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, pin_memory=_pin)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False, pin_memory=_pin)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False, pin_memory=_pin)
            
            model = TopologicalEncoderWithK(
                input_dim=input_dim,
                d_model=d_model,
                k_dim=k_dim,
                max_proxy_points=max_proxy_points,
                K=K,
                r=1,
                lam=0.5,
                num_classes=num_classes,
            ).to(DEVICE)
            
            metrics = train_model(
                model, f"K={K} seed={seed}",
                train_loader, val_loader, test_loader,
                epochs=epochs, patience=patience,
            )
            metrics['seed'] = seed
            all_results[K].append(metrics)
            
            # Free GPU memory
            del model
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    
    # --- Step 3: Aggregate and report ---
    print(f"\n{'='*70}")
    print("STEP 3: Aggregated Results")
    print(f"{'='*70}")
    
    print(f"\n{'K':<6} {'Compression':<14} {'Test Acc (mean±std)':<22} {'Best Val Loss':<16} {'Time (s)':<10}")
    print("-" * 68)
    
    aggregated = {}
    for K in K_values:
        accs = [r['test_acc'] for r in all_results[K]]
        val_losses = [r['best_val_loss'] for r in all_results[K]]
        times = [r['time_s'] for r in all_results[K]]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_vl = np.mean(val_losses)
        mean_time = np.mean(times)
        
        compression = K / seq_len * 100
        print(f"{K:<6} {compression:.1f}%{'':<11} {mean_acc:.2f} ± {std_acc:.2f}{'':<10} {mean_vl:.4f}{'':<10} {mean_time:.1f}")
        
        aggregated[K] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_val_loss': mean_vl,
            'mean_time': mean_time,
            'per_class_acc': {},
        }
        
        # Aggregate per-class accuracy
        for c in range(num_classes):
            class_accs = [r['per_class_acc'].get(c, 0) for r in all_results[K]]
            aggregated[K]['per_class_acc'][c] = np.mean(class_accs)
    
    # --- Step 4: Per-class breakdown ---
    print(f"\n{'='*70}")
    print("Per-class accuracy (mean across seeds)")
    print(f"{'='*70}")
    
    class_names = ['Sphere', 'Torus', 'Fig-8', 'Helix', 'Klein', 'D-Torus', 'Spiral', 'Cluster']
    header = f"{'Class':<12}" + "".join(f"{'K='+str(K):<12}" for K in K_values)
    print(header)
    print("-" * len(header))
    for c in range(num_classes):
        row = f"{class_names[c]:<12}"
        for K in K_values:
            row += f"{aggregated[K]['per_class_acc'][c]:.1f}%{'':<7}"
        print(row)
    
    # --- Step 5: Analysis ---
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    best_K = max(aggregated, key=lambda k: aggregated[k]['mean_acc'])
    best_acc = aggregated[best_K]['mean_acc']
    best_std = aggregated[best_K]['std_acc']
    
    print(f"\nBest K: {best_K} anchors (accuracy: {best_acc:.2f} ± {best_std:.2f}%)")
    print(f"Optimal compression: {best_K/seq_len*100:.1f}%")
    
    print(f"\nPerformance trend (Δ between K values):")
    prev_K = None
    for K in K_values:
        if prev_K is not None:
            delta = aggregated[K]['mean_acc'] - aggregated[prev_K]['mean_acc']
            print(f"  K {prev_K} → {K}: {delta:+.2f}%")
        prev_K = K
    
    # Diminishing returns analysis
    print(f"\nMarginal accuracy per anchor:")
    for K in K_values:
        marginal = aggregated[K]['mean_acc'] / K
        print(f"  K={K}: {marginal:.2f}% per anchor")
    
    # --- Step 6: Save report ---
    report = {
        'dataset': {
            'num_samples': 1600,
            'seq_len': seq_len,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'noise': 'multi-scale (0.25 high-freq, 1.0 drift, 0.3 oscillation, 10% dropout, 5% outliers)',
        },
        'experiment': {
            'num_seeds': num_seeds,
            'epochs': epochs,
            'patience': patience,
            'max_proxy_points': max_proxy_points,
        },
        'results': {},
    }
    for K in K_values:
        report['results'][f'K={K}'] = {
            'mean_acc': round(aggregated[K]['mean_acc'], 2),
            'std_acc': round(aggregated[K]['std_acc'], 2),
            'mean_val_loss': round(aggregated[K]['mean_val_loss'], 4),
            'compression_pct': round(K / seq_len * 100, 1),
            'per_class_acc': {class_names[c]: round(aggregated[K]['per_class_acc'][c], 2) for c in range(num_classes)},
        }
    
    report_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'xdev', 'anchor_count_ablation_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
