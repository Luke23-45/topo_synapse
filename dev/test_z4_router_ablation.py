"""
Z4 History-Aware Anchor Router Ablation Study

Compares the new Z4 learned router against the Z3 baselines:
1. Z3 Saliency selection (SoftSelectorProxy — current encoder)
2. Z3 Uniform selection (evenly spaced — best Z3 baseline)
3. Z4 History-Aware Router (L=1, single-stage)
4. Z4 History-Aware Router (L=2, two-stage with memory)

All variants produce the same number of tokens [B, K, d_model], so the
downstream transformer sees identical input shapes.  This isolates the
effect of *how* anchors are selected.

Design choices:
- K values: 8 and 32 (from anchor count ablation)
- Multi-seed (3 seeds) for statistical robustness
- Same data, same training protocol, same downstream model
- Only the encoder/selection mechanism differs
- Z4 variants receive feedback from the task loss during training
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

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from synapse.synapse_arch.deep_hodge import DeepHodgeTransformer
from synapse.synapse_core.event import CausalEventModel
from synapse.synapse_core.lift import dense_anchor_vectors
from synapse.synapse_arch.normalized_lift import NormalizedLift
from synapse.synapse_core.topology_features import structural_feature_dim
from synapse.common.encoders.topological_encoder import SoftSelectorProxy
from synapse.common.encoders.z4_topological_encoder import Z4TopologicalEncoder

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATA GENERATION (identical to anchor_selection_ablation)
# ============================================================

CLASS_NAMES = ["Sphere", "Torus", "Fig-8", "Helix", "Klein", "D-Torus", "Spiral", "Cluster"]


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
                angles = torch.randn(T, D)
                angles = angles / angles.norm(dim=1, keepdim=True).clamp(min=1e-8)
                radius = 1.0 + torch.rand(T, 1) * 1.0
                X[i] = angles * radius
            elif class_id == 1:
                t = torch.linspace(0, 2 * np.pi, T)
                s = torch.linspace(0, 2 * np.pi, T)
                R, r = 2.0, 0.5
                X[i, :, 0] = (R + r * torch.cos(s)) * torch.cos(t)
                X[i, :, 1] = (R + r * torch.cos(s)) * torch.sin(t)
                X[i, :, 2] = r * torch.sin(s)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.3 + X[i, :, 0] * 0.1
            elif class_id == 2:
                t = torch.linspace(0, 2 * np.pi, T)
                mask = (t < np.pi).float()
                X[i, :, 0] = 2 * torch.cos(t)
                X[i, :, 1] = 2 * torch.sin(t) * mask
                X[i, :, 2] = 2 * torch.sin(t) * (1 - mask)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.2
            elif class_id == 3:
                t = torch.linspace(0, 4 * np.pi, T)
                X[i, :, 0] = torch.cos(t)
                X[i, :, 1] = torch.sin(t)
                X[i, :, 2] = t / (4 * np.pi)
                for d in range(3, D):
                    X[i, :, d] = torch.sin(t + d * np.pi / 4) * 0.5
            elif class_id == 4:
                t = torch.linspace(0, 2 * np.pi, T)
                s = torch.linspace(0, 2 * np.pi, T)
                X[i, :, 0] = (2 + torch.cos(s / 2) * torch.cos(t) - torch.sin(s / 2) * torch.sin(2 * t)) * torch.cos(s)
                X[i, :, 1] = (2 + torch.cos(s / 2) * torch.cos(t) - torch.sin(s / 2) * torch.sin(2 * t)) * torch.sin(s)
                X[i, :, 2] = torch.sin(s / 2) * torch.cos(t) + torch.cos(s / 2) * torch.sin(2 * t)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.4
            elif class_id == 5:
                t = torch.linspace(0, 2 * np.pi, T)
                mask = (t < np.pi).float()
                R1, r1 = 2.0, 0.5
                R2, r2 = 1.5, 0.3
                X[i, :, 0] = mask * ((R1 + r1 * torch.cos(t)) * torch.cos(t))
                X[i, :, 1] = mask * ((R1 + r1 * torch.cos(t)) * torch.sin(t))
                X[i, :, 2] = mask * (r1 * torch.sin(t))
                X[i, :, 0] += (1 - mask) * ((R2 + r2 * torch.cos(t)) * torch.cos(t) + 3)
                X[i, :, 1] += (1 - mask) * ((R2 + r2 * torch.cos(t)) * torch.sin(t))
                X[i, :, 2] += (1 - mask) * (r2 * torch.sin(t))
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.3
            elif class_id == 6:
                t = torch.linspace(0, 4 * np.pi, T)
                radius = 0.5 + t / (4 * np.pi) * 1.5
                X[i, :, 0] = radius * torch.cos(t)
                X[i, :, 1] = radius * torch.sin(t)
                X[i, :, 2] = t / (4 * np.pi)
                for d in range(3, D):
                    X[i, :, d] = torch.randn(T) * 0.2
            elif class_id == 7:
                centers = torch.randn(5, D) * 2.0
                cluster_idx = torch.randint(0, 5, (T,))
                X[i] = centers[cluster_idx] + torch.randn(T, D) * 0.5

            X[i] = X[i] + noise_high[i]
            X[i] = X[i] + drift[i].unsqueeze(0)
            X[i] = X[i] + torch.sin(torch.arange(T).float() / T * 2 * np.pi + osc_phase[i]).unsqueeze(-1) * 0.3
            X[i] = X[i] * dropout_mask[i]
            X[i][outlier_mask[i]] = outlier_vals[i][outlier_mask[i]]

        self._X = X
        self._y = y
        return X, y

    @property
    def size_mb(self):
        if self._X is None:
            self.generate()
        return (self._X.element_size() * self._X.nelement() +
                self._y.element_size() * self._y.nelement()) / (1024 * 1024)

    def get_splits(self, train_ratio=0.6, val_ratio=0.2):
        if self._X is None:
            self.generate()

        N = self.num_samples
        indices = torch.randperm(N, generator=torch.Generator().manual_seed(self.seed))

        train_end = int(N * train_ratio)
        val_end = train_end + int(N * val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return (
            self._X[train_idx], self._y[train_idx],
            self._X[val_idx], self._y[val_idx],
            self._X[test_idx], self._y[test_idx],
        )


# ============================================================
# ENCODER VARIANTS
# ============================================================

class EncoderZ3Saliency(nn.Module):
    """Z3 Baseline: saliency-based anchor selection via SoftSelectorProxy."""

    def __init__(self, input_dim, d_model, k_dim=16, K=8, r=1, lam=0.5, max_proxy_points=16):
        super().__init__()
        self.K = K
        self.max_proxy_points = max_proxy_points
        self.d_model = d_model

        anchor_dim = structural_feature_dim(input_dim, include_selection=False)
        self.event_model = CausalEventModel(input_dim, 64)
        self.selector = SoftSelectorProxy(K=K, r=r, lam=lam)
        self.lift = NormalizedLift(anchor_dim, k_dim)
        self.topology_proj = nn.Linear(k_dim, d_model)

    def set_normalization(self, mu, sigma):
        self.lift.set_normalization(mu, sigma)

    def forward(self, x, feedback=None):
        event_scores, saliency_scores = self.event_model(x)
        y_star = self.selector(saliency_scores)

        dense_vectors = dense_anchor_vectors(x, saliency_scores)
        _, dense_lifted_cloud = self.lift(dense_vectors)

        B, N, k_dim_out = dense_lifted_cloud.shape
        K_eff = min(N, self.max_proxy_points)
        _, top_idx = torch.topk(y_star, K_eff, dim=1)
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, k_dim_out)
        cloud = torch.gather(dense_lifted_cloud, 1, top_idx_exp)

        tokens = self.topology_proj(cloud)
        return tokens, y_star


class EncoderZ3Uniform(nn.Module):
    """Z3 Baseline: uniform (evenly spaced) anchor selection."""

    def __init__(self, input_dim, d_model, k_dim=16, K=8, r=1, lam=0.5, max_proxy_points=16):
        super().__init__()
        self.K = K
        self.max_proxy_points = max_proxy_points
        self.d_model = d_model

        anchor_dim = structural_feature_dim(input_dim, include_selection=False)
        self.event_model = CausalEventModel(input_dim, 64)
        self.lift = NormalizedLift(anchor_dim, k_dim)
        self.topology_proj = nn.Linear(k_dim, d_model)

    def set_normalization(self, mu, sigma):
        self.lift.set_normalization(mu, sigma)

    def forward(self, x, feedback=None):
        event_scores, saliency_scores = self.event_model(x)

        B, T = saliency_scores.shape
        K_eff = min(T, self.max_proxy_points)

        uniform_positions = torch.linspace(0, T - 1, K_eff, device=x.device)
        uniform_idx = uniform_positions.long().unsqueeze(0).expand(B, -1)

        y_star = torch.zeros(B, T, device=x.device)
        y_star.scatter_(1, uniform_idx, 1.0)

        dense_vectors = dense_anchor_vectors(x, saliency_scores)
        _, dense_lifted_cloud = self.lift(dense_vectors)

        k_dim_out = dense_lifted_cloud.shape[-1]
        uniform_idx_exp = uniform_idx.unsqueeze(-1).expand(-1, -1, k_dim_out)
        cloud = torch.gather(dense_lifted_cloud, 1, uniform_idx_exp)

        tokens = self.topology_proj(cloud)
        return tokens, y_star


class EncoderZ4Router(nn.Module):
    """Z4: History-Aware Anchor Router with configurable routing stages L."""

    def __init__(self, input_dim, d_model, k_dim=16, K=8, r=1, L=1,
                 d_u=64, d_a=32, d_m=64, coverage_gamma=1.0,
                 init_temperature=1.0, max_proxy_points=16):
        super().__init__()
        self.K = K
        self.max_proxy_points = max_proxy_points
        self.d_model = d_model
        self.L = L

        self.z4_encoder = Z4TopologicalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            d_u=d_u,
            d_a=d_a,
            d_m=d_m,
            k=k_dim,
            K=K,
            r=r,
            L=L,
            coverage_gamma=coverage_gamma,
            init_temperature=init_temperature,
            feedback_dim=2,
            max_proxy_points=max_proxy_points,
        )
        self.d_model = d_model

    def set_normalization(self, mu, sigma):
        self.z4_encoder.set_normalization(mu, sigma)

    def forward(self, x, feedback=None):
        tokens, y_star, all_y, all_memory = self.z4_encoder(x, feedback=feedback)
        return tokens, y_star


# ============================================================
# MODEL WRAPPER
# ============================================================

class ModelWithEncoder(nn.Module):
    """Encoder + DeepHodgeTransformer + classifier."""

    def __init__(self, encoder, num_classes=8, k_dim=16, max_points=16, uses_feedback=False):
        super().__init__()
        self.encoder = encoder
        self.uses_feedback = uses_feedback
        self.transformer = DeepHodgeTransformer(
            num_layers=2,
            d_model=encoder.d_model,
            k_dim=k_dim,
            num_scales=3,
            max_points=max_points,
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoder.d_model, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, feedback=None):
        if self.uses_feedback and feedback is not None:
            tokens, y_star = self.encoder(x, feedback=feedback)
        else:
            tokens, y_star = self.encoder(x)
        transformer_out = self.transformer(tokens)
        pooled = transformer_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, y_star


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(model, name, train_loader, val_loader, test_loader,
                epochs=20, patience=5, lambda_task=1.0, lambda_reg=0.01):
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
        total_cls_loss = 0
        total_reg_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # Compute feedback for Z4 router: task loss signal
            if model.uses_feedback:
                # First forward without feedback to get initial logits
                logits, y_star = model(batch_x, feedback=None)
                cls_loss = criterion(logits, batch_y)
                # Regularization: encourage sparse, spread-out selection
                reg_loss = lambda_reg * (
                    # Budget regularization: penalize over-selection
                    (y_star.sum(dim=1) - model.encoder.K).pow(2).mean()
                    # Entropy regularization: encourage peaked selection
                    + 0.1 * (y_star * (y_star + 1e-6).log()).sum(dim=1).mean()
                )
                # Feedback signal for router memory
                feedback = torch.stack([
                    -lambda_task * cls_loss.detach(),
                    -lambda_reg * reg_loss.detach(),
                ], dim=-1).expand(batch_x.size(0), -1)  # [B, 2]

                # Second forward with feedback
                optimizer.zero_grad()
                logits, y_star = model(batch_x, feedback=feedback)
                cls_loss = criterion(logits, batch_y)
                reg_loss = lambda_reg * (
                    (y_star.sum(dim=1) - model.encoder.K).pow(2).mean()
                    + 0.1 * (y_star * (y_star + 1e-6).log()).sum(dim=1).mean()
                )
                loss = cls_loss + reg_loss
            else:
                logits, y_star = model(batch_x)
                cls_loss = criterion(logits, batch_y)
                reg_loss = lambda_reg * (
                    (y_star.sum(dim=1) - model.encoder.K).pow(2).mean()
                    + 0.1 * (y_star * (y_star + 1e-6).log()).sum(dim=1).mean()
                )
                loss = cls_loss + reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
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
                logits, _ = model(batch_x)
                val_loss += criterion(logits, batch_y).item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total * 100
        avg_val_loss = val_loss / len(val_loader)

        epoch_pbar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}",
                               val_loss=f"{avg_val_loss:.4f}", val_acc=f"{val_acc:.1f}%")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:2d}/{epochs} | "
                  f"Train Loss: {total_loss / len(train_loader):.4f} "
                  f"(cls: {total_cls_loss / len(train_loader):.4f}, "
                  f"reg: {total_reg_loss / len(train_loader):.4f}) | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early stopping on val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
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
            logits, _ = model(batch_x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(batch_y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

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
# MAIN
# ============================================================

if __name__ == "__main__":
    seq_len = 64
    input_dim = 10
    d_model = 64
    k_dim = 16
    num_classes = 8
    num_seeds = 3
    epochs = 25
    patience = 7

    # Set to True to skip legacy Z3 baselines and only run Z4 variants
    skip_z3_baselines = True

    # Encoder variants to compare
    encoder_variants = [
        ("z3_saliency", EncoderZ3Saliency, {"uses_feedback": False}),
        ("z3_uniform",  EncoderZ3Uniform,  {"uses_feedback": False}),
        ("z4_router_L1", EncoderZ4Router,  {"uses_feedback": True, "L": 1}),
        ("z4_router_L2", EncoderZ4Router,  {"uses_feedback": True, "L": 2}),
    ]

    if skip_z3_baselines:
        encoder_variants = [
            v for v in encoder_variants if not v[0].startswith("z3_")
        ]

    K_values = [8, 32]

    # --- Step 1: Generate data once ---
    print("=" * 70)
    print("STEP 1: Generating and caching synthetic data")
    print("=" * 70)

    data_cache = SyntheticDataCache(
        num_samples=1600,
        seq_len=seq_len,
        input_dim=input_dim,
        num_classes=num_classes,
        seed=42,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = data_cache.get_splits()

    print(f"Data generated: {data_cache.size_mb:.1f} MB in memory")
    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)" if DEVICE.type == "cuda" else " (CPU)"))
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class distribution (train): {torch.bincount(y_train).tolist()}")

    # --- Step 2: Run ablation ---
    print(f"\n{'=' * 70}")
    print("STEP 2: Z4 Router Ablation (multi-seed)")
    print(f"{'=' * 70}")

    all_results = defaultdict(lambda: defaultdict(list))

    for variant_name, encoder_class, extra_kwargs in encoder_variants:
        uses_feedback = extra_kwargs.pop("uses_feedback", False)
        L = extra_kwargs.pop("L", 1)

        for K in K_values:
            print(f"\n{'=' * 70}")
            print(f"Variant: {variant_name}, K={K} (compression: {K}/{seq_len} = {K / seq_len * 100:.1f}%)")
            print(f"{'=' * 70}")

            for seed_idx in tqdm(range(num_seeds), desc=f"{variant_name} K={K}", leave=False):
                seed = 42 + seed_idx * 100
                torch.manual_seed(seed)
                np.random.seed(seed)

                _pin = DEVICE.type == "cuda"
                train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, pin_memory=_pin)
                val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False, pin_memory=_pin)
                test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False, pin_memory=_pin)

                encoder_kwargs = dict(
                    input_dim=input_dim,
                    d_model=d_model,
                    k_dim=k_dim,
                    K=K,
                    r=1,
                    max_proxy_points=K,
                )
                if uses_feedback:
                    encoder_kwargs["L"] = L
                encoder = encoder_class(**encoder_kwargs)

                model = ModelWithEncoder(
                    encoder, num_classes=num_classes,
                    k_dim=k_dim, max_points=K,
                    uses_feedback=uses_feedback,
                ).to(DEVICE)

                metrics = train_model(
                    model, f"{variant_name}_K{K}_seed{seed}",
                    train_loader, val_loader, test_loader,
                    epochs=epochs, patience=patience,
                )
                metrics['seed'] = seed
                all_results[variant_name][K].append(metrics)

                del model
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    # --- Step 3: Aggregate and report ---
    print(f"\n{'=' * 70}")
    print("STEP 3: Aggregated Results")
    print(f"{'=' * 70}")

    aggregated = {}
    for variant_name in all_results:
        aggregated[variant_name] = {}
        for K in all_results[variant_name]:
            results = all_results[variant_name][K]
            test_accs = [r['test_acc'] for r in results]
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs, ddof=0)
            mean_val_loss = np.mean([r['best_val_loss'] for r in results])
            mean_time = np.mean([r['time_s'] for r in results])

            per_class_means = {}
            for c in range(num_classes):
                class_accs = [r['per_class_acc'].get(c, 0.0) for r in results]
                per_class_means[c] = np.mean(class_accs)

            aggregated[variant_name][K] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'mean_val_loss': mean_val_loss,
                'mean_time': mean_time,
                'per_class_acc_mean': per_class_means,
            }

    # Print summary table
    print(f"\n{'Variant':<18} {'K':<6} {'Test Acc (mean±std)':<24} {'Best Val Loss':<16} {'Time (s)':<10}")
    print("-" * 74)

    variant_order = [v[0] for v in encoder_variants]
    for variant_name in variant_order:
        for K in K_values:
            stats = aggregated[variant_name][K]
            print(f"{variant_name:<18} {K:<6} {stats['mean_acc']:.2f} ± {stats['std_acc']:.2f}%{'':<8} "
                  f"{stats['mean_val_loss']:.4f}{'':<8} {stats['mean_time']:.1f}")

    # Per-class accuracy breakdown
    print(f"\n{'=' * 70}")
    print("Per-Class Accuracy (mean across seeds)")
    print(f"{'=' * 70}")
    header = f"{'Variant':<18} {'K':<6}" + "".join(f"{name:<10}" for name in CLASS_NAMES)
    print(header)
    print("-" * len(header))

    for variant_name in variant_order:
        for K in K_values:
            per_class = aggregated[variant_name][K]['per_class_acc_mean']
            row = f"{variant_name:<18} {K:<6}"
            for c in range(num_classes):
                row += f"{per_class.get(c, 0.0):.1f}{'':<6}"
            print(row)

    # Z4 vs Z3 advantage analysis (only when Z3 baselines are present)
    if not skip_z3_baselines:
        print(f"\n{'=' * 70}")
        print("Z4 Router Advantage Analysis (vs Z3 baselines)")
        print(f"{'=' * 70}")

        for K in K_values:
            z3_sal = aggregated["z3_saliency"][K]['mean_acc']
            z3_uni = aggregated["z3_uniform"][K]['mean_acc']
            z4_l1 = aggregated["z4_router_L1"][K]['mean_acc']
            z4_l2 = aggregated["z4_router_L2"][K]['mean_acc']

            print(f"\nK={K}:")
            print(f"  Z4-L1 vs Z3-Saliency:  {z4_l1 - z3_sal:+.2f}%")
            print(f"  Z4-L1 vs Z3-Uniform:   {z4_l1 - z3_uni:+.2f}%")
            print(f"  Z4-L2 vs Z3-Saliency:  {z4_l2 - z3_sal:+.2f}%")
            print(f"  Z4-L2 vs Z3-Uniform:   {z4_l2 - z3_uni:+.2f}%")
            print(f"  Z4-L2 vs Z4-L1:        {z4_l2 - z4_l1:+.2f}% (memory benefit)")

            best_z3 = max(z3_sal, z3_uni)
            best_z4 = max(z4_l1, z4_l2)
            if best_z4 > best_z3:
                print(f"  → Z4 router IMPROVES over best Z3 baseline by {best_z4 - best_z3:+.2f}%")
            else:
                print(f"  → Z4 router does NOT improve over best Z3 baseline ({best_z3 - best_z4:.2f}% worse)")

        # Per-class delta analysis
        print(f"\n{'=' * 70}")
        print("Per-Class Delta: Z4-L2 vs Z3-Uniform (best Z3 baseline)")
        print(f"{'=' * 70}")
        for K in K_values:
            print(f"\nK={K}:")
            z3_uni_pc = aggregated["z3_uniform"][K]['per_class_acc_mean']
            z4_l2_pc = aggregated["z4_router_L2"][K]['per_class_acc_mean']
            for c, name in enumerate(CLASS_NAMES):
                delta = z4_l2_pc.get(c, 0.0) - z3_uni_pc.get(c, 0.0)
                marker = "↑" if delta > 0 else "↓" if delta < 0 else "="
                print(f"  {name:<10}: {z4_l2_pc.get(c, 0.0):.1f}% vs {z3_uni_pc.get(c, 0.0):.1f}%  ({delta:+.1f}% {marker})")
    else:
        # Z4-only comparison
        print(f"\n{'=' * 70}")
        print("Z4 Router Comparison (L1 vs L2)")
        print(f"{'=' * 70}")
        for K in K_values:
            z4_l1 = aggregated["z4_router_L1"][K]['mean_acc']
            z4_l2 = aggregated["z4_router_L2"][K]['mean_acc']
            print(f"\nK={K}:")
            print(f"  Z4-L1: {z4_l1:.2f}%")
            print(f"  Z4-L2: {z4_l2:.2f}%")
            print(f"  L2 vs L1: {z4_l2 - z4_l1:+.2f}% (memory benefit)")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'xdev')
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'z4_router_ablation_results.json')

    json_results = {'raw_results': {}, 'aggregated': {}}
    for variant_name in all_results:
        json_results['raw_results'][variant_name] = {}
        for K in all_results[variant_name]:
            json_results['raw_results'][variant_name][str(K)] = all_results[variant_name][K]

    for variant_name in aggregated:
        json_results['aggregated'][variant_name] = {}
        for K in aggregated[variant_name]:
            json_results['aggregated'][variant_name][str(K)] = aggregated[variant_name][K]

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
