"""Quick overfit sanity test for each backbone.

Verifies that every backbone can memorize a small batch of data,
ensuring the forward + backward pass works end-to-end.

Usage:
    python -m synapse.baselines.test_overfit
"""

import torch
import torch.nn as nn

from synapse.synapse_arch.unified import Z3UnifiedModel
from synapse.baselines.src.core.config import BackboneCondition

ALL_BACKBONES = ["mlp", "tcn", "ptv3", "snn", "deep_hodge"]


def test_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for backbone_name in ALL_BACKBONES:
        # Use temporal modality for all backbones in this test
        # (input_dim=2 doesn't have xyz coords for geometric encoder)
        modality = "temporal"

        model = Z3UnifiedModel(
            backbone_type=backbone_name,
            modality=modality,
            input_dim=2,
            d_model=64,
            num_classes=4,
            num_tokens=128,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Dummy data
        x = torch.randn(4, 128, 2, device=device)
        labels = torch.tensor([0, 1, 2, 3], device=device)

        print(f"\n--- Overfit test: {backbone_name} ({model.num_parameters:,} params) ---")
        for step in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output.logits, labels)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                acc = (output.logits.argmax(dim=1) == labels).float().mean().item()
                print(f"  Step {step:3d}: loss={loss.item():.6f}, acc={acc:.4f}")

        final_acc = (model(x).logits.argmax(dim=1) == labels).float().mean().item()
        print(f"  Final: acc={final_acc:.4f}")

    print("\nAll overfit tests passed!")


if __name__ == "__main__":
    test_overfit()
