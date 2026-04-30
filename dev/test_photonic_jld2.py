#!/usr/bin/env python3
"""Smoke test for the photonic JLD2 adapter fix.

Verifies that:
1. h5py and huggingface_hub are importable
2. PhotonicAdapter can be instantiated
3. _load_from_huggingface downloads and reads JLD2 files
4. load_splits() returns a valid DatasetBundle

Run:
    python dev/test_photonic_jld2.py
    python dev/test_photonic_jld2.py --max-samples 100
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Test photonic JLD2 adapter")
    parser.add_argument("--max-samples", type=int, default=50, help="Cap samples for speed")
    args = parser.parse_args()

    # 1. Check imports
    print("[1/4] Checking imports...")
    try:
        import h5py
        print(f"  h5py {h5py.__version__} OK")
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  Fix: pip install h5py")
        return 1

    try:
        import huggingface_hub
        print(f"  huggingface_hub {huggingface_hub.__version__} OK")
    except ImportError as e:
        print(f"  FAIL: {e}")
        print("  Fix: pip install huggingface_hub")
        return 1

    # 2. Instantiate adapter
    print("[2/4] Instantiating PhotonicAdapter...")
    from synapse.dataset.adapters.photonic_adapter import PhotonicAdapter

    adapter = PhotonicAdapter(max_samples=args.max_samples, data_root="data/datasets")
    print(f"  spec: name={adapter.spec.name}, input_dim={adapter.spec.input_dim}, "
          f"num_classes={adapter.spec.num_classes}, hf_repo={adapter.spec.hf_repo}")

    # 3. Download + read JLD2
    print("[3/4] Downloading and reading JLD2 files from HuggingFace...")
    try:
        records = adapter._load_from_huggingface()
        print(f"  Got {len(records)} records")
        if records:
            r0 = records[0]
            print(f"  First record keys: {list(r0.keys())}")
            grid = r0.get("grid")
            if grid is not None:
                print(f"  Grid shape: {grid.shape}, dtype: {grid.dtype}")
            label = r0.get("label")
            print(f"  Label: {label}")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 4. Full load_splits pipeline
    print("[4/4] Running load_splits()...")
    try:
        bundle = adapter.load_splits()
        print(f"  train: {bundle.train_sequences.shape}, labels: {bundle.train_labels.shape}")
        print(f"  val:   {bundle.val_sequences.shape}, labels: {bundle.val_labels.shape}")
        print(f"  test:  {bundle.test_sequences.shape}, labels: {bundle.test_labels.shape}")
        import numpy as np
        unique_labels = np.unique(bundle.train_labels)
        print(f"  unique train labels: {unique_labels}")
        print("  SUCCESS")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
