2026-04-30 07:46:33,456 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-04-30 07:46:33,481 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 07:46:33,482 [INFO] __main__: Z3 Baseline Study: 1 datasets × 1 backbones × 3 seeds
2026-04-30 07:46:33,482 [INFO] __main__:   Datasets: ['photonic']
2026-04-30 07:46:33,482 [INFO] __main__:   Backbones: ['deep_hodge']
2026-04-30 07:46:33,482 [INFO] __main__:   Seeds: 3 (base=42)
2026-04-30 07:46:33,482 [INFO] __main__:   Device: cuda
2026-04-30 07:46:33,482 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 07:46:33,482 [INFO] __main__: 
▓▓▓ DATASET: PHOTONIC ▓▓▓

2026-04-30 07:46:33,482 [INFO] __main__: ═══ PHOTONIC × Deep Hodge (Proposed) ═══
2026-04-30 07:46:33,484 [INFO] synapse.arch.data.data: Building dataloaders for dataset: photonic
2026-04-30 07:46:34,101 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T64_D10_C4_S42_R80_10
2026-04-30 07:46:34,102 [INFO] synapse.arch.data.data: Dataset 'photonic': train=88000, val=11000, test=11000, input_dim=10, seq_len=64
2026-04-30 07:46:34,131 [INFO] synapse.arch.training.builders.builder: Using preprocessor normalization stats (skipping recomputation)
2026-04-30 07:46:34,172 [INFO] synapse.baselines.src.engine.train: Training backbone=deep_hodge, params=200572
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-30 07:46:34,214 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  200 K │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 200 K                                                         
Non-trainable params: 0                                                         
Total params: 200 K                                                             
Total estimated model params size (MB): 0                                       
Modules in train mode: 46                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0 | val_loss=1.1601 | val_acc=0.4797
Metric val/loss improved. New best score: 1.160
Epoch 1 | train_loss=1.1587 | val_loss=0.8768 | val_acc=0.5965
Metric val/loss improved by 0.283 >= min_delta = 0.0. New best score: 0.877
Epoch 2 | train_loss=0.8926 | val_loss=0.8488 | val_acc=0.6023
Metric val/loss improved by 0.028 >= min_delta = 0.0. New best score: 0.849
Epoch 3 | train_loss=0.8697 | val_loss=0.8437 | val_acc=0.6053
Metric val/loss improved by 0.005 >= min_delta = 0.0. New best score: 0.844
Epoch 4 | train_loss=0.8576 | val_loss=0.8366 | val_acc=0.6059
Metric val/loss improved by 0.007 >= min_delta = 0.0. New best score: 0.837
Epoch 5 | train_loss=0.8486 | val_loss=0.8291 | val_acc=0.6064
Metric val/loss improved by 0.008 >= min_delta = 0.0. New best score: 0.829
Epoch 6 | train_loss=0.8452 | val_loss=0.8359 | val_acc=0.6057
Epoch 7 | train_loss=0.8460 | val_loss=0.8236 | val_acc=0.6075
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.824
