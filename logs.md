2026-04-30 06:51:38,204 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-04-30 06:51:38,230 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 06:51:38,230 [INFO] __main__: Z3 Baseline Study: 1 datasets × 5 backbones × 3 seeds
2026-04-30 06:51:38,230 [INFO] __main__:   Datasets: ['photonic']
2026-04-30 06:51:38,230 [INFO] __main__:   Backbones: ['mlp', 'tcn', 'ptv3', 'snn', 'deep_hodge']
2026-04-30 06:51:38,230 [INFO] __main__:   Seeds: 3 (base=42)
2026-04-30 06:51:38,230 [INFO] __main__:   Device: cuda
2026-04-30 06:51:38,230 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 06:51:38,230 [INFO] __main__: 
▓▓▓ DATASET: PHOTONIC ▓▓▓

2026-04-30 06:51:38,230 [INFO] __main__: ═══ PHOTONIC × MLP (Sanity Check) ═══
2026-04-30 06:51:38,231 [INFO] synapse.arch.data.data: Building dataloaders for dataset: photonic
2026-04-30 06:51:38,848 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T64_D10_C4_S42_R80_10
2026-04-30 06:51:38,848 [INFO] synapse.arch.data.data: Dataset 'photonic': train=88000, val=11000, test=11000, input_dim=10, seq_len=64
2026-04-30 06:51:38,858 [INFO] synapse.baselines.src.engine.train: Training backbone=mlp, params=1120580
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-30 06:51:38,902 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  1.1 M │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 1.1 M                                                         
Non-trainable params: 0                                                         
Total params: 1.1 M                                                             
Total estimated model params size (MB): 4                                       
Modules in train mode: 15                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0 | val_loss=0.8786 | val_acc=0.6075
Metric val/loss improved. New best score: 0.879
Epoch 1 | train_loss=0.9552 | val_loss=0.8476 | val_acc=0.6108
Metric val/loss improved by 0.031 >= min_delta = 0.0. New best score: 0.848
Epoch 2 | train_loss=0.8870 | val_loss=0.8387 | val_acc=0.6182
Metric val/loss improved by 0.009 >= min_delta = 0.0. New best score: 0.839
Epoch 3 | train_loss=0.8737 | val_loss=0.8306 | val_acc=0.6160
Metric val/loss improved by 0.008 >= min_delta = 0.0. New best score: 0.831
Epoch 4 | train_loss=0.8645 | val_loss=0.8293 | val_acc=0.6089
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.829
Epoch 5 | train_loss=0.8574 | val_loss=0.8235 | val_acc=0.6118
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.823
Epoch 6 | train_loss=0.8520 | val_loss=0.8138 | val_acc=0.6185
Metric val/loss improved by 0.010 >= min_delta = 0.0. New best score: 0.814
Epoch 7 | train_loss=0.8432 | val_loss=0.8081 | val_acc=0.6255
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.808
Epoch 8 | train_loss=0.8392 | val_loss=0.8020 | val_acc=0.6312
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.802
Epoch 9 | train_loss=0.8321 | val_loss=0.7999 | val_acc=0.6337
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.800
Epoch 10 | train_loss=0.8273 | val_loss=0.8005 | val_acc=0.6337
Epoch 11 | train_loss=0.8232 | val_loss=0.7969 | val_acc=0.6342
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.797
Epoch 12 | train_loss=0.8180 | val_loss=0.7952 | val_acc=0.6364
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.795
Epoch 13 | train_loss=0.8172 | val_loss=0.7946 | val_acc=0.6350
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.795
Epoch 14 | train_loss=0.8147 | val_loss=0.7913 | val_acc=0.6340
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.791
Epoch 15 | train_loss=0.8111 | val_loss=0.7917 | val_acc=0.6350
Epoch 16 | train_loss=0.8111 | val_loss=0.7901 | val_acc=0.6360
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.790
Epoch 17 | train_loss=0.8101 | val_loss=0.7896 | val_acc=0.6396
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.790
Epoch 18 | train_loss=0.8090 | val_loss=0.7879 | val_acc=0.6390
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.788
Epoch 19 | train_loss=0.8053 | val_loss=0.7884 | val_acc=0.6394
Epoch 20 | train_loss=0.8049 | val_loss=0.7862 | val_acc=0.6401
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.786
Epoch 21 | train_loss=0.8021 | val_loss=0.7859 | val_acc=0.6406
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.786
Epoch 22 | train_loss=0.8036 | val_loss=0.7842 | val_acc=0.6412
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.784
Epoch 23 | train_loss=0.8019 | val_loss=0.7850 | val_acc=0.6421
Epoch 24 | train_loss=0.7996 | val_loss=0.7833 | val_acc=0.6434
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.783
Epoch 25 | train_loss=0.7996 | val_loss=0.7823 | val_acc=0.6437
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.782
Epoch 26 | train_loss=0.7967 | val_loss=0.7823 | val_acc=0.6424
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.782
Epoch 27 | train_loss=0.7960 | val_loss=0.7813 | val_acc=0.6424
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.781
Epoch 28 | train_loss=0.7946 | val_loss=0.7800 | val_acc=0.6435
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.780
Epoch 29 | train_loss=0.7934 | val_loss=0.7801 | val_acc=0.6437
Epoch 30 | train_loss=0.7929 | val_loss=0.7807 | val_acc=0.6429
Epoch 31 | train_loss=0.7920 | val_loss=0.7783 | val_acc=0.6438
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.778
Epoch 32 | train_loss=0.7899 | val_loss=0.7786 | val_acc=0.6434
Epoch 33 | train_loss=0.7898 | val_loss=0.7770 | val_acc=0.6440
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.777
Epoch 34 | train_loss=0.7903 | val_loss=0.7767 | val_acc=0.6444
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.777
Epoch 35 | train_loss=0.7878 | val_loss=0.7768 | val_acc=0.6436
Epoch 36 | train_loss=0.7877 | val_loss=0.7763 | val_acc=0.6444
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.776
Epoch 37 | train_loss=0.7869 | val_loss=0.7750 | val_acc=0.6441
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.775
Epoch 38 | train_loss=0.7855 | val_loss=0.7740 | val_acc=0.6446
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.774
Epoch 39 | train_loss=0.7843 | val_loss=0.7736 | val_acc=0.6446
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.774
Epoch 40 | train_loss=0.7841 | val_loss=0.7736 | val_acc=0.6450
Epoch 41 | train_loss=0.7825 | val_loss=0.7732 | val_acc=0.6452
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.773
Epoch 42 | train_loss=0.7825 | val_loss=0.7730 | val_acc=0.6461
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.773
Epoch 43 | train_loss=0.7819 | val_loss=0.7725 | val_acc=0.6463
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.772
Epoch 44 | train_loss=0.7810 | val_loss=0.7725 | val_acc=0.6458
Epoch 45 | train_loss=0.7812 | val_loss=0.7724 | val_acc=0.6454
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.772
Epoch 46 | train_loss=0.7811 | val_loss=0.7723 | val_acc=0.6460
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.772
Epoch 47 | train_loss=0.7801 | val_loss=0.7722 | val_acc=0.6458
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.772
Epoch 48 | train_loss=0.7805 | val_loss=0.7720 | val_acc=0.6455
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.772
Epoch 49 | train_loss=0.7795 | val_loss=0.7720 | val_acc=0.6455
`Trainer.fit` stopped: `max_epochs=50` reached.
2026-04-30 07:10:23,057 [INFO] synapse.baselines.src.engine.train: Loaded evaluation checkpoint from /content/topo_synapse/outputs/baselines/20260430_065138_z3_baseline_full/dataset_photonic/backbone_mlp/seed_42/checkpoints/best-epoch=048-val/loss=0.7720.ckpt
2026-04-30 07:10:23,449 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-30 07:10:23,450 [INFO] __main__:   photonic × mlp × seed=42: rollout_auc=0.4240, degradation_slope=-0.1364
2026-04-30 07:10:23,450 [INFO] __main__:   photonic × mlp × seed=42: accuracy=0.6382, f1=0.5838
2026-04-30 07:10:23,451 [INFO] __main__:   photonic × mlp: mean_acc=0.6382 ± 0.0000 (1 seeds)
2026-04-30 07:10:23,451 [INFO] __main__: ═══ PHOTONIC × TCN (Temporal) ═══
2026-04-30 07:10:23,452 [INFO] synapse.arch.data.data: Building dataloaders for dataset: photonic
2026-04-30 07:10:24,073 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T64_D10_C4_S42_R80_10
2026-04-30 07:10:24,073 [INFO] synapse.arch.data.data: Dataset 'photonic': train=88000, val=11000, test=11000, input_dim=10, seq_len=64
/usr/local/lib/python3.12/dist-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
2026-04-30 07:10:24,079 [INFO] synapse.baselines.src.engine.train: Training backbone=tcn, params=105028
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-30 07:10:24,118 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  105 K │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 105 K                                                         
Non-trainable params: 0                                                         
Total params: 105 K                                                             
Total estimated model params size (MB): 0                                       
Modules in train mode: 57                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0 | val_loss=0.8608 | val_acc=0.6133
Metric val/loss improved. New best score: 0.861
