2026-04-30 08:03:54,773 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-04-30 08:03:54,799 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 08:03:54,799 [INFO] __main__: Z3 Baseline Study: 1 datasets × 1 backbones × 3 seeds
2026-04-30 08:03:54,799 [INFO] __main__:   Datasets: ['photonic']
2026-04-30 08:03:54,799 [INFO] __main__:   Backbones: ['snn']
2026-04-30 08:03:54,799 [INFO] __main__:   Seeds: 3 (base=42)
2026-04-30 08:03:54,799 [INFO] __main__:   Device: cuda
2026-04-30 08:03:54,799 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 08:03:54,799 [INFO] __main__: 
▓▓▓ DATASET: PHOTONIC ▓▓▓

2026-04-30 08:03:54,799 [INFO] __main__: ═══ PHOTONIC × SNN (Topological) ═══
2026-04-30 08:03:54,801 [INFO] synapse.arch.data.data: Building dataloaders for dataset: photonic
2026-04-30 08:03:55,419 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T64_D10_C4_S42_R80_10
2026-04-30 08:03:55,419 [INFO] synapse.arch.data.data: Dataset 'photonic': train=88000, val=11000, test=11000, input_dim=10, seq_len=64
2026-04-30 08:03:55,426 [INFO] synapse.baselines.src.engine.train: Training backbone=snn, params=251076
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-30 08:03:55,468 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  251 K │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 251 K                                                         
Non-trainable params: 0                                                         
Total params: 251 K                                                             
Total estimated model params size (MB): 1                                       
Modules in train mode: 93                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0 | val_loss=0.9300 | val_acc=0.6012
Metric val/loss improved. New best score: 0.930
Epoch 1 | train_loss=0.9711 | val_loss=0.8514 | val_acc=0.6165
Metric val/loss improved by 0.079 >= min_delta = 0.0. New best score: 0.851
Epoch 2 | train_loss=0.8683 | val_loss=0.8387 | val_acc=0.6210
Metric val/loss improved by 0.013 >= min_delta = 0.0. New best score: 0.839
Epoch 3 | train_loss=0.8535 | val_loss=0.8317 | val_acc=0.6242
Metric val/loss improved by 0.007 >= min_delta = 0.0. New best score: 0.832
Epoch 4 | train_loss=0.8463 | val_loss=0.8267 | val_acc=0.6250
Metric val/loss improved by 0.005 >= min_delta = 0.0. New best score: 0.827
Epoch 5 | train_loss=0.8395 | val_loss=0.8214 | val_acc=0.6260
Metric val/loss improved by 0.005 >= min_delta = 0.0. New best score: 0.821
Epoch 6 | train_loss=0.8369 | val_loss=0.8159 | val_acc=0.6296
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.816
Epoch 7 | train_loss=0.8313 | val_loss=0.8131 | val_acc=0.6294
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.813
Epoch 8 | train_loss=0.8285 | val_loss=0.8105 | val_acc=0.6309
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.811
Epoch 9 | train_loss=0.8262 | val_loss=0.8081 | val_acc=0.6310
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.808
Epoch 10 | train_loss=0.8233 | val_loss=0.8059 | val_acc=0.6312
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.806
Epoch 11 | train_loss=0.8184 | val_loss=0.8045 | val_acc=0.6311
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.804
Epoch 12 | train_loss=0.8186 | val_loss=0.8039 | val_acc=0.6319
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.804
Epoch 13 | train_loss=0.8163 | val_loss=0.8020 | val_acc=0.6347
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.802
Epoch 14 | train_loss=0.8140 | val_loss=0.8011 | val_acc=0.6333
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.801
Epoch 15 | train_loss=0.8121 | val_loss=0.7991 | val_acc=0.6353
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.799
Epoch 16 | train_loss=0.8101 | val_loss=0.7980 | val_acc=0.6341
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.798
Epoch 17 | train_loss=0.8103 | val_loss=0.7966 | val_acc=0.6354
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.797
Epoch 18 | train_loss=0.8067 | val_loss=0.7959 | val_acc=0.6346
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.796
Epoch 19 | train_loss=0.8045 | val_loss=0.7957 | val_acc=0.6363
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.796
Epoch 20 | train_loss=0.8053 | val_loss=0.7937 | val_acc=0.6361
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.794
Epoch 21 | train_loss=0.8030 | val_loss=0.7928 | val_acc=0.6366
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.793
Epoch 22 | train_loss=0.8020 | val_loss=0.7915 | val_acc=0.6375
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.791
Epoch 23 | train_loss=0.8005 | val_loss=0.7915 | val_acc=0.6374
Epoch 24 | train_loss=0.8000 | val_loss=0.7909 | val_acc=0.6381
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.791
Epoch 25 | train_loss=0.7982 | val_loss=0.7890 | val_acc=0.6386
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.789
Epoch 26 | train_loss=0.7970 | val_loss=0.7885 | val_acc=0.6382
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.789
Epoch 27 | train_loss=0.7955 | val_loss=0.7878 | val_acc=0.6393
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.788
Epoch 28 | train_loss=0.7961 | val_loss=0.7868 | val_acc=0.6385
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.787
Epoch 29 | train_loss=0.7939 | val_loss=0.7859 | val_acc=0.6387
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.786
Epoch 30 | train_loss=0.7939 | val_loss=0.7851 | val_acc=0.6406
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.785
Epoch 31 | train_loss=0.7931 | val_loss=0.7841 | val_acc=0.6396
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.784
Epoch 32 | train_loss=0.7911 | val_loss=0.7835 | val_acc=0.6397
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.783
Epoch 33 | train_loss=0.7909 | val_loss=0.7823 | val_acc=0.6400
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.782
Epoch 34 | train_loss=0.7898 | val_loss=0.7824 | val_acc=0.6416
Epoch 35 | train_loss=0.7888 | val_loss=0.7817 | val_acc=0.6417
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.782
Epoch 36 | train_loss=0.7878 | val_loss=0.7817 | val_acc=0.6414
Epoch 37 | train_loss=0.7867 | val_loss=0.7817 | val_acc=0.6406
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.782
Epoch 38 | train_loss=0.7859 | val_loss=0.7811 | val_acc=0.6414
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.781
Epoch 39 | train_loss=0.7856 | val_loss=0.7806 | val_acc=0.6416
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.781
Epoch 40 | train_loss=0.7847 | val_loss=0.7801 | val_acc=0.6413
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.780
Epoch 41 | train_loss=0.7844 | val_loss=0.7794 | val_acc=0.6423
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.779
Epoch 42 | train_loss=0.7833 | val_loss=0.7786 | val_acc=0.6417
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.779
Epoch 43 | train_loss=0.7831 | val_loss=0.7783 | val_acc=0.6425
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.778
Epoch 44 | train_loss=0.7830 | val_loss=0.7778 | val_acc=0.6417
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.778
Epoch 45 | train_loss=0.7819 | val_loss=0.7776 | val_acc=0.6416
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.778
Epoch 46 | train_loss=0.7819 | val_loss=0.7770 | val_acc=0.6428
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.777
Epoch 47 | train_loss=0.7813 | val_loss=0.7774 | val_acc=0.6420
Epoch 48 | train_loss=0.7816 | val_loss=0.7770 | val_acc=0.6423
Epoch 49 | train_loss=0.7812 | val_loss=0.7771 | val_acc=0.6422
`Trainer.fit` stopped: `max_epochs=50` reached.
2026-04-30 09:02:27,537 [INFO] synapse.baselines.src.engine.train: Loaded evaluation checkpoint from /content/topo_synapse/outputs/baselines/20260430_080354_z3_baseline_full/dataset_photonic/backbone_snn/seed_42/checkpoints/best-epoch=046-val/loss=0.7770.ckpt
2026-04-30 09:02:30,253 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-30 09:02:30,253 [INFO] __main__:   photonic × snn × seed=42: rollout_auc=0.1870, degradation_slope=-0.2164
2026-04-30 09:02:30,253 [INFO] __main__:   photonic × snn × seed=42: accuracy=0.6373, f1=0.5758
2026-04-30 09:02:30,254 [INFO] __main__:   photonic × snn: mean_acc=0.6373 ± 0.0000 (1 seeds)
2026-04-30 09:02:30,254 [INFO] __main__: Dataset 'photonic' complete in 3515.5s (1 backbones)
2026-04-30 09:02:30,255 [INFO] synapse.baselines.src.reporting.report: JSON report saved to outputs/baselines/20260430_080354_z3_baseline_full/cross_backbone/photonic_results.json
2026-04-30 09:02:30,255 [INFO] synapse.baselines.src.reporting.report: Markdown report saved to outputs/baselines/20260430_080354_z3_baseline_full/cross_backbone/photonic_summary.md
2026-04-30 09:02:30,437 [INFO] synapse.baselines.src.reporting.visualize: Saved accuracy plot to outputs/baselines/20260430_080354_z3_baseline_full/cross_backbone/photonic_accuracy.pdf
2026-04-30 09:02:30,562 [INFO] synapse.baselines.src.reporting.visualize: Saved learning curves to outputs/baselines/20260430_080354_z3_baseline_full/cross_backbone/photonic_learning.pdf
2026-04-30 09:02:30,665 [INFO] __main__: Rollout report saved for photonic
2026-04-30 09:02:30,665 [INFO] __main__: Cross-backbone reports saved to outputs/baselines/20260430_080354_z3_baseline_full/cross_backbone
2026-04-30 09:02:30,665 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 09:02:30,665 [INFO] __main__: EXPERIMENT COMPLETE: 1 datasets, 3515.5s total
2026-04-30 09:02:30,665 [INFO] __main__: Output directory: outputs/baselines/20260430_080354_z3_baseline_full
2026-04-30 09:02:30,665 [INFO] __main__: ═══════════════════════════════════════════════════════════