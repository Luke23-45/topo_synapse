2026-04-30 09:05:23,719 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-04-30 09:05:23,745 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 09:05:23,745 [INFO] __main__: Z3 Baseline Study: 1 datasets × 1 backbones × 3 seeds
2026-04-30 09:05:23,745 [INFO] __main__:   Datasets: ['photonic']
2026-04-30 09:05:23,745 [INFO] __main__:   Backbones: ['deep_hodge']
2026-04-30 09:05:23,745 [INFO] __main__:   Seeds: 3 (base=42)
2026-04-30 09:05:23,745 [INFO] __main__:   Device: cuda
2026-04-30 09:05:23,745 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 09:05:23,746 [INFO] __main__: 
▓▓▓ DATASET: PHOTONIC ▓▓▓

2026-04-30 09:05:23,746 [INFO] __main__: ═══ PHOTONIC × Deep Hodge (Proposed) ═══
2026-04-30 09:05:23,747 [INFO] synapse.arch.data.data: Building dataloaders for dataset: photonic
2026-04-30 09:05:23,748 [INFO] synapse.dataset.adapters.photonic_adapter: No prepared cache found for PhotonicTopology. Starting extraction...
2026-04-30 09:05:23,780 [INFO] synapse.dataset.adapters.photonic_adapter: Loading 2D photonic topology from HuggingFace: cgeorgiaw/2d-photonic-topology
2026-04-30 09:05:24,001 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/revision/main "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 11 files: 100% 11/11 [00:00<00:00, 84346.15it/s]
Download complete: : 0.00B [00:00, ?B/s]              2026-04-30 09:05:24,010 [INFO] synapse.dataset.adapters.photonic_adapter: Downloaded photonic lattices to: /root/.cache/huggingface/hub/datasets--cgeorgiaw--2d-photonic-topology/snapshots/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad

Download complete: : 0.00B [00:00, ?B/s]

Reading photonic JLD2:   9% 1/11 [00:00<00:03,  3.17it/s]
Reading photonic JLD2:  18% 2/11 [00:00<00:02,  3.21it/s]
Reading photonic JLD2:  27% 3/11 [00:00<00:02,  3.20it/s]
Reading photonic JLD2:  36% 4/11 [00:01<00:02,  3.17it/s]
Reading photonic JLD2:  45% 5/11 [00:01<00:01,  3.21it/s]
Reading photonic JLD2:  55% 6/11 [00:01<00:01,  3.25it/s]
Reading photonic JLD2:  64% 7/11 [00:02<00:01,  3.20it/s]
Reading photonic JLD2:  73% 8/11 [00:02<00:00,  3.24it/s]
Reading photonic JLD2:  82% 9/11 [00:02<00:00,  3.27it/s]
Reading photonic JLD2:  91% 10/11 [00:03<00:00,  3.26it/s]
Reading photonic JLD2: 100% 11/11 [00:03<00:00,  3.13it/s]
2026-04-30 09:05:27,523 [INFO] synapse.dataset.adapters.photonic_adapter: Read 110000 photonic records from 11 JLD2 files
Extracting PhotonicTopology: 100% 110000/110000 [00:00<00:00, 835145.16it/s]
2026-04-30 09:05:28,066 [INFO] synapse.dataset.adapters.photonic_adapter: PhotonicAdapter: loaded 110000 grids, shape=(110000, 8, 8, 8), classes=4
2026-04-30 09:05:35,287 [INFO] synapse.dataset.adapters.persistence: Saved modular prepared bundle to: data/datasets/photonic/prepared/T64_D10_C4_S42_R80_10
2026-04-30 09:05:35,328 [INFO] synapse.arch.data.data: Dataset 'photonic': train=88000, val=11000, test=11000, input_dim=10, seq_len=64
2026-04-30 09:05:35,359 [INFO] synapse.arch.training.builders.builder: Using preprocessor normalization stats (skipping recomputation)
2026-04-30 09:05:35,398 [INFO] synapse.baselines.src.engine.train: Training backbone=deep_hodge, params=200573
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-30 09:05:35,439 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
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
Modules in train mode: 47                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0 | val_loss=1.2691 | val_acc=0.3776
Metric val/loss improved. New best score: 1.269
Epoch 1 | train_loss=1.0846 | val_loss=0.8752 | val_acc=0.5985
Metric val/loss improved by 0.394 >= min_delta = 0.0. New best score: 0.875
Epoch 2 | train_loss=0.8886 | val_loss=0.8620 | val_acc=0.5997
Metric val/loss improved by 0.013 >= min_delta = 0.0. New best score: 0.862
Epoch 3 | train_loss=0.8692 | val_loss=0.8515 | val_acc=0.6041
Metric val/loss improved by 0.010 >= min_delta = 0.0. New best score: 0.852
Epoch 4 | train_loss=0.8581 | val_loss=0.8370 | val_acc=0.6081
Metric val/loss improved by 0.014 >= min_delta = 0.0. New best score: 0.837
Epoch 5 | train_loss=0.8516 | val_loss=0.8419 | val_acc=0.6075
Epoch 6 | train_loss=0.8474 | val_loss=0.8366 | val_acc=0.6100
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.837
Epoch 7 | train_loss=0.8458 | val_loss=0.8359 | val_acc=0.6116
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.836
Epoch 8 | train_loss=0.8432 | val_loss=0.8458 | val_acc=0.6120
Epoch 9 | train_loss=0.8450 | val_loss=0.8221 | val_acc=0.6174
Metric val/loss improved by 0.014 >= min_delta = 0.0. New best score: 0.822
Epoch 10 | train_loss=0.8359 | val_loss=0.8210 | val_acc=0.6185
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.821
Epoch 11 | train_loss=0.8317 | val_loss=0.8201 | val_acc=0.6179
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.820
Epoch 12 | train_loss=0.8313 | val_loss=0.8154 | val_acc=0.6199
Metric val/loss improved by 0.005 >= min_delta = 0.0. New best score: 0.815
Epoch 13 | train_loss=0.8276 | val_loss=0.8160 | val_acc=0.6209
Epoch 14 | train_loss=0.8292 | val_loss=0.8122 | val_acc=0.6200
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.812
Epoch 15 | train_loss=0.8264 | val_loss=0.8072 | val_acc=0.6217
Metric val/loss improved by 0.005 >= min_delta = 0.0. New best score: 0.807
Epoch 16 | train_loss=0.8217 | val_loss=0.8101 | val_acc=0.6213
Epoch 17 | train_loss=0.8220 | val_loss=0.8075 | val_acc=0.6207
Epoch 18 | train_loss=0.8193 | val_loss=0.8075 | val_acc=0.6219
Epoch 19 | train_loss=0.8181 | val_loss=0.8059 | val_acc=0.6229
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.806
Epoch 20 | train_loss=0.8185 | val_loss=0.8026 | val_acc=0.6268
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.803
Epoch 21 | train_loss=0.8144 | val_loss=0.7993 | val_acc=0.6287
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.799
Epoch 22 | train_loss=0.8114 | val_loss=0.7983 | val_acc=0.6315
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.798
Epoch 23 | train_loss=0.8108 | val_loss=0.7995 | val_acc=0.6275
Epoch 24 | train_loss=0.8126 | val_loss=0.7976 | val_acc=0.6318
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.798
Epoch 25 | train_loss=0.8103 | val_loss=0.7966 | val_acc=0.6320
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.797
Epoch 26 | train_loss=0.8081 | val_loss=0.7951 | val_acc=0.6321
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.795
Epoch 27 | train_loss=0.8061 | val_loss=0.7967 | val_acc=0.6307
Epoch 28 | train_loss=0.8056 | val_loss=0.7956 | val_acc=0.6298
Epoch 29 | train_loss=0.8050 | val_loss=0.7950 | val_acc=0.6294
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.795
Epoch 30 | train_loss=0.8039 | val_loss=0.7924 | val_acc=0.6327
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.792
Epoch 31 | train_loss=0.8018 | val_loss=0.7936 | val_acc=0.6333
Epoch 32 | train_loss=0.8019 | val_loss=0.7922 | val_acc=0.6335
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.792
Epoch 33 | train_loss=0.8001 | val_loss=0.7907 | val_acc=0.6343
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.791
Epoch 34 | train_loss=0.7987 | val_loss=0.7904 | val_acc=0.6339
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.790
Epoch 35 | train_loss=0.7984 | val_loss=0.7902 | val_acc=0.6340
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.790
Epoch 36 | train_loss=0.7968 | val_loss=0.7896 | val_acc=0.6336
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.790
Epoch 37 | train_loss=0.7957 | val_loss=0.7889 | val_acc=0.6345
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.789
Epoch 38 | train_loss=0.7946 | val_loss=0.7880 | val_acc=0.6346
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.788
Epoch 39 | train_loss=0.7948 | val_loss=0.7879 | val_acc=0.6362
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.788
Epoch 40 | train_loss=0.7941 | val_loss=0.7876 | val_acc=0.6360
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.788
Epoch 41 | train_loss=0.7930 | val_loss=0.7871 | val_acc=0.6364
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.787
Epoch 42 | train_loss=0.7921 | val_loss=0.7858 | val_acc=0.6353
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.786
Epoch 43 | train_loss=0.7915 | val_loss=0.7867 | val_acc=0.6359
Epoch 44 | train_loss=0.7907 | val_loss=0.7860 | val_acc=0.6377
Epoch 45 | train_loss=0.7906 | val_loss=0.7852 | val_acc=0.6364
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.785
Epoch 46 | train_loss=0.7896 | val_loss=0.7850 | val_acc=0.6360
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.785
Epoch 47 | train_loss=0.7890 | val_loss=0.7849 | val_acc=0.6367
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.785
Epoch 48 | train_loss=0.7890 | val_loss=0.7846 | val_acc=0.6359
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.785
Epoch 49 | train_loss=0.7885 | val_loss=0.7848 | val_acc=0.6366
`Trainer.fit` stopped: `max_epochs=50` reached.
2026-04-30 10:25:06,578 [INFO] synapse.baselines.src.engine.train: Loaded evaluation checkpoint from /content/topo_synapse/outputs/baselines/20260430_090523_z3_baseline_full/dataset_photonic/backbone_deep_hodge/seed_42/checkpoints/best-epoch=048-val/loss=0.7846.ckpt
2026-04-30 10:25:10,857 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-30 10:25:10,857 [INFO] __main__:   photonic × deep_hodge × seed=42: rollout_auc=0.1850, degradation_slope=-0.2836
2026-04-30 10:25:10,857 [INFO] __main__:   photonic × deep_hodge × seed=42: accuracy=0.6331, f1=0.5559
2026-04-30 10:25:10,858 [INFO] __main__:   photonic × deep_hodge: mean_acc=0.6331 ± 0.0000 (1 seeds)
2026-04-30 10:25:10,858 [INFO] __main__: Dataset 'photonic' complete in 4787.1s (1 backbones)
2026-04-30 10:25:10,858 [INFO] synapse.baselines.src.reporting.report: JSON report saved to outputs/baselines/20260430_090523_z3_baseline_full/cross_backbone/photonic_results.json
2026-04-30 10:25:10,859 [INFO] synapse.baselines.src.reporting.report: Markdown report saved to outputs/baselines/20260430_090523_z3_baseline_full/cross_backbone/photonic_summary.md
2026-04-30 10:25:11,060 [INFO] synapse.baselines.src.reporting.visualize: Saved accuracy plot to outputs/baselines/20260430_090523_z3_baseline_full/cross_backbone/photonic_accuracy.pdf
2026-04-30 10:25:11,180 [INFO] synapse.baselines.src.reporting.visualize: Saved learning curves to outputs/baselines/20260430_090523_z3_baseline_full/cross_backbone/photonic_learning.pdf
2026-04-30 10:25:11,291 [INFO] __main__: Rollout report saved for photonic
2026-04-30 10:25:11,292 [INFO] __main__: Cross-backbone reports saved to outputs/baselines/20260430_090523_z3_baseline_full/cross_backbone
2026-04-30 10:25:11,292 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 10:25:11,292 [INFO] __main__: EXPERIMENT COMPLETE: 1 datasets, 4787.1s total
2026-04-30 10:25:11,292 [INFO] __main__: Output directory: outputs/baselines/20260430_090523_z3_baseline_full
2026-04-30 10:25:11,292 [INFO] __main__: ═══════════════════════════════════════════════════════════