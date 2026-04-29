2026-04-29 16:13:06,344 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-04-29 16:13:06,369 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-29 16:13:06,370 [INFO] __main__: Z3 Baseline Study: 1 datasets × 5 backbones × 3 seeds
2026-04-29 16:13:06,370 [INFO] __main__:   Datasets: ['telecom']
2026-04-29 16:13:06,370 [INFO] __main__:   Backbones: ['mlp', 'tcn', 'ptv3', 'snn', 'deep_hodge']
2026-04-29 16:13:06,370 [INFO] __main__:   Seeds: 3 (base=42)
2026-04-29 16:13:06,370 [INFO] __main__:   Device: cuda
2026-04-29 16:13:06,370 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-29 16:13:06,370 [INFO] __main__: 
▓▓▓ DATASET: TELECOM ▓▓▓

2026-04-29 16:13:06,370 [INFO] __main__: ═══ TELECOM × MLP (Sanity Check) ═══
2026-04-29 16:13:06,371 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:13:06,371 [INFO] synapse.dataset.adapters.telecom_adapter: No prepared cache found for TelecomTS. Starting extraction...
2026-04-29 16:13:06,382 [INFO] datasets: TensorFlow version 2.20.0 available.
2026-04-29 16:13:06,383 [INFO] datasets: JAX version 0.7.2 available.
2026-04-29 16:13:06,601 [INFO] synapse.dataset.adapters.telecom_adapter: Loading TelecomTS from HuggingFace: AliMaatouk/TelecomTS
2026-04-29 16:13:06,754 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-04-29 16:13:06,764 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/AliMaatouk/TelecomTS/01e44b1b75e9b229c71a801ccd55320c28669e7f/README.md "HTTP/1.1 200 OK"
2026-04-29 16:13:06,870 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/TelecomTS.py "HTTP/1.1 404 Not Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-29 16:13:06,871 [WARNING] huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-29 16:13:07,081 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/AliMaatouk/TelecomTS/AliMaatouk/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 16:13:07,291 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/revision/01e44b1b75e9b229c71a801ccd55320c28669e7f "HTTP/1.1 200 OK"
2026-04-29 16:13:07,380 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-29 16:13:07,518 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=AliMaatouk/TelecomTS "HTTP/1.1 200 OK"
2026-04-29 16:13:07,612 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f/data?recursive=true&expand=false "HTTP/1.1 404 Not Found"
2026-04-29 16:13:07,699 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f?recursive=false&expand=false "HTTP/1.1 200 OK"
2026-04-29 16:13:07,794 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f?recursive=true&expand=false "HTTP/1.1 200 OK"
Resolving data files: 100% 99/99 [00:00<00:00, 23154.86it/s]
2026-04-29 16:13:07,981 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/dataset_infos.json "HTTP/1.1 404 Not Found"
Extracting TelecomTS: 100% 32000/32000 [02:07<00:00, 250.80it/s]
2026-04-29 16:15:15,844 [INFO] synapse.dataset.adapters.telecom_adapter: TelecomAdapter: loaded 32000 samples, shape=(32000, 128, 16), classes=3
2026-04-29 16:15:30,887 [INFO] synapse.dataset.adapters.persistence: Saved modular prepared bundle to: data/datasets/telecom/prepared/T256_D16_S42_R80_10
2026-04-29 16:15:30,889 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
2026-04-29 16:15:30,914 [INFO] synapse.baselines.src.engine.train: Training backbone=mlp, params=4278723
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:15:30,953 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  4.3 M │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 4.3 M                                                         
Non-trainable params: 0                                                         
Total params: 4.3 M                                                             
Total estimated model params size (MB): 17                                      
Modules in train mode: 15                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0/49:  99% 395/400 [00:03<00:00, 90.77batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  32% 16/50 [00:00<00:00, 156.98batch/s]
  Validating:  66% 33/50 [00:00<00:00, 158.95batch/s]
  Validating:  98% 49/50 [00:00<00:00, 149.97batch/s]
Epoch 0  │ val_loss=0.3148  │ val_acc=0.9691
Metric val/loss improved. New best score: 0.315
Epoch 1/49:  98% 390/400 [00:03<00:00, 108.31batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 208.00batch/s]
  Validating:  84% 42/50 [00:00<00:00, 197.81batch/s]
Epoch 1  │ train_loss=0.1616  │ val_loss=0.1328  │ val_acc=0.9750
Metric val/loss improved by 0.182 >= min_delta = 0.0. New best score: 0.133
Epoch 2/49:  99% 395/400 [00:03<00:00, 106.67batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 213.10batch/s]
  Validating:  88% 44/50 [00:00<00:00, 215.97batch/s]
Epoch 2  │ train_loss=0.0758  │ val_loss=0.0755  │ val_acc=0.9759
Metric val/loss improved by 0.057 >= min_delta = 0.0. New best score: 0.076
Epoch 3/49:  99% 396/400 [00:04<00:00, 90.30batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 204.36batch/s]
  Validating:  84% 42/50 [00:00<00:00, 199.01batch/s]
Epoch 3  │ train_loss=0.0589  │ val_loss=0.0570  │ val_acc=0.9762
Metric val/loss improved by 0.019 >= min_delta = 0.0. New best score: 0.057
Epoch 4/49:  99% 396/400 [00:03<00:00, 108.00batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 199.88batch/s]
  Validating:  82% 41/50 [00:00<00:00, 205.22batch/s]
Epoch 4  │ train_loss=0.0472  │ val_loss=0.0463  │ val_acc=0.9800
Metric val/loss improved by 0.011 >= min_delta = 0.0. New best score: 0.046
Epoch 5/49:  98% 392/400 [00:03<00:00, 110.49batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 185.95batch/s]
  Validating:  80% 40/50 [00:00<00:00, 197.72batch/s]
Epoch 5  │ train_loss=0.0362  │ val_loss=0.0399  │ val_acc=0.9831
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.040
Epoch 6/49:  99% 395/400 [00:04<00:00, 108.07batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 206.30batch/s]
  Validating:  88% 44/50 [00:00<00:00, 215.06batch/s]
Epoch 6  │ train_loss=0.0322  │ val_loss=0.0365  │ val_acc=0.9859
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.037
Epoch 7/49:  99% 397/400 [00:03<00:00, 109.31batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 204.25batch/s]
  Validating:  86% 43/50 [00:00<00:00, 209.13batch/s]
Epoch 7  │ train_loss=0.0298  │ val_loss=0.0336  │ val_acc=0.9878
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.034
Epoch 8/49: 100% 400/400 [00:03<00:00, 91.46batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 136.32batch/s]
  Validating:  56% 28/50 [00:00<00:00, 128.89batch/s]
  Validating:  86% 43/50 [00:00<00:00, 137.06batch/s]
Epoch 8  │ train_loss=0.0232  │ val_loss=0.0315  │ val_acc=0.9881
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.031
Epoch 9/49:  98% 393/400 [00:04<00:00, 106.34batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 212.03batch/s]
  Validating:  88% 44/50 [00:00<00:00, 202.69batch/s]
Epoch 9  │ train_loss=0.0238  │ val_loss=0.0308  │ val_acc=0.9887
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.031
Epoch 10/49: 100% 399/400 [00:03<00:00, 108.90batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 195.09batch/s]
  Validating:  80% 40/50 [00:00<00:00, 196.16batch/s]
Epoch 10  │ train_loss=0.0204  │ val_loss=0.0313  │ val_acc=0.9894
Epoch 11/49:  99% 397/400 [00:03<00:00, 88.17batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 142.02batch/s]
  Validating:  64% 32/50 [00:00<00:00, 149.57batch/s]
  Validating:  94% 47/50 [00:00<00:00, 142.67batch/s]
Epoch 11  │ train_loss=0.0208  │ val_loss=0.0319  │ val_acc=0.9909
Epoch 12/49:  99% 396/400 [00:03<00:00, 110.60batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 208.37batch/s]
  Validating:  86% 43/50 [00:00<00:00, 202.94batch/s]
Epoch 12  │ train_loss=0.0219  │ val_loss=0.0319  │ val_acc=0.9912
Epoch 13/49:  98% 392/400 [00:03<00:00, 108.63batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  46% 23/50 [00:00<00:00, 223.41batch/s]
  Validating:  92% 46/50 [00:00<00:00, 217.82batch/s]
Epoch 13  │ train_loss=0.0207  │ val_loss=0.0343  │ val_acc=0.9916
Epoch 14/49:  98% 392/400 [00:03<00:00, 86.00batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  32% 16/50 [00:00<00:00, 150.78batch/s]
  Validating:  64% 32/50 [00:00<00:00, 140.19batch/s]
  Validating:  94% 47/50 [00:00<00:00, 137.45batch/s]
Epoch 14  │ train_loss=0.0178  │ val_loss=0.0360  │ val_acc=0.9916
Epoch 15/49:  99% 397/400 [00:03<00:00, 108.20batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 196.95batch/s]
  Validating:  80% 40/50 [00:00<00:00, 194.17batch/s]
Epoch 15  │ train_loss=0.0198  │ val_loss=0.0386  │ val_acc=0.9916
Epoch 16/49:  98% 394/400 [00:03<00:00, 107.54batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 211.75batch/s]
  Validating:  88% 44/50 [00:00<00:00, 213.54batch/s]
Epoch 16  │ train_loss=0.0143  │ val_loss=0.0396  │ val_acc=0.9912
Epoch 17/49:  99% 397/400 [00:04<00:00, 83.96batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 145.45batch/s]
  Validating:  60% 30/50 [00:00<00:00, 141.81batch/s]
  Validating:  90% 45/50 [00:00<00:00, 140.31batch/s]
Epoch 17  │ train_loss=0.0150  │ val_loss=0.0407  │ val_acc=0.9916
Epoch 18/49:  98% 391/400 [00:03<00:00, 109.07batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  34% 17/50 [00:00<00:00, 169.58batch/s]
  Validating:  74% 37/50 [00:00<00:00, 182.29batch/s]
Epoch 18  │ train_loss=0.0144  │ val_loss=0.0405  │ val_acc=0.9912
Epoch 19/49:  98% 392/400 [00:03<00:00, 107.86batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 194.00batch/s]
  Validating:  80% 40/50 [00:00<00:00, 179.98batch/s]
Epoch 19  │ train_loss=0.0161  │ val_loss=0.0420  │ val_acc=0.9912
Monitored metric val/loss did not improve in the last 10 records. Best score: 0.031. Signaling Trainer to stop.
2026-04-29 16:16:59,371 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_161306_z3_baseline_full/dataset_telecom/backbone_mlp/seed_42/checkpoints/last.ckpt
2026-04-29 16:16:59,879 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:16:59,880 [INFO] __main__:   telecom × mlp × seed=42: rollout_auc=9.9300, degradation_slope=0.0018
2026-04-29 16:16:59,880 [INFO] __main__:   telecom × mlp × seed=42: accuracy=0.9900, f1=0.9900
2026-04-29 16:16:59,882 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:16:59,882 [INFO] synapse.dataset.adapters.telecom_adapter: No prepared cache found for TelecomTS. Starting extraction...
2026-04-29 16:16:59,883 [INFO] synapse.dataset.adapters.telecom_adapter: Loading TelecomTS from HuggingFace: AliMaatouk/TelecomTS
2026-04-29 16:17:00,003 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-04-29 16:17:00,013 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/AliMaatouk/TelecomTS/01e44b1b75e9b229c71a801ccd55320c28669e7f/README.md "HTTP/1.1 200 OK"
2026-04-29 16:17:00,099 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 16:17:00,308 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/AliMaatouk/TelecomTS/AliMaatouk/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 16:17:00,395 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-29 16:17:00,535 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=AliMaatouk/TelecomTS "HTTP/1.1 200 OK"
2026-04-29 16:17:00,626 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f/data?recursive=true&expand=false "HTTP/1.1 404 Not Found"
Resolving data files: 100% 99/99 [00:00<00:00, 260990.63it/s]
2026-04-29 16:17:00,807 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/dataset_infos.json "HTTP/1.1 404 Not Found"
Extracting TelecomTS: 100% 32000/32000 [02:08<00:00, 248.42it/s]
2026-04-29 16:19:09,893 [INFO] synapse.dataset.adapters.telecom_adapter: TelecomAdapter: loaded 32000 samples, shape=(32000, 128, 16), classes=3
2026-04-29 16:19:25,125 [INFO] synapse.dataset.adapters.persistence: Saved modular prepared bundle to: data/datasets/telecom/prepared/T256_D16_S43_R80_10
2026-04-29 16:19:25,128 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
2026-04-29 16:19:25,153 [INFO] synapse.baselines.src.engine.train: Training backbone=mlp, params=4278723
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:19:25,190 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  4.3 M │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 4.3 M                                                         
Non-trainable params: 0                                                         
Total params: 4.3 M                                                             
Total estimated model params size (MB): 17                                      
Modules in train mode: 15                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0/49: 100% 399/400 [00:03<00:00, 111.53batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 205.93batch/s]
  Validating:  86% 43/50 [00:00<00:00, 212.99batch/s]
Epoch 0  │ val_loss=0.3356  │ val_acc=0.9684
Metric val/loss improved. New best score: 0.336
Epoch 1/49:  99% 397/400 [00:04<00:00, 84.92batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 146.43batch/s]
  Validating:  60% 30/50 [00:00<00:00, 135.41batch/s]
  Validating:  88% 44/50 [00:00<00:00, 135.68batch/s]
Epoch 1  │ train_loss=0.1661  │ val_loss=0.1394  │ val_acc=0.9781
Metric val/loss improved by 0.196 >= min_delta = 0.0. New best score: 0.139
Epoch 2/49:  98% 390/400 [00:03<00:00, 106.92batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 211.35batch/s]
  Validating:  88% 44/50 [00:00<00:00, 215.37batch/s]
Epoch 2  │ train_loss=0.0787  │ val_loss=0.0751  │ val_acc=0.9800
Metric val/loss improved by 0.064 >= min_delta = 0.0. New best score: 0.075
Epoch 3/49:  98% 394/400 [00:03<00:00, 102.55batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 202.17batch/s]
  Validating:  84% 42/50 [00:00<00:00, 199.68batch/s]
Epoch 3  │ train_loss=0.0589  │ val_loss=0.0542  │ val_acc=0.9834
Metric val/loss improved by 0.021 >= min_delta = 0.0. New best score: 0.054
Epoch 4/49:  99% 395/400 [00:04<00:00, 99.45batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 217.21batch/s]
  Validating:  90% 45/50 [00:00<00:00, 220.81batch/s]
Epoch 4  │ train_loss=0.0499  │ val_loss=0.0452  │ val_acc=0.9887
Metric val/loss improved by 0.009 >= min_delta = 0.0. New best score: 0.045
Epoch 5/49:  98% 391/400 [00:03<00:00, 104.64batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 214.35batch/s]
  Validating:  88% 44/50 [00:00<00:00, 212.21batch/s]
Epoch 5  │ train_loss=0.0398  │ val_loss=0.0388  │ val_acc=0.9875
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.039
Epoch 6/49:  98% 393/400 [00:03<00:00, 106.53batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 191.83batch/s]
  Validating:  82% 41/50 [00:00<00:00, 199.37batch/s]
Epoch 6  │ train_loss=0.0335  │ val_loss=0.0360  │ val_acc=0.9875
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.036
Epoch 7/49:  98% 391/400 [00:04<00:00, 107.23batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 198.21batch/s]
  Validating:  80% 40/50 [00:00<00:00, 195.40batch/s]
Epoch 7  │ train_loss=0.0285  │ val_loss=0.0373  │ val_acc=0.9887
Epoch 8/49:  98% 390/400 [00:03<00:00, 105.90batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 197.66batch/s]
  Validating:  80% 40/50 [00:00<00:00, 195.42batch/s]
Epoch 8  │ train_loss=0.0265  │ val_loss=0.0359  │ val_acc=0.9897
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.036
Epoch 9/49:  98% 392/400 [00:03<00:00, 107.74batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 127.69batch/s]
  Validating:  54% 27/50 [00:00<00:00, 134.73batch/s]
  Validating:  82% 41/50 [00:00<00:00, 134.30batch/s]
Epoch 9  │ train_loss=0.0240  │ val_loss=0.0362  │ val_acc=0.9903
Epoch 10/49: 100% 399/400 [00:04<00:00, 109.39batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 197.13batch/s]
  Validating:  82% 41/50 [00:00<00:00, 187.21batch/s]
Epoch 10  │ train_loss=0.0235  │ val_loss=0.0354  │ val_acc=0.9909
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.035
Epoch 11/49: 100% 399/400 [00:03<00:00, 106.48batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  46% 23/50 [00:00<00:00, 225.28batch/s]
  Validating:  92% 46/50 [00:00<00:00, 219.19batch/s]
Epoch 11  │ train_loss=0.0171  │ val_loss=0.0359  │ val_acc=0.9922
Epoch 12/49:  99% 395/400 [00:03<00:00, 89.35batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 146.02batch/s]
  Validating:  60% 30/50 [00:00<00:00, 147.72batch/s]
  Validating:  92% 46/50 [00:00<00:00, 149.33batch/s]
Epoch 12  │ train_loss=0.0191  │ val_loss=0.0355  │ val_acc=0.9937
Epoch 13/49:  98% 394/400 [00:04<00:00, 105.32batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 193.94batch/s]
  Validating:  80% 40/50 [00:00<00:00, 192.30batch/s]
Epoch 13  │ train_loss=0.0166  │ val_loss=0.0355  │ val_acc=0.9934
Epoch 14/49:  98% 391/400 [00:03<00:00, 108.49batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  46% 23/50 [00:00<00:00, 223.41batch/s]
  Validating:  92% 46/50 [00:00<00:00, 219.90batch/s]
Epoch 14  │ train_loss=0.0191  │ val_loss=0.0372  │ val_acc=0.9937
Epoch 15/49:  99% 397/400 [00:03<00:00, 87.50batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 123.38batch/s]
  Validating:  54% 27/50 [00:00<00:00, 131.38batch/s]
  Validating:  84% 42/50 [00:00<00:00, 138.90batch/s]
Epoch 15  │ train_loss=0.0181  │ val_loss=0.0402  │ val_acc=0.9934
Epoch 16/49:  98% 392/400 [00:04<00:00, 106.00batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 203.77batch/s]
  Validating:  84% 42/50 [00:00<00:00, 204.46batch/s]
Epoch 16  │ train_loss=0.0140  │ val_loss=0.0399  │ val_acc=0.9928
Epoch 17/49: 100% 398/400 [00:03<00:00, 105.26batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 200.64batch/s]
  Validating:  84% 42/50 [00:00<00:00, 205.93batch/s]
Epoch 17  │ train_loss=0.0124  │ val_loss=0.0404  │ val_acc=0.9922
Epoch 18/49:  98% 392/400 [00:03<00:00, 86.53batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 145.74batch/s]
  Validating:  60% 30/50 [00:00<00:00, 147.55batch/s]
  Validating:  90% 45/50 [00:00<00:00, 146.13batch/s]
Epoch 18  │ train_loss=0.0146  │ val_loss=0.0399  │ val_acc=0.9931
Epoch 19/49:  97% 389/400 [00:04<00:00, 108.59batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 197.57batch/s]
  Validating:  80% 40/50 [00:00<00:00, 198.47batch/s]
Epoch 19  │ train_loss=0.0128  │ val_loss=0.0430  │ val_acc=0.9931
Epoch 20/49:  99% 397/400 [00:03<00:00, 105.48batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 205.18batch/s]
  Validating:  86% 43/50 [00:00<00:00, 211.58batch/s]
Epoch 20  │ train_loss=0.0110  │ val_loss=0.0435  │ val_acc=0.9928
Monitored metric val/loss did not improve in the last 10 records. Best score: 0.035. Signaling Trainer to stop.
2026-04-29 16:20:56,728 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_161306_z3_baseline_full/dataset_telecom/backbone_mlp/seed_43/checkpoints/last.ckpt
2026-04-29 16:20:57,248 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:20:57,249 [INFO] __main__:   telecom × mlp × seed=43: rollout_auc=9.8000, degradation_slope=0.0000
2026-04-29 16:20:57,249 [INFO] __main__:   telecom × mlp × seed=43: accuracy=0.9875, f1=0.9874
2026-04-29 16:20:57,251 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:20:57,251 [INFO] synapse.dataset.adapters.telecom_adapter: No prepared cache found for TelecomTS. Starting extraction...
2026-04-29 16:20:57,251 [INFO] synapse.dataset.adapters.telecom_adapter: Loading TelecomTS from HuggingFace: AliMaatouk/TelecomTS
2026-04-29 16:20:57,362 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-04-29 16:20:57,374 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/AliMaatouk/TelecomTS/01e44b1b75e9b229c71a801ccd55320c28669e7f/README.md "HTTP/1.1 200 OK"
2026-04-29 16:20:57,461 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 16:20:57,705 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/AliMaatouk/TelecomTS/AliMaatouk/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 16:20:57,795 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-29 16:20:58,020 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=AliMaatouk/TelecomTS "HTTP/1.1 200 OK"
2026-04-29 16:20:58,116 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f/data?recursive=true&expand=false "HTTP/1.1 404 Not Found"
Resolving data files: 100% 99/99 [00:00<00:00, 343454.17it/s]
2026-04-29 16:20:58,273 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/dataset_infos.json "HTTP/1.1 404 Not Found"
Extracting TelecomTS: 100% 32000/32000 [02:10<00:00, 245.99it/s]
2026-04-29 16:23:08,599 [INFO] synapse.dataset.adapters.telecom_adapter: TelecomAdapter: loaded 32000 samples, shape=(32000, 128, 16), classes=3
2026-04-29 16:23:24,756 [INFO] synapse.dataset.adapters.persistence: Saved modular prepared bundle to: data/datasets/telecom/prepared/T256_D16_S44_R80_10
2026-04-29 16:23:24,761 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
2026-04-29 16:23:24,794 [INFO] synapse.baselines.src.engine.train: Training backbone=mlp, params=4278723
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:23:24,834 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  4.3 M │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 4.3 M                                                         
Non-trainable params: 0                                                         
Total params: 4.3 M                                                             
Total estimated model params size (MB): 17                                      
Modules in train mode: 15                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0/49:  99% 397/400 [00:03<00:00, 112.84batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 187.97batch/s]
  Validating:  82% 41/50 [00:00<00:00, 203.84batch/s]
Epoch 0  │ val_loss=0.2755  │ val_acc=0.9259
Metric val/loss improved. New best score: 0.275
Epoch 1/49:  98% 393/400 [00:04<00:00, 105.88batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 186.97batch/s]
  Validating:  78% 39/50 [00:00<00:00, 190.89batch/s]
Epoch 1  │ train_loss=0.1661  │ val_loss=0.1608  │ val_acc=0.9588
Metric val/loss improved by 0.115 >= min_delta = 0.0. New best score: 0.161
Epoch 2/49: 100% 398/400 [00:03<00:00, 107.89batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 184.23batch/s]
  Validating:  82% 41/50 [00:00<00:00, 199.88batch/s]
Epoch 2  │ train_loss=0.0723  │ val_loss=0.0901  │ val_acc=0.9812
Metric val/loss improved by 0.071 >= min_delta = 0.0. New best score: 0.090
Epoch 3/49:  98% 393/400 [00:03<00:00, 106.63batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 196.28batch/s]
  Validating:  80% 40/50 [00:00<00:00, 192.35batch/s]
Epoch 3  │ train_loss=0.0566  │ val_loss=0.0602  │ val_acc=0.9847
Metric val/loss improved by 0.030 >= min_delta = 0.0. New best score: 0.060
Epoch 4/49:  98% 393/400 [00:04<00:00, 108.17batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 193.87batch/s]
  Validating:  82% 41/50 [00:00<00:00, 200.44batch/s]
Epoch 4  │ train_loss=0.0479  │ val_loss=0.0493  │ val_acc=0.9866
Metric val/loss improved by 0.011 >= min_delta = 0.0. New best score: 0.049
Epoch 5/49: 100% 400/400 [00:03<00:00, 101.19batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 182.50batch/s]
  Validating:  80% 40/50 [00:00<00:00, 193.72batch/s]
Epoch 5  │ train_loss=0.0360  │ val_loss=0.0430  │ val_acc=0.9869
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.043
Epoch 6/49: 100% 400/400 [00:04<00:00, 82.84batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 116.28batch/s]
  Validating:  52% 26/50 [00:00<00:00, 126.21batch/s]
  Validating:  82% 41/50 [00:00<00:00, 133.90batch/s]
Epoch 6  │ train_loss=0.0328  │ val_loss=0.0409  │ val_acc=0.9878
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.041
Epoch 7/49:  98% 391/400 [00:03<00:00, 104.06batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 206.98batch/s]
  Validating:  84% 42/50 [00:00<00:00, 202.66batch/s]
Epoch 7  │ train_loss=0.0301  │ val_loss=0.0397  │ val_acc=0.9881
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.040
Epoch 8/49:  98% 391/400 [00:03<00:00, 105.77batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 199.15batch/s]
  Validating:  80% 40/50 [00:00<00:00, 195.44batch/s]
Epoch 8  │ train_loss=0.0270  │ val_loss=0.0402  │ val_acc=0.9887
Epoch 9/49: 100% 398/400 [00:04<00:00, 85.02batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 205.30batch/s]
  Validating:  84% 42/50 [00:00<00:00, 192.07batch/s]
Epoch 9  │ train_loss=0.0256  │ val_loss=0.0417  │ val_acc=0.9887
Epoch 10/49:  99% 397/400 [00:03<00:00, 105.41batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 194.66batch/s]
  Validating:  82% 41/50 [00:00<00:00, 202.27batch/s]
Epoch 10  │ train_loss=0.0236  │ val_loss=0.0437  │ val_acc=0.9884
Epoch 11/49:  99% 397/400 [00:03<00:00, 109.69batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 216.25batch/s]
  Validating:  88% 44/50 [00:00<00:00, 205.54batch/s]
Epoch 11  │ train_loss=0.0215  │ val_loss=0.0461  │ val_acc=0.9891
Epoch 12/49:  99% 395/400 [00:04<00:00, 84.66batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 199.98batch/s]
  Validating:  82% 41/50 [00:00<00:00, 198.67batch/s]
Epoch 12  │ train_loss=0.0188  │ val_loss=0.0469  │ val_acc=0.9894
Epoch 13/49: 100% 399/400 [00:03<00:00, 103.52batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 205.59batch/s]
  Validating:  84% 42/50 [00:00<00:00, 202.46batch/s]
Epoch 13  │ train_loss=0.0192  │ val_loss=0.0488  │ val_acc=0.9897
Epoch 14/49: 100% 399/400 [00:03<00:00, 108.99batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  34% 17/50 [00:00<00:00, 168.00batch/s]
  Validating:  76% 38/50 [00:00<00:00, 187.89batch/s]
Epoch 14  │ train_loss=0.0185  │ val_loss=0.0512  │ val_acc=0.9900
Epoch 15/49:  98% 392/400 [00:04<00:00, 90.21batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 215.62batch/s]
  Validating:  88% 44/50 [00:00<00:00, 214.43batch/s]
Epoch 15  │ train_loss=0.0196  │ val_loss=0.0519  │ val_acc=0.9906
Epoch 16/49: 100% 399/400 [00:03<00:00, 104.54batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 181.61batch/s]
  Validating:  78% 39/50 [00:00<00:00, 190.89batch/s]
Epoch 16  │ train_loss=0.0178  │ val_loss=0.0578  │ val_acc=0.9912
Epoch 17/49:  99% 397/400 [00:03<00:00, 101.04batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 190.83batch/s]
  Validating:  80% 40/50 [00:00<00:00, 185.45batch/s]
Epoch 17  │ train_loss=0.0133  │ val_loss=0.0616  │ val_acc=0.9912
Monitored metric val/loss did not improve in the last 10 records. Best score: 0.040. Signaling Trainer to stop.
2026-04-29 16:24:44,638 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_161306_z3_baseline_full/dataset_telecom/backbone_mlp/seed_44/checkpoints/last.ckpt
2026-04-29 16:24:45,185 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:24:45,186 [INFO] __main__:   telecom × mlp × seed=44: rollout_auc=9.6600, degradation_slope=0.0016
2026-04-29 16:24:45,186 [INFO] __main__:   telecom × mlp × seed=44: accuracy=0.9919, f1=0.9919
2026-04-29 16:24:45,187 [INFO] __main__:   telecom × mlp: mean_acc=0.9898 ± 0.0018 (3 seeds)
2026-04-29 16:24:45,187 [INFO] __main__: ═══ TELECOM × TCN (Temporal) ═══
2026-04-29 16:24:45,188 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:24:47,676 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T256_D16_S42_R80_10
2026-04-29 16:24:47,676 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
/usr/local/lib/python3.12/dist-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
2026-04-29 16:24:47,696 [INFO] synapse.baselines.src.engine.train: Training backbone=tcn, params=117635
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:24:47,759 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  117 K │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 117 K                                                         
Non-trainable params: 0                                                         
Total params: 117 K                                                             
Total estimated model params size (MB): 0                                       
Modules in train mode: 57                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0/49: 100% 399/400 [00:06<00:00, 62.81batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 133.70batch/s]
  Validating:  56% 28/50 [00:00<00:00, 136.81batch/s]
  Validating:  84% 42/50 [00:00<00:00, 130.94batch/s]
Epoch 0  │ val_loss=0.6814  │ val_acc=0.8666
Metric val/loss improved. New best score: 0.681
Epoch 1/49:  99% 395/400 [00:07<00:00, 61.94batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 118.47batch/s]
  Validating:  54% 27/50 [00:00<00:00, 132.37batch/s]
  Validating:  82% 41/50 [00:00<00:00, 133.27batch/s]
Epoch 1  │ train_loss=0.2851  │ val_loss=0.2229  │ val_acc=0.9312
Metric val/loss improved by 0.458 >= min_delta = 0.0. New best score: 0.223
Epoch 2/49: 100% 398/400 [00:06<00:00, 62.06batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 132.52batch/s]
  Validating:  56% 28/50 [00:00<00:00, 133.29batch/s]
  Validating:  84% 42/50 [00:00<00:00, 134.00batch/s]
Epoch 2  │ train_loss=0.0652  │ val_loss=0.0935  │ val_acc=0.9659
Metric val/loss improved by 0.129 >= min_delta = 0.0. New best score: 0.094
Epoch 3/49:  98% 394/400 [00:07<00:00, 61.45batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.88batch/s]
  Validating:  54% 27/50 [00:00<00:00, 134.70batch/s]
  Validating:  84% 42/50 [00:00<00:00, 138.13batch/s]
Epoch 3  │ train_loss=0.0421  │ val_loss=0.0406  │ val_acc=0.9869
Metric val/loss improved by 0.053 >= min_delta = 0.0. New best score: 0.041
Epoch 4/49: 100% 398/400 [00:06<00:00, 48.69batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 113.01batch/s]
  Validating:  48% 24/50 [00:00<00:00, 104.21batch/s]
  Validating:  70% 35/50 [00:00<00:00, 105.89batch/s]
  Validating:  94% 47/50 [00:00<00:00, 109.09batch/s]
Epoch 4  │ train_loss=0.0357  │ val_loss=0.0256  │ val_acc=0.9919
Metric val/loss improved by 0.015 >= min_delta = 0.0. New best score: 0.026
Epoch 5/49: 100% 399/400 [00:06<00:00, 63.40batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 139.17batch/s]
  Validating:  58% 29/50 [00:00<00:00, 140.59batch/s]
  Validating:  88% 44/50 [00:00<00:00, 130.71batch/s]
Epoch 5  │ train_loss=0.0320  │ val_loss=0.0185  │ val_acc=0.9950
Metric val/loss improved by 0.007 >= min_delta = 0.0. New best score: 0.018
Epoch 6/49: 100% 398/400 [00:06<00:00, 60.82batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 142.80batch/s]
  Validating:  60% 30/50 [00:00<00:00, 134.47batch/s]
  Validating:  88% 44/50 [00:00<00:00, 133.78batch/s]
Epoch 6  │ train_loss=0.0236  │ val_loss=0.0156  │ val_acc=0.9953
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.016
Epoch 7/49:  99% 396/400 [00:06<00:00, 62.04batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 127.49batch/s]
  Validating:  54% 27/50 [00:00<00:00, 132.61batch/s]
  Validating:  82% 41/50 [00:00<00:00, 132.90batch/s]
Epoch 7  │ train_loss=0.0255  │ val_loss=0.0130  │ val_acc=0.9959
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.013
Epoch 8/49: 100% 400/400 [00:07<00:00, 62.27batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 139.43batch/s]
  Validating:  58% 29/50 [00:00<00:00, 141.92batch/s]
  Validating:  88% 44/50 [00:00<00:00, 143.65batch/s]
Epoch 8  │ train_loss=0.0198  │ val_loss=0.0112  │ val_acc=0.9959
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.011
Epoch 9/49: 100% 398/400 [00:06<00:00, 60.73batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 134.55batch/s]
  Validating:  56% 28/50 [00:00<00:00, 136.48batch/s]
  Validating:  84% 42/50 [00:00<00:00, 134.37batch/s]
Epoch 9  │ train_loss=0.0206  │ val_loss=0.0096  │ val_acc=0.9966
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.010
Epoch 10/49:  99% 395/400 [00:07<00:00, 58.83batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 102.31batch/s]
  Validating:  48% 24/50 [00:00<00:00, 113.06batch/s]
  Validating:  74% 37/50 [00:00<00:00, 117.64batch/s]
  Validating: 100% 50/50 [00:00<00:00, 121.60batch/s]
Epoch 10  │ train_loss=0.0190  │ val_loss=0.0085  │ val_acc=0.9969
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.009
Epoch 11/49:  99% 397/400 [00:07<00:00, 42.80batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 124.98batch/s]
  Validating:  54% 27/50 [00:00<00:00, 129.59batch/s]
  Validating:  80% 40/50 [00:00<00:00, 126.03batch/s]
Epoch 11  │ train_loss=0.0172  │ val_loss=0.0078  │ val_acc=0.9972
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.008
Epoch 12/49:  98% 394/400 [00:06<00:00, 57.94batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 134.38batch/s]
  Validating:  56% 28/50 [00:00<00:00, 129.31batch/s]
  Validating:  86% 43/50 [00:00<00:00, 134.45batch/s]
Epoch 12  │ train_loss=0.0167  │ val_loss=0.0073  │ val_acc=0.9975
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.007
Epoch 13/49: 100% 399/400 [00:08<00:00, 55.72batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 115.49batch/s]
  Validating:  50% 25/50 [00:00<00:00, 119.74batch/s]
  Validating:  74% 37/50 [00:00<00:00, 119.62batch/s]
  Validating: 100% 50/50 [00:00<00:00, 120.90batch/s]
Epoch 13  │ train_loss=0.0159  │ val_loss=0.0070  │ val_acc=0.9975
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.007
Epoch 14/49: 100% 398/400 [00:07<00:00, 42.72batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 98.49batch/s]
  Validating:  40% 20/50 [00:00<00:00, 91.56batch/s]
  Validating:  62% 31/50 [00:00<00:00, 95.66batch/s]
  Validating:  82% 41/50 [00:00<00:00, 94.61batch/s]
Epoch 14  │ train_loss=0.0159  │ val_loss=0.0064  │ val_acc=0.9975
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.006
Epoch 15/49:  98% 394/400 [00:07<00:00, 58.20batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 132.05batch/s]
  Validating:  56% 28/50 [00:00<00:00, 134.85batch/s]
  Validating:  84% 42/50 [00:00<00:00, 122.56batch/s]
Epoch 15  │ train_loss=0.0133  │ val_loss=0.0056  │ val_acc=0.9975
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.006
Epoch 16/49: 100% 399/400 [00:07<00:00, 52.28batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 121.14batch/s]
  Validating:  52% 26/50 [00:00<00:00, 121.62batch/s]
  Validating:  78% 39/50 [00:00<00:00, 120.59batch/s]
Epoch 16  │ train_loss=0.0110  │ val_loss=0.0050  │ val_acc=0.9978
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.005
Epoch 17/49:  99% 396/400 [00:07<00:00, 58.48batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 112.78batch/s]
  Validating:  52% 26/50 [00:00<00:00, 125.11batch/s]
  Validating:  78% 39/50 [00:00<00:00, 126.88batch/s]
Epoch 17  │ train_loss=0.0109  │ val_loss=0.0045  │ val_acc=0.9978
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.005
Epoch 18/49: 100% 400/400 [00:07<00:00, 58.37batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 118.62batch/s]
  Validating:  48% 24/50 [00:00<00:00, 113.40batch/s]
  Validating:  74% 37/50 [00:00<00:00, 118.78batch/s]
Epoch 18  │ train_loss=0.0102  │ val_loss=0.0039  │ val_acc=0.9984
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.004
Epoch 19/49:  99% 396/400 [00:07<00:00, 38.38batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 120.38batch/s]
  Validating:  54% 27/50 [00:00<00:00, 129.49batch/s]
  Validating:  80% 40/50 [00:00<00:00, 129.04batch/s]
Epoch 19  │ train_loss=0.0086  │ val_loss=0.0038  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 20/49:  99% 395/400 [00:06<00:00, 58.78batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 131.82batch/s]
  Validating:  56% 28/50 [00:00<00:00, 116.91batch/s]
  Validating:  82% 41/50 [00:00<00:00, 122.01batch/s]
Epoch 20  │ train_loss=0.0089  │ val_loss=0.0036  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 21/49:  98% 394/400 [00:07<00:00, 59.68batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 133.17batch/s]
  Validating:  56% 28/50 [00:00<00:00, 118.65batch/s]
  Validating:  84% 42/50 [00:00<00:00, 125.98batch/s]
Epoch 21  │ train_loss=0.0082  │ val_loss=0.0034  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 22/49:  99% 396/400 [00:07<00:00, 39.70batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  18% 9/50 [00:00<00:00, 86.92batch/s]
  Validating:  38% 19/50 [00:00<00:00, 92.20batch/s]
  Validating:  58% 29/50 [00:00<00:00, 90.58batch/s]
  Validating:  78% 39/50 [00:00<00:00, 89.31batch/s]
  Validating:  96% 48/50 [00:00<00:00, 84.68batch/s]
Epoch 22  │ train_loss=0.0070  │ val_loss=0.0032  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 23/49:  99% 395/400 [00:07<00:00, 61.30batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 133.13batch/s]
  Validating:  56% 28/50 [00:00<00:00, 121.72batch/s]
  Validating:  84% 42/50 [00:00<00:00, 126.79batch/s]
Epoch 23  │ train_loss=0.0072  │ val_loss=0.0029  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 24/49:  99% 395/400 [00:07<00:00, 58.29batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 125.34batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.98batch/s]
  Validating:  82% 41/50 [00:00<00:00, 132.35batch/s]
Epoch 24  │ train_loss=0.0061  │ val_loss=0.0029  │ val_acc=0.9994
Epoch 25/49:  99% 395/400 [00:06<00:00, 59.72batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 126.07batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.42batch/s]
  Validating:  82% 41/50 [00:00<00:00, 133.78batch/s]
Epoch 25  │ train_loss=0.0061  │ val_loss=0.0029  │ val_acc=0.9994
Epoch 26/49:  98% 394/400 [00:07<00:00, 59.84batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 126.24batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.48batch/s]
  Validating:  82% 41/50 [00:00<00:00, 132.81batch/s]
Epoch 26  │ train_loss=0.0060  │ val_loss=0.0027  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 27/49:  99% 395/400 [00:06<00:00, 51.81batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 90.91batch/s]
  Validating:  40% 20/50 [00:00<00:00, 93.94batch/s]
  Validating:  62% 31/50 [00:00<00:00, 98.09batch/s]
  Validating:  84% 42/50 [00:00<00:00, 101.40batch/s]
Epoch 27  │ train_loss=0.0052  │ val_loss=0.0028  │ val_acc=0.9991
Epoch 28/49:  98% 394/400 [00:07<00:00, 58.55batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 129.46batch/s]
  Validating:  52% 26/50 [00:00<00:00, 124.65batch/s]
  Validating:  80% 40/50 [00:00<00:00, 129.03batch/s]
Epoch 28  │ train_loss=0.0047  │ val_loss=0.0027  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 29/49:  99% 395/400 [00:07<00:00, 46.47batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 133.38batch/s]
  Validating:  56% 28/50 [00:00<00:00, 131.51batch/s]
  Validating:  84% 42/50 [00:00<00:00, 132.98batch/s]
Epoch 29  │ train_loss=0.0044  │ val_loss=0.0027  │ val_acc=0.9991
Epoch 30/49:  99% 395/400 [00:06<00:00, 60.63batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 118.07batch/s]
  Validating:  48% 24/50 [00:00<00:00, 116.12batch/s]
  Validating:  76% 38/50 [00:00<00:00, 124.11batch/s]
Epoch 30  │ train_loss=0.0039  │ val_loss=0.0028  │ val_acc=0.9991
Epoch 31/49:  99% 396/400 [00:07<00:00, 56.15batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 121.97batch/s]
  Validating:  52% 26/50 [00:00<00:00, 117.16batch/s]
  Validating:  78% 39/50 [00:00<00:00, 119.40batch/s]
Epoch 31  │ train_loss=0.0036  │ val_loss=0.0026  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 32/49: 100% 399/400 [00:07<00:00, 37.65batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 90.79batch/s]
  Validating:  40% 20/50 [00:00<00:00, 93.05batch/s]
  Validating:  60% 30/50 [00:00<00:00, 89.59batch/s]
  Validating:  78% 39/50 [00:00<00:00, 88.73batch/s]
  Validating:  96% 48/50 [00:00<00:00, 89.12batch/s]
Epoch 32  │ train_loss=0.0033  │ val_loss=0.0023  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 33/49: 100% 400/400 [00:08<00:00, 47.63batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 107.19batch/s]
  Validating:  46% 23/50 [00:00<00:00, 114.56batch/s]
  Validating:  70% 35/50 [00:00<00:00, 113.39batch/s]
  Validating:  94% 47/50 [00:00<00:00, 115.40batch/s]
Epoch 33  │ train_loss=0.0029  │ val_loss=0.0021  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 34/49:  99% 395/400 [00:08<00:00, 56.19batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 107.47batch/s]
  Validating:  44% 22/50 [00:00<00:00, 106.07batch/s]
  Validating:  66% 33/50 [00:00<00:00, 105.22batch/s]
  Validating:  88% 44/50 [00:00<00:00, 105.63batch/s]
Epoch 34  │ train_loss=0.0028  │ val_loss=0.0018  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 35/49: 100% 400/400 [00:07<00:00, 47.58batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 102.47batch/s]
  Validating:  44% 22/50 [00:00<00:00, 94.85batch/s] 
  Validating:  64% 32/50 [00:00<00:00, 84.77batch/s]
  Validating:  84% 42/50 [00:00<00:00, 89.47batch/s]
Epoch 35  │ train_loss=0.0025  │ val_loss=0.0016  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 36/49:  99% 395/400 [00:06<00:00, 60.49batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 115.79batch/s]
  Validating:  50% 25/50 [00:00<00:00, 119.17batch/s]
  Validating:  76% 38/50 [00:00<00:00, 122.98batch/s]
Epoch 36  │ train_loss=0.0022  │ val_loss=0.0015  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 37/49:  99% 396/400 [00:08<00:00, 46.45batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 110.24batch/s]
  Validating:  50% 25/50 [00:00<00:00, 117.90batch/s]
  Validating:  78% 39/50 [00:00<00:00, 123.69batch/s]
Epoch 37  │ train_loss=0.0020  │ val_loss=0.0015  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 38/49:  98% 394/400 [00:06<00:00, 56.01batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 122.51batch/s]
  Validating:  52% 26/50 [00:00<00:00, 126.26batch/s]
  Validating:  80% 40/50 [00:00<00:00, 130.26batch/s]
Epoch 38  │ train_loss=0.0013  │ val_loss=0.0015  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 39/49:  99% 397/400 [00:07<00:00, 58.14batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 132.29batch/s]
  Validating:  56% 28/50 [00:00<00:00, 130.57batch/s]
  Validating:  84% 42/50 [00:00<00:00, 126.99batch/s]
Epoch 39  │ train_loss=0.0014  │ val_loss=0.0015  │ val_acc=0.9994
Epoch 40/49: 100% 398/400 [00:07<00:00, 40.53batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 126.97batch/s]
  Validating:  54% 27/50 [00:00<00:00, 127.01batch/s]
  Validating:  80% 40/50 [00:00<00:00, 120.99batch/s]
Epoch 40  │ train_loss=0.0012  │ val_loss=0.0015  │ val_acc=0.9994
Epoch 41/49: 100% 398/400 [00:06<00:00, 61.41batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 122.60batch/s]
  Validating:  54% 27/50 [00:00<00:00, 127.93batch/s]
  Validating:  80% 40/50 [00:00<00:00, 123.72batch/s]
Epoch 41  │ train_loss=0.0010  │ val_loss=0.0014  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 42/49:  99% 397/400 [00:07<00:00, 58.48batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 138.75batch/s]
  Validating:  56% 28/50 [00:00<00:00, 132.74batch/s]
  Validating:  84% 42/50 [00:00<00:00, 132.42batch/s]
Epoch 42  │ train_loss=0.0009  │ val_loss=0.0014  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 43/49: 100% 399/400 [00:06<00:00, 56.01batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 130.54batch/s]
  Validating:  56% 28/50 [00:00<00:00, 129.89batch/s]
  Validating:  82% 41/50 [00:00<00:00, 128.74batch/s]
Epoch 43  │ train_loss=0.0008  │ val_loss=0.0013  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 44/49: 100% 399/400 [00:07<00:00, 60.17batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 132.64batch/s]
  Validating:  56% 28/50 [00:00<00:00, 135.75batch/s]
  Validating:  84% 42/50 [00:00<00:00, 133.67batch/s]
Epoch 44  │ train_loss=0.0006  │ val_loss=0.0013  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 45/49:  99% 397/400 [00:07<00:00, 43.68batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  16% 8/50 [00:00<00:00, 78.45batch/s]
  Validating:  34% 17/50 [00:00<00:00, 81.47batch/s]
  Validating:  62% 31/50 [00:00<00:00, 104.40batch/s]
  Validating:  88% 44/50 [00:00<00:00, 112.66batch/s]
Epoch 45  │ train_loss=0.0006  │ val_loss=0.0013  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 46/49:  98% 394/400 [00:06<00:00, 60.42batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 129.46batch/s]
  Validating:  52% 26/50 [00:00<00:00, 123.25batch/s]
  Validating:  78% 39/50 [00:00<00:00, 125.04batch/s]
Epoch 46  │ train_loss=0.0005  │ val_loss=0.0012  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 47/49:  99% 396/400 [00:07<00:00, 60.25batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 136.54batch/s]
  Validating:  58% 29/50 [00:00<00:00, 138.81batch/s]
  Validating:  86% 43/50 [00:00<00:00, 137.45batch/s]
Epoch 47  │ train_loss=0.0005  │ val_loss=0.0012  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 48/49:  99% 396/400 [00:06<00:00, 57.21batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 123.65batch/s]
  Validating:  54% 27/50 [00:00<00:00, 128.53batch/s]
  Validating:  82% 41/50 [00:00<00:00, 131.90batch/s]
Epoch 48  │ train_loss=0.0006  │ val_loss=0.0012  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
Epoch 49/49: 100% 399/400 [00:07<00:00, 61.33batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 113.21batch/s]
  Validating:  52% 26/50 [00:00<00:00, 124.13batch/s]
  Validating:  78% 39/50 [00:00<00:00, 125.58batch/s]
Epoch 49  │ train_loss=0.0005  │ val_loss=0.0012  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.001
`Trainer.fit` stopped: `max_epochs=50` reached.
2026-04-29 16:31:17,054 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_161306_z3_baseline_full/dataset_telecom/backbone_tcn/seed_42/checkpoints/last.ckpt
2026-04-29 16:31:18,764 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:31:18,765 [INFO] __main__:   telecom × tcn × seed=42: rollout_auc=9.9600, degradation_slope=0.0002
2026-04-29 16:31:18,765 [INFO] __main__:   telecom × tcn × seed=42: accuracy=0.9997, f1=0.9997
2026-04-29 16:31:18,767 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:31:20,897 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T256_D16_S43_R80_10
2026-04-29 16:31:20,897 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
/usr/local/lib/python3.12/dist-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
2026-04-29 16:31:20,902 [INFO] synapse.baselines.src.engine.train: Training backbone=tcn, params=117635
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:31:20,946 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  117 K │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 117 K                                                         
Non-trainable params: 0                                                         
Total params: 117 K                                                             
Total estimated model params size (MB): 0                                       
Modules in train mode: 57                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0/49:  99% 396/400 [00:07<00:00, 60.32batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 131.91batch/s]
  Validating:  56% 28/50 [00:00<00:00, 131.06batch/s]
  Validating:  84% 42/50 [00:00<00:00, 128.18batch/s]
Epoch 0  │ val_loss=0.6737  │ val_acc=0.8572
Metric val/loss improved. New best score: 0.674
Epoch 1/49: 100% 398/400 [00:06<00:00, 45.35batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 103.81batch/s]
  Validating:  44% 22/50 [00:00<00:00, 105.28batch/s]
  Validating:  66% 33/50 [00:00<00:00, 106.93batch/s]
  Validating:  90% 45/50 [00:00<00:00, 108.29batch/s]
Epoch 1  │ train_loss=0.2802  │ val_loss=0.2435  │ val_acc=0.8941
Metric val/loss improved by 0.430 >= min_delta = 0.0. New best score: 0.244
Epoch 2/49:  99% 397/400 [00:07<00:00, 60.51batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 121.60batch/s]
  Validating:  52% 26/50 [00:00<00:00, 123.36batch/s]
  Validating:  78% 39/50 [00:00<00:00, 114.50batch/s]
Epoch 2  │ train_loss=0.0605  │ val_loss=0.1120  │ val_acc=0.9622
Metric val/loss improved by 0.132 >= min_delta = 0.0. New best score: 0.112
Epoch 3/49: 100% 400/400 [00:07<00:00, 58.24batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 129.85batch/s]
  Validating:  52% 26/50 [00:00<00:00, 125.26batch/s]
  Validating:  78% 39/50 [00:00<00:00, 127.19batch/s]
Epoch 3  │ train_loss=0.0478  │ val_loss=0.0442  │ val_acc=0.9884
Metric val/loss improved by 0.068 >= min_delta = 0.0. New best score: 0.044
Epoch 4/49: 100% 398/400 [00:06<00:00, 59.21batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 121.70batch/s]
  Validating:  54% 27/50 [00:00<00:00, 127.62batch/s]
  Validating:  80% 40/50 [00:00<00:00, 123.42batch/s]
Epoch 4  │ train_loss=0.0348  │ val_loss=0.0305  │ val_acc=0.9922
Metric val/loss improved by 0.014 >= min_delta = 0.0. New best score: 0.031
Epoch 5/49:  98% 394/400 [00:07<00:00, 60.67batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 103.78batch/s]
  Validating:  48% 24/50 [00:00<00:00, 116.20batch/s]
  Validating:  76% 38/50 [00:00<00:00, 123.01batch/s]
Epoch 5  │ train_loss=0.0295  │ val_loss=0.0249  │ val_acc=0.9937
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.025
Epoch 6/49:  99% 396/400 [00:07<00:00, 37.45batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  18% 9/50 [00:00<00:00, 80.24batch/s]
  Validating:  36% 18/50 [00:00<00:00, 67.53batch/s]
  Validating:  52% 26/50 [00:00<00:00, 70.75batch/s]
  Validating:  70% 35/50 [00:00<00:00, 75.21batch/s]
  Validating:  86% 43/50 [00:00<00:00, 74.62batch/s]
Epoch 6  │ train_loss=0.0260  │ val_loss=0.0229  │ val_acc=0.9944
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.023
Epoch 7/49: 100% 400/400 [00:08<00:00, 49.21batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 108.07batch/s]
  Validating:  46% 23/50 [00:00<00:00, 112.97batch/s]
  Validating:  72% 36/50 [00:00<00:00, 116.57batch/s]
  Validating:  96% 48/50 [00:00<00:00, 116.28batch/s]
Epoch 7  │ train_loss=0.0237  │ val_loss=0.0222  │ val_acc=0.9941
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.022
Epoch 8/49: 100% 398/400 [00:08<00:00, 48.86batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 122.12batch/s]
  Validating:  52% 26/50 [00:00<00:00, 124.49batch/s]
  Validating:  78% 39/50 [00:00<00:00, 119.37batch/s]
Epoch 8  │ train_loss=0.0193  │ val_loss=0.0218  │ val_acc=0.9941
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.022
Epoch 9/49:  99% 396/400 [00:08<00:00, 48.58batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 98.98batch/s]
  Validating:  44% 22/50 [00:00<00:00, 109.20batch/s]
  Validating:  68% 34/50 [00:00<00:00, 112.39batch/s]
  Validating:  92% 46/50 [00:00<00:00, 112.28batch/s]
Epoch 9  │ train_loss=0.0171  │ val_loss=0.0208  │ val_acc=0.9941
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.021
Epoch 10/49: 100% 399/400 [00:08<00:00, 49.53batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 116.70batch/s]
  Validating:  48% 24/50 [00:00<00:00, 117.92batch/s]
  Validating:  72% 36/50 [00:00<00:00, 118.83batch/s]
  Validating:  96% 48/50 [00:00<00:00, 118.64batch/s]
Epoch 10  │ train_loss=0.0154  │ val_loss=0.0206  │ val_acc=0.9941
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.021
Epoch 11/49:  99% 396/400 [00:08<00:00, 51.63batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 108.24batch/s]
  Validating:  48% 24/50 [00:00<00:00, 119.02batch/s]
  Validating:  72% 36/50 [00:00<00:00, 117.40batch/s]
  Validating:  98% 49/50 [00:00<00:00, 119.21batch/s]
Epoch 11  │ train_loss=0.0232  │ val_loss=0.0194  │ val_acc=0.9944
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.019
Epoch 12/49:  99% 396/400 [00:09<00:00, 53.03batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 126.46batch/s]
  Validating:  52% 26/50 [00:00<00:00, 122.55batch/s]
  Validating:  80% 40/50 [00:00<00:00, 126.51batch/s]
Epoch 12  │ train_loss=0.0140  │ val_loss=0.0183  │ val_acc=0.9944
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.018
Epoch 13/49: 100% 399/400 [00:07<00:00, 45.24batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  14% 7/50 [00:00<00:00, 69.85batch/s]
  Validating:  28% 14/50 [00:00<00:00, 65.24batch/s]
  Validating:  48% 24/50 [00:00<00:00, 78.10batch/s]
  Validating:  66% 33/50 [00:00<00:00, 82.20batch/s]
  Validating:  86% 43/50 [00:00<00:00, 87.71batch/s]
Epoch 13  │ train_loss=0.0136  │ val_loss=0.0171  │ val_acc=0.9947
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.017
Epoch 14/49:  99% 397/400 [00:07<00:00, 56.40batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 130.78batch/s]
  Validating:  56% 28/50 [00:00<00:00, 126.65batch/s]
  Validating:  82% 41/50 [00:00<00:00, 127.65batch/s]
Epoch 14  │ train_loss=0.0147  │ val_loss=0.0153  │ val_acc=0.9953
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.015
Epoch 15/49: 100% 400/400 [00:08<00:00, 51.93batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 111.07batch/s]
  Validating:  48% 24/50 [00:00<00:00, 112.75batch/s]
  Validating:  72% 36/50 [00:00<00:00, 113.31batch/s]
  Validating:  96% 48/50 [00:00<00:00, 115.09batch/s]
Epoch 15  │ train_loss=0.0123  │ val_loss=0.0139  │ val_acc=0.9956
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.014
Epoch 16/49:  99% 397/400 [00:07<00:00, 38.59batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  16% 8/50 [00:00<00:00, 73.51batch/s]
  Validating:  32% 16/50 [00:00<00:00, 75.26batch/s]
  Validating:  48% 24/50 [00:00<00:00, 77.07batch/s]
  Validating:  66% 33/50 [00:00<00:00, 79.92batch/s]
  Validating:  82% 41/50 [00:00<00:00, 78.12batch/s]
  Validating:  98% 49/50 [00:00<00:00, 74.57batch/s]
Epoch 16  │ train_loss=0.0101  │ val_loss=0.0123  │ val_acc=0.9959
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.012
Epoch 17/49:  99% 396/400 [00:08<00:00, 58.19batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 127.02batch/s]
  Validating:  56% 28/50 [00:00<00:00, 132.81batch/s]
  Validating:  84% 42/50 [00:00<00:00, 135.54batch/s]
Epoch 17  │ train_loss=0.0100  │ val_loss=0.0118  │ val_acc=0.9959
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.012
Epoch 18/49:  99% 397/400 [00:07<00:00, 58.78batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 123.36batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.32batch/s]
  Validating:  82% 41/50 [00:00<00:00, 130.94batch/s]
Epoch 18  │ train_loss=0.0112  │ val_loss=0.0109  │ val_acc=0.9962
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.011
Epoch 19/49: 100% 398/400 [00:06<00:00, 61.32batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 119.30batch/s]
  Validating:  52% 26/50 [00:00<00:00, 128.54batch/s]
  Validating:  80% 40/50 [00:00<00:00, 130.90batch/s]
Epoch 19  │ train_loss=0.0090  │ val_loss=0.0101  │ val_acc=0.9966
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.010
Epoch 20/49:  99% 396/400 [00:07<00:00, 59.51batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 125.27batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.11batch/s]
  Validating:  82% 41/50 [00:00<00:00, 128.19batch/s]
Epoch 20  │ train_loss=0.0090  │ val_loss=0.0095  │ val_acc=0.9969
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.009
Epoch 21/49: 100% 398/400 [00:07<00:00, 46.98batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  16% 8/50 [00:00<00:00, 78.06batch/s]
  Validating:  34% 17/50 [00:00<00:00, 80.62batch/s]
  Validating:  54% 27/50 [00:00<00:00, 87.22batch/s]
  Validating:  74% 37/50 [00:00<00:00, 91.55batch/s]
  Validating:  94% 47/50 [00:00<00:00, 91.37batch/s]
Epoch 21  │ train_loss=0.0093  │ val_loss=0.0082  │ val_acc=0.9975
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.008
Epoch 22/49:  99% 397/400 [00:07<00:00, 46.72batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 106.01batch/s]
  Validating:  44% 22/50 [00:00<00:00, 105.06batch/s]
  Validating:  66% 33/50 [00:00<00:00, 101.29batch/s]
  Validating:  88% 44/50 [00:00<00:00, 101.02batch/s]
Epoch 22  │ train_loss=0.0061  │ val_loss=0.0074  │ val_acc=0.9978
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.007
Epoch 23/49:  99% 396/400 [00:09<00:00, 55.74batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 130.86batch/s]
  Validating:  58% 29/50 [00:00<00:00, 137.03batch/s]
  Validating:  86% 43/50 [00:00<00:00, 126.47batch/s]
Epoch 23  │ train_loss=0.0057  │ val_loss=0.0071  │ val_acc=0.9978
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.007
Epoch 24/49: 100% 398/400 [00:08<00:00, 31.37batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  16% 8/50 [00:00<00:00, 69.91batch/s]
  Validating:  36% 18/50 [00:00<00:00, 84.04batch/s]
  Validating:  58% 29/50 [00:00<00:00, 95.32batch/s]
  Validating:  82% 41/50 [00:00<00:00, 101.86batch/s]
Epoch 24  │ train_loss=0.0066  │ val_loss=0.0065  │ val_acc=0.9978
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.006
Epoch 25/49: 100% 399/400 [00:07<00:00, 51.85batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 125.49batch/s]
  Validating:  52% 26/50 [00:00<00:00, 121.96batch/s]
  Validating:  78% 39/50 [00:00<00:00, 117.17batch/s]
Epoch 25  │ train_loss=0.0057  │ val_loss=0.0061  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.006
Epoch 26/49:  99% 397/400 [00:08<00:00, 55.32batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 108.84batch/s]
  Validating:  48% 24/50 [00:00<00:00, 116.74batch/s]
  Validating:  74% 37/50 [00:00<00:00, 122.37batch/s]
  Validating: 100% 50/50 [00:00<00:00, 121.65batch/s]
Epoch 26  │ train_loss=0.0049  │ val_loss=0.0058  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.006
Epoch 27/49:  99% 397/400 [00:07<00:00, 36.78batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  16% 8/50 [00:00<00:00, 73.68batch/s]
  Validating:  32% 16/50 [00:00<00:00, 67.40batch/s]
  Validating:  46% 23/50 [00:00<00:00, 66.92batch/s]
  Validating:  62% 31/50 [00:00<00:00, 69.72batch/s]
  Validating:  84% 42/50 [00:00<00:00, 81.31batch/s]
Epoch 27  │ train_loss=0.0057  │ val_loss=0.0057  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.006
Epoch 28/49:  99% 397/400 [00:07<00:00, 51.01batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 107.45batch/s]
  Validating:  48% 24/50 [00:00<00:00, 113.81batch/s]
  Validating:  74% 37/50 [00:00<00:00, 119.93batch/s]
  Validating: 100% 50/50 [00:00<00:00, 120.22batch/s]
Epoch 28  │ train_loss=0.0042  │ val_loss=0.0056  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.006
Epoch 29/49: 100% 400/400 [00:07<00:00, 52.49batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 97.15batch/s]
  Validating:  44% 22/50 [00:00<00:00, 108.61batch/s]
  Validating:  68% 34/50 [00:00<00:00, 110.38batch/s]
  Validating:  94% 47/50 [00:00<00:00, 116.14batch/s]
Epoch 29  │ train_loss=0.0041  │ val_loss=0.0051  │ val_acc=0.9978
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.005
Epoch 30/49:  99% 395/400 [00:08<00:00, 38.50batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 107.99batch/s]
  Validating:  44% 22/50 [00:00<00:00, 104.19batch/s]
  Validating:  68% 34/50 [00:00<00:00, 108.39batch/s]
  Validating:  92% 46/50 [00:00<00:00, 112.37batch/s]
Epoch 30  │ train_loss=0.0043  │ val_loss=0.0046  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.005
Epoch 31/49:  99% 397/400 [00:07<00:00, 57.53batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 124.88batch/s]
  Validating:  52% 26/50 [00:00<00:00, 126.50batch/s]
  Validating:  78% 39/50 [00:00<00:00, 123.88batch/s]
Epoch 31  │ train_loss=0.0030  │ val_loss=0.0044  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 32/49: 100% 400/400 [00:08<00:00, 50.26batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 107.80batch/s]
  Validating:  44% 22/50 [00:00<00:00, 101.00batch/s]
  Validating:  68% 34/50 [00:00<00:00, 105.40batch/s]
  Validating:  90% 45/50 [00:00<00:00, 103.52batch/s]
Epoch 32  │ train_loss=0.0032  │ val_loss=0.0043  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 33/49:  99% 397/400 [00:08<00:00, 40.44batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.90batch/s]
  Validating:  52% 26/50 [00:00<00:00, 127.58batch/s]
  Validating:  78% 39/50 [00:00<00:00, 128.15batch/s]
Epoch 33  │ train_loss=0.0025  │ val_loss=0.0041  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 34/49: 100% 400/400 [00:06<00:00, 61.11batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 117.00batch/s]
  Validating:  50% 25/50 [00:00<00:00, 120.96batch/s]
  Validating:  78% 39/50 [00:00<00:00, 127.37batch/s]
Epoch 34  │ train_loss=0.0027  │ val_loss=0.0038  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 35/49: 100% 399/400 [00:08<00:00, 49.37batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 110.78batch/s]
  Validating:  48% 24/50 [00:00<00:00, 112.10batch/s]
  Validating:  72% 36/50 [00:00<00:00, 110.88batch/s]
  Validating:  98% 49/50 [00:00<00:00, 115.89batch/s]
Epoch 35  │ train_loss=0.0020  │ val_loss=0.0036  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 36/49: 100% 398/400 [00:08<00:00, 37.98batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  16% 8/50 [00:00<00:00, 77.30batch/s]
  Validating:  34% 17/50 [00:00<00:00, 81.48batch/s]
  Validating:  52% 26/50 [00:00<00:00, 82.75batch/s]
  Validating:  70% 35/50 [00:00<00:00, 79.62batch/s]
  Validating:  88% 44/50 [00:00<00:00, 80.48batch/s]
Epoch 36  │ train_loss=0.0015  │ val_loss=0.0035  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.004
Epoch 37/49:  99% 396/400 [00:06<00:00, 61.61batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 125.88batch/s]
  Validating:  54% 27/50 [00:00<00:00, 129.61batch/s]
  Validating:  82% 41/50 [00:00<00:00, 131.98batch/s]
Epoch 37  │ train_loss=0.0012  │ val_loss=0.0033  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 38/49:  99% 396/400 [00:07<00:00, 58.17batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.68batch/s]
  Validating:  52% 26/50 [00:00<00:00, 126.53batch/s]
  Validating:  78% 39/50 [00:00<00:00, 127.35batch/s]
Epoch 38  │ train_loss=0.0010  │ val_loss=0.0029  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 39/49: 100% 398/400 [00:06<00:00, 60.94batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 122.24batch/s]
  Validating:  54% 27/50 [00:00<00:00, 128.57batch/s]
  Validating:  80% 40/50 [00:00<00:00, 127.41batch/s]
Epoch 39  │ train_loss=0.0011  │ val_loss=0.0028  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 40/49: 100% 400/400 [00:09<00:00, 51.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 120.44batch/s]
  Validating:  52% 26/50 [00:00<00:00, 121.53batch/s]
  Validating:  78% 39/50 [00:00<00:00, 117.03batch/s]
Epoch 40  │ train_loss=0.0008  │ val_loss=0.0027  │ val_acc=0.9991
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 41/49: 100% 398/400 [00:08<00:00, 43.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 121.75batch/s]
  Validating:  52% 26/50 [00:00<00:00, 111.54batch/s]
  Validating:  76% 38/50 [00:00<00:00, 115.03batch/s]
  Validating: 100% 50/50 [00:00<00:00, 113.91batch/s]
Epoch 41  │ train_loss=0.0008  │ val_loss=0.0027  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 42/49: 100% 399/400 [00:07<00:00, 58.17batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 125.16batch/s]
  Validating:  52% 26/50 [00:00<00:00, 114.60batch/s]
  Validating:  80% 40/50 [00:00<00:00, 123.76batch/s]
Epoch 42  │ train_loss=0.0007  │ val_loss=0.0026  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 43/49: 100% 398/400 [00:07<00:00, 59.86batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 125.13batch/s]
  Validating:  54% 27/50 [00:00<00:00, 128.41batch/s]
  Validating:  80% 40/50 [00:00<00:00, 126.83batch/s]
Epoch 43  │ train_loss=0.0005  │ val_loss=0.0025  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.003
Epoch 44/49:  99% 397/400 [00:07<00:00, 38.09batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  18% 9/50 [00:00<00:00, 81.34batch/s]
  Validating:  36% 18/50 [00:00<00:00, 82.69batch/s]
  Validating:  54% 27/50 [00:00<00:00, 78.38batch/s]
  Validating:  70% 35/50 [00:00<00:00, 76.48batch/s]
  Validating:  88% 44/50 [00:00<00:00, 78.75batch/s]
Epoch 44  │ train_loss=0.0004  │ val_loss=0.0024  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 45/49:  99% 396/400 [00:07<00:00, 60.33batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.31batch/s]
  Validating:  54% 27/50 [00:00<00:00, 131.27batch/s]
  Validating:  82% 41/50 [00:00<00:00, 131.23batch/s]
Epoch 45  │ train_loss=0.0003  │ val_loss=0.0024  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 46/49: 100% 399/400 [00:07<00:00, 54.45batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 111.69batch/s]
  Validating:  48% 24/50 [00:00<00:00, 115.60batch/s]
  Validating:  72% 36/50 [00:00<00:00, 113.21batch/s]
  Validating:  96% 48/50 [00:00<00:00, 111.34batch/s]
Epoch 46  │ train_loss=0.0004  │ val_loss=0.0023  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 47/49: 100% 400/400 [00:08<00:00, 42.83batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  18% 9/50 [00:00<00:00, 89.96batch/s]
  Validating:  38% 19/50 [00:00<00:00, 94.11batch/s]
  Validating:  58% 29/50 [00:00<00:00, 84.71batch/s]
  Validating:  78% 39/50 [00:00<00:00, 89.39batch/s]
  Validating:  98% 49/50 [00:00<00:00, 92.86batch/s]
Epoch 47  │ train_loss=0.0003  │ val_loss=0.0023  │ val_acc=0.9994
Epoch 48/49: 100% 399/400 [00:07<00:00, 55.82batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 113.14batch/s]
  Validating:  48% 24/50 [00:00<00:00, 114.53batch/s]
  Validating:  72% 36/50 [00:00<00:00, 112.37batch/s]
  Validating:  96% 48/50 [00:00<00:00, 112.17batch/s]
Epoch 48  │ train_loss=0.0003  │ val_loss=0.0023  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
Epoch 49/49:  99% 395/400 [00:08<00:00, 52.64batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 111.28batch/s]
  Validating:  48% 24/50 [00:00<00:00, 115.88batch/s]
  Validating:  72% 36/50 [00:00<00:00, 115.15batch/s]
  Validating:  98% 49/50 [00:00<00:00, 117.68batch/s]
Epoch 49  │ train_loss=0.0003  │ val_loss=0.0023  │ val_acc=0.9994
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.002
`Trainer.fit` stopped: `max_epochs=50` reached.
2026-04-29 16:38:22,025 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_161306_z3_baseline_full/dataset_telecom/backbone_tcn/seed_43/checkpoints/last.ckpt
2026-04-29 16:38:23,867 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:38:23,867 [INFO] __main__:   telecom × tcn × seed=43: rollout_auc=9.8700, degradation_slope=-0.0056
2026-04-29 16:38:23,867 [INFO] __main__:   telecom × tcn × seed=43: accuracy=0.9984, f1=0.9984
2026-04-29 16:38:23,870 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:38:26,198 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T256_D16_S44_R80_10
2026-04-29 16:38:26,198 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
/usr/local/lib/python3.12/dist-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
2026-04-29 16:38:26,203 [INFO] synapse.baselines.src.engine.train: Training backbone=tcn, params=117635
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:38:26,249 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  117 K │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 117 K                                                         
Non-trainable params: 0                                                         
Total params: 117 K                                                             
Total estimated model params size (MB): 0                                       
Modules in train mode: 57                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0/49: 100% 398/400 [00:08<00:00, 47.38batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 116.65batch/s]
  Validating:  50% 25/50 [00:00<00:00, 118.38batch/s]
  Validating:  74% 37/50 [00:00<00:00, 115.68batch/s]
  Validating:  98% 49/50 [00:00<00:00, 116.86batch/s]
Epoch 0  │ val_loss=0.6914  │ val_acc=0.9184
Metric val/loss improved. New best score: 0.691
Epoch 1/49: 100% 400/400 [00:09<00:00, 32.38batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  14% 7/50 [00:00<00:00, 66.97batch/s]
  Validating:  28% 14/50 [00:00<00:00, 68.40batch/s]
  Validating:  44% 22/50 [00:00<00:00, 71.00batch/s]
  Validating:  60% 30/50 [00:00<00:00, 70.43batch/s]
  Validating:  76% 38/50 [00:00<00:00, 69.99batch/s]
  Validating:  92% 46/50 [00:00<00:00, 71.72batch/s]
Epoch 1  │ train_loss=0.2906  │ val_loss=0.1993  │ val_acc=0.9438
Metric val/loss improved by 0.492 >= min_delta = 0.0. New best score: 0.199
Epoch 2/49:  99% 395/400 [00:07<00:00, 58.03batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 127.19batch/s]
  Validating:  54% 27/50 [00:00<00:00, 128.39batch/s]
  Validating:  80% 40/50 [00:00<00:00, 125.32batch/s]
Epoch 2  │ train_loss=0.0636  │ val_loss=0.1019  │ val_acc=0.9544
Metric val/loss improved by 0.097 >= min_delta = 0.0. New best score: 0.102
Epoch 3/49: 100% 400/400 [00:07<00:00, 59.42batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 128.34batch/s]
  Validating:  56% 28/50 [00:00<00:00, 130.02batch/s]
  Validating:  84% 42/50 [00:00<00:00, 134.22batch/s]
Epoch 3  │ train_loss=0.0446  │ val_loss=0.0539  │ val_acc=0.9794
Metric val/loss improved by 0.048 >= min_delta = 0.0. New best score: 0.054
Epoch 4/49: 100% 398/400 [00:06<00:00, 58.62batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 130.71batch/s]
  Validating:  56% 28/50 [00:00<00:00, 123.69batch/s]
  Validating:  82% 41/50 [00:00<00:00, 126.22batch/s]
Epoch 4  │ train_loss=0.0349  │ val_loss=0.0386  │ val_acc=0.9856
Metric val/loss improved by 0.015 >= min_delta = 0.0. New best score: 0.039
Epoch 5/49: 100% 400/400 [00:07<00:00, 58.63batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 129.77batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.31batch/s]
  Validating:  82% 41/50 [00:00<00:00, 119.48batch/s]
Epoch 5  │ train_loss=0.0288  │ val_loss=0.0301  │ val_acc=0.9903
Metric val/loss improved by 0.008 >= min_delta = 0.0. New best score: 0.030
Epoch 6/49:  99% 396/400 [00:07<00:00, 42.06batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  16% 8/50 [00:00<00:00, 78.55batch/s]
  Validating:  34% 17/50 [00:00<00:00, 82.64batch/s]
  Validating:  52% 26/50 [00:00<00:00, 84.99batch/s]
  Validating:  70% 35/50 [00:00<00:00, 86.37batch/s]
  Validating:  96% 48/50 [00:00<00:00, 101.36batch/s]
Epoch 6  │ train_loss=0.0250  │ val_loss=0.0249  │ val_acc=0.9934
Metric val/loss improved by 0.005 >= min_delta = 0.0. New best score: 0.025
Epoch 7/49: 100% 399/400 [00:06<00:00, 58.23batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 128.79batch/s]
  Validating:  56% 28/50 [00:00<00:00, 130.63batch/s]
  Validating:  84% 42/50 [00:00<00:00, 127.15batch/s]
Epoch 7  │ train_loss=0.0207  │ val_loss=0.0223  │ val_acc=0.9937
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.022
Epoch 8/49:  99% 395/400 [00:07<00:00, 59.91batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 131.55batch/s]
  Validating:  56% 28/50 [00:00<00:00, 119.41batch/s]
  Validating:  84% 42/50 [00:00<00:00, 126.07batch/s]
Epoch 8  │ train_loss=0.0198  │ val_loss=0.0206  │ val_acc=0.9941
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.021
Epoch 9/49:  99% 396/400 [00:06<00:00, 60.00batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 130.23batch/s]
  Validating:  56% 28/50 [00:00<00:00, 117.97batch/s]
  Validating:  86% 43/50 [00:00<00:00, 127.69batch/s]
Epoch 9  │ train_loss=0.0181  │ val_loss=0.0199  │ val_acc=0.9941
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.020
Epoch 10/49: 100% 398/400 [00:07<00:00, 59.82batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 126.27batch/s]
  Validating:  52% 26/50 [00:00<00:00, 120.77batch/s]
  Validating:  80% 40/50 [00:00<00:00, 125.63batch/s]
Epoch 10  │ train_loss=0.0173  │ val_loss=0.0190  │ val_acc=0.9950
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.019
Epoch 11/49: 100% 398/400 [00:07<00:00, 43.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  18% 9/50 [00:00<00:00, 83.72batch/s]
  Validating:  36% 18/50 [00:00<00:00, 76.12batch/s]
  Validating:  54% 27/50 [00:00<00:00, 81.50batch/s]
  Validating:  72% 36/50 [00:00<00:00, 81.57batch/s]
  Validating:  92% 46/50 [00:00<00:00, 86.91batch/s]
Epoch 11  │ train_loss=0.0165  │ val_loss=0.0181  │ val_acc=0.9950
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.018
Epoch 12/49: 100% 398/400 [00:06<00:00, 59.58batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 126.39batch/s]
  Validating:  52% 26/50 [00:00<00:00, 122.29batch/s]
  Validating:  78% 39/50 [00:00<00:00, 121.63batch/s]
Epoch 12  │ train_loss=0.0145  │ val_loss=0.0173  │ val_acc=0.9953
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.017
Epoch 13/49: 100% 398/400 [00:07<00:00, 57.91batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 134.28batch/s]
  Validating:  56% 28/50 [00:00<00:00, 132.13batch/s]
  Validating:  84% 42/50 [00:00<00:00, 131.98batch/s]
Epoch 13  │ train_loss=0.0135  │ val_loss=0.0165  │ val_acc=0.9953
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.016
Epoch 14/49: 100% 399/400 [00:06<00:00, 59.85batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 132.05batch/s]
  Validating:  58% 29/50 [00:00<00:00, 137.43batch/s]
  Validating:  86% 43/50 [00:00<00:00, 134.65batch/s]
Epoch 14  │ train_loss=0.0149  │ val_loss=0.0157  │ val_acc=0.9953
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.016
Epoch 15/49: 100% 400/400 [00:07<00:00, 59.75batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 132.29batch/s]
  Validating:  56% 28/50 [00:00<00:00, 132.68batch/s]
  Validating:  84% 42/50 [00:00<00:00, 121.49batch/s]
Epoch 15  │ train_loss=0.0120  │ val_loss=0.0143  │ val_acc=0.9956
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.014
Epoch 16/49: 100% 398/400 [00:07<00:00, 45.68batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  18% 9/50 [00:00<00:00, 85.78batch/s]
  Validating:  36% 18/50 [00:00<00:00, 87.01batch/s]
  Validating:  56% 28/50 [00:00<00:00, 90.93batch/s]
  Validating:  76% 38/50 [00:00<00:00, 92.50batch/s]
  Validating:  96% 48/50 [00:00<00:00, 90.94batch/s]
Epoch 16  │ train_loss=0.0105  │ val_loss=0.0132  │ val_acc=0.9962
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.013
Epoch 17/49: 100% 399/400 [00:06<00:00, 57.37batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 131.11batch/s]
  Validating:  56% 28/50 [00:00<00:00, 130.30batch/s]
  Validating:  84% 42/50 [00:00<00:00, 128.08batch/s]
Epoch 17  │ train_loss=0.0093  │ val_loss=0.0124  │ val_acc=0.9975
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.012
Epoch 18/49:  99% 397/400 [00:07<00:00, 60.48batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 133.73batch/s]
  Validating:  56% 28/50 [00:00<00:00, 133.16batch/s]
  Validating:  86% 43/50 [00:00<00:00, 136.82batch/s]
Epoch 18  │ train_loss=0.0071  │ val_loss=0.0121  │ val_acc=0.9972
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.012
Epoch 19/49: 100% 400/400 [00:06<00:00, 59.66batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 136.13batch/s]
  Validating:  56% 28/50 [00:00<00:00, 135.81batch/s]
  Validating:  84% 42/50 [00:00<00:00, 132.97batch/s]
Epoch 19  │ train_loss=0.0088  │ val_loss=0.0118  │ val_acc=0.9972
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.012
Epoch 20/49:  98% 394/400 [00:07<00:00, 61.33batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 137.70batch/s]
  Validating:  56% 28/50 [00:00<00:00, 136.34batch/s]
  Validating:  84% 42/50 [00:00<00:00, 132.33batch/s]
Epoch 20  │ train_loss=0.0069  │ val_loss=0.0116  │ val_acc=0.9975
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.012
Epoch 21/49:  99% 396/400 [00:06<00:00, 46.63batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 106.56batch/s]
  Validating:  44% 22/50 [00:00<00:00, 103.35batch/s]
  Validating:  66% 33/50 [00:00<00:00, 103.79batch/s]
  Validating:  88% 44/50 [00:00<00:00, 104.37batch/s]
Epoch 21  │ train_loss=0.0060  │ val_loss=0.0115  │ val_acc=0.9978
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.011
Epoch 22/49:  99% 396/400 [00:06<00:00, 58.66batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.06batch/s]
  Validating:  52% 26/50 [00:00<00:00, 127.35batch/s]
  Validating:  80% 40/50 [00:00<00:00, 130.36batch/s]
Epoch 22  │ train_loss=0.0059  │ val_loss=0.0115  │ val_acc=0.9978
Epoch 23/49:  98% 394/400 [00:07<00:00, 60.11batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 121.48batch/s]
  Validating:  54% 27/50 [00:00<00:00, 128.09batch/s]
  Validating:  80% 40/50 [00:00<00:00, 123.20batch/s]
Epoch 23  │ train_loss=0.0069  │ val_loss=0.0111  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.011
Epoch 24/49: 100% 399/400 [00:06<00:00, 61.13batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.98batch/s]
  Validating:  54% 27/50 [00:00<00:00, 132.77batch/s]
  Validating:  82% 41/50 [00:00<00:00, 134.87batch/s]
Epoch 24  │ train_loss=0.0048  │ val_loss=0.0110  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.011
Epoch 25/49:  99% 397/400 [00:07<00:00, 60.19batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 133.77batch/s]
  Validating:  56% 28/50 [00:00<00:00, 128.05batch/s]
  Validating:  84% 42/50 [00:00<00:00, 130.94batch/s]
Epoch 25  │ train_loss=0.0043  │ val_loss=0.0108  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.011
Epoch 26/49: 100% 398/400 [00:06<00:00, 49.64batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 96.73batch/s]
  Validating:  40% 20/50 [00:00<00:00, 87.98batch/s]
  Validating:  60% 30/50 [00:00<00:00, 91.59batch/s]
  Validating:  80% 40/50 [00:00<00:00, 92.95batch/s]
Epoch 26  │ train_loss=0.0046  │ val_loss=0.0107  │ val_acc=0.9981
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.011
Epoch 27/49: 100% 399/400 [00:07<00:00, 61.42batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 123.96batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.06batch/s]
  Validating:  82% 41/50 [00:00<00:00, 130.95batch/s]
Epoch 27  │ train_loss=0.0046  │ val_loss=0.0105  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 28/49: 100% 400/400 [00:07<00:00, 53.09batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 136.69batch/s]
  Validating:  56% 28/50 [00:00<00:00, 128.90batch/s]
  Validating:  84% 42/50 [00:00<00:00, 130.74batch/s]
Epoch 28  │ train_loss=0.0034  │ val_loss=0.0104  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 29/49:  99% 397/400 [00:07<00:00, 52.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 119.64batch/s]
  Validating:  50% 25/50 [00:00<00:00, 123.31batch/s]
  Validating:  76% 38/50 [00:00<00:00, 117.89batch/s]
Epoch 29  │ train_loss=0.0035  │ val_loss=0.0104  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 30/49:  99% 396/400 [00:08<00:00, 51.09batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 127.13batch/s]
  Validating:  54% 27/50 [00:00<00:00, 131.59batch/s]
  Validating:  82% 41/50 [00:00<00:00, 129.14batch/s]
Epoch 30  │ train_loss=0.0037  │ val_loss=0.0104  │ val_acc=0.9984
Epoch 31/49: 100% 398/400 [00:07<00:00, 45.88batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 105.51batch/s]
  Validating:  44% 22/50 [00:00<00:00, 98.26batch/s] 
  Validating:  64% 32/50 [00:00<00:00, 95.20batch/s]
  Validating:  84% 42/50 [00:00<00:00, 92.55batch/s]
Epoch 31  │ train_loss=0.0030  │ val_loss=0.0104  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 32/49: 100% 400/400 [00:07<00:00, 60.47batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 131.16batch/s]
  Validating:  56% 28/50 [00:00<00:00, 133.92batch/s]
  Validating:  84% 42/50 [00:00<00:00, 135.14batch/s]
Epoch 32  │ train_loss=0.0023  │ val_loss=0.0103  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 33/49: 100% 398/400 [00:08<00:00, 60.47batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 123.70batch/s]
  Validating:  54% 27/50 [00:00<00:00, 128.52batch/s]
  Validating:  80% 40/50 [00:00<00:00, 127.64batch/s]
Epoch 33  │ train_loss=0.0029  │ val_loss=0.0102  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 34/49: 100% 398/400 [00:07<00:00, 61.81batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 96.89batch/s]
  Validating:  40% 20/50 [00:00<00:00, 86.70batch/s]
  Validating:  62% 31/50 [00:00<00:00, 94.96batch/s]
  Validating:  82% 41/50 [00:00<00:00, 89.76batch/s]
Epoch 34  │ train_loss=0.0023  │ val_loss=0.0102  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 35/49: 100% 400/400 [00:08<00:00, 52.59batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 120.62batch/s]
  Validating:  52% 26/50 [00:00<00:00, 117.82batch/s]
  Validating:  76% 38/50 [00:00<00:00, 111.33batch/s]
  Validating: 100% 50/50 [00:00<00:00, 113.32batch/s]
Epoch 35  │ train_loss=0.0024  │ val_loss=0.0101  │ val_acc=0.9984
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 36/49:  99% 395/400 [00:08<00:00, 50.97batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 115.64batch/s]
  Validating:  48% 24/50 [00:00<00:00, 116.62batch/s]
  Validating:  74% 37/50 [00:00<00:00, 120.55batch/s]
  Validating: 100% 50/50 [00:00<00:00, 120.13batch/s]
Epoch 36  │ train_loss=0.0016  │ val_loss=0.0099  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 37/49:  99% 397/400 [00:06<00:00, 60.75batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 129.58batch/s]
  Validating:  56% 28/50 [00:00<00:00, 136.50batch/s]
  Validating:  84% 42/50 [00:00<00:00, 136.78batch/s]
Epoch 37  │ train_loss=0.0017  │ val_loss=0.0098  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 38/49: 100% 399/400 [00:07<00:00, 63.02batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 129.83batch/s]
  Validating:  54% 27/50 [00:00<00:00, 130.84batch/s]
  Validating:  84% 42/50 [00:00<00:00, 136.96batch/s]
Epoch 38  │ train_loss=0.0015  │ val_loss=0.0096  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.010
Epoch 39/49:  99% 395/400 [00:06<00:00, 45.09batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 109.22batch/s]
  Validating:  44% 22/50 [00:00<00:00, 100.71batch/s]
  Validating:  66% 33/50 [00:00<00:00, 96.86batch/s] 
  Validating:  90% 45/50 [00:00<00:00, 104.80batch/s]
Epoch 39  │ train_loss=0.0013  │ val_loss=0.0094  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.009
Epoch 40/49:  98% 394/400 [00:06<00:00, 62.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 136.82batch/s]
  Validating:  56% 28/50 [00:00<00:00, 130.61batch/s]
  Validating:  86% 43/50 [00:00<00:00, 137.87batch/s]
Epoch 40  │ train_loss=0.0012  │ val_loss=0.0091  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.009
Epoch 41/49:  99% 397/400 [00:07<00:00, 57.86batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.48batch/s]
  Validating:  52% 26/50 [00:00<00:00, 127.02batch/s]
  Validating:  82% 41/50 [00:00<00:00, 135.38batch/s]
Epoch 41  │ train_loss=0.0009  │ val_loss=0.0089  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.009
Epoch 42/49: 100% 399/400 [00:06<00:00, 62.85batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 142.59batch/s]
  Validating:  60% 30/50 [00:00<00:00, 135.06batch/s]
  Validating:  90% 45/50 [00:00<00:00, 138.91batch/s]
Epoch 42  │ train_loss=0.0009  │ val_loss=0.0087  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.009
Epoch 43/49: 100% 400/400 [00:07<00:00, 62.88batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 129.31batch/s]
  Validating:  56% 28/50 [00:00<00:00, 135.61batch/s]
  Validating:  84% 42/50 [00:00<00:00, 136.21batch/s]
Epoch 43  │ train_loss=0.0006  │ val_loss=0.0086  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.009
Epoch 44/49: 100% 399/400 [00:06<00:00, 61.22batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 138.70batch/s]
  Validating:  56% 28/50 [00:00<00:00, 131.71batch/s]
  Validating:  84% 42/50 [00:00<00:00, 134.00batch/s]
Epoch 44  │ train_loss=0.0005  │ val_loss=0.0085  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.008
Epoch 45/49: 100% 399/400 [00:07<00:00, 62.98batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 137.73batch/s]
  Validating:  58% 29/50 [00:00<00:00, 139.93batch/s]
  Validating:  88% 44/50 [00:00<00:00, 141.51batch/s]
Epoch 45  │ train_loss=0.0006  │ val_loss=0.0083  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.008
Epoch 46/49:  99% 396/400 [00:06<00:00, 46.60batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 108.12batch/s]
  Validating:  44% 22/50 [00:00<00:00, 107.22batch/s]
  Validating:  66% 33/50 [00:00<00:00, 107.84batch/s]
  Validating:  88% 44/50 [00:00<00:00, 104.33batch/s]
Epoch 46  │ train_loss=0.0006  │ val_loss=0.0082  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.008
Epoch 47/49:  99% 396/400 [00:06<00:00, 62.37batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  28% 14/50 [00:00<00:00, 138.32batch/s]
  Validating:  56% 28/50 [00:00<00:00, 137.01batch/s]
  Validating:  84% 42/50 [00:00<00:00, 132.14batch/s]
Epoch 47  │ train_loss=0.0005  │ val_loss=0.0082  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.008
Epoch 48/49: 100% 399/400 [00:07<00:00, 61.91batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.95batch/s]
  Validating:  56% 28/50 [00:00<00:00, 136.91batch/s]
  Validating:  86% 43/50 [00:00<00:00, 140.49batch/s]
Epoch 48  │ train_loss=0.0004  │ val_loss=0.0082  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.008
Epoch 49/49: 100% 399/400 [00:06<00:00, 60.90batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 141.97batch/s]
  Validating:  60% 30/50 [00:00<00:00, 143.83batch/s]
  Validating:  90% 45/50 [00:00<00:00, 140.93batch/s]
Epoch 49  │ train_loss=0.0004  │ val_loss=0.0081  │ val_acc=0.9987
Metric val/loss improved by 0.000 >= min_delta = 0.0. New best score: 0.008
`Trainer.fit` stopped: `max_epochs=50` reached.
2026-04-29 16:44:51,859 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_161306_z3_baseline_full/dataset_telecom/backbone_tcn/seed_44/checkpoints/last.ckpt
2026-04-29 16:44:53,446 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:44:53,447 [INFO] __main__:   telecom × tcn × seed=44: rollout_auc=10.0000, degradation_slope=0.0000
2026-04-29 16:44:53,447 [INFO] __main__:   telecom × tcn × seed=44: accuracy=0.9987, f1=0.9987
2026-04-29 16:44:53,448 [INFO] __main__:   telecom × tcn: mean_acc=0.9990 ± 0.0005 (3 seeds)
2026-04-29 16:44:53,448 [INFO] __main__: ═══ TELECOM × PTv3 (Geometric) ═══
2026-04-29 16:44:53,450 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:44:56,211 [INFO] synapse.dataset.adapters.persistence: Loaded modular prepared bundle from: T256_D16_S42_R80_10
2026-04-29 16:44:56,212 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
2026-04-29 16:44:56,233 [INFO] synapse.baselines.src.engine.train: Training backbone=ptv3, params=219651
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:44:56,290 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ Z3UnifiedModel │  219 K │ train │     0 │
└───┴───────┴────────────────┴────────┴───────┴───────┘
Trainable params: 219 K                                                         
Non-trainable params: 0                                                         
Total params: 219 K                                                             
Total estimated model params size (MB): 0                                       
Modules in train mode: 97                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Epoch 0/49:  79% 317/400 [03:34<00:56,  1.48batch/s]