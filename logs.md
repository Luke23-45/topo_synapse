2026-04-29 15:53:20,768 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-04-29 15:53:20,794 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-29 15:53:20,794 [INFO] __main__: Z3 Baseline Study: 1 datasets × 1 backbones × 3 seeds
2026-04-29 15:53:20,794 [INFO] __main__:   Datasets: ['telecom']
2026-04-29 15:53:20,794 [INFO] __main__:   Backbones: ['mlp']
2026-04-29 15:53:20,794 [INFO] __main__:   Seeds: 3 (base=42)
2026-04-29 15:53:20,794 [INFO] __main__:   Device: cuda
2026-04-29 15:53:20,794 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-29 15:53:20,794 [INFO] __main__: 
▓▓▓ DATASET: TELECOM ▓▓▓

2026-04-29 15:53:20,794 [INFO] __main__: ═══ TELECOM × MLP (Sanity Check) ═══
2026-04-29 15:53:20,796 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 15:53:20,796 [WARNING] synapse.dataset.adapters.persistence: Failed to load modular bundle from data/datasets/telecom/prepared/T256_D16_S42_R80_10: Expecting value: line 11 column 27 (char 212)
2026-04-29 15:53:20,796 [INFO] synapse.dataset.adapters.telecom_adapter: No prepared cache found for TelecomTS. Starting extraction...
2026-04-29 15:53:20,807 [INFO] datasets: TensorFlow version 2.20.0 available.
2026-04-29 15:53:20,808 [INFO] datasets: JAX version 0.7.2 available.
2026-04-29 15:53:21,005 [INFO] synapse.dataset.adapters.telecom_adapter: Loading TelecomTS from HuggingFace: AliMaatouk/TelecomTS
2026-04-29 15:53:21,158 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-29 15:53:21,159 [WARNING] huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-29 15:53:21,169 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/AliMaatouk/TelecomTS/01e44b1b75e9b229c71a801ccd55320c28669e7f/README.md "HTTP/1.1 200 OK"
2026-04-29 15:53:21,252 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 15:53:21,469 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/AliMaatouk/TelecomTS/AliMaatouk/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 15:53:21,556 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/revision/01e44b1b75e9b229c71a801ccd55320c28669e7f "HTTP/1.1 200 OK"
2026-04-29 15:53:21,642 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-29 15:53:21,775 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=AliMaatouk/TelecomTS "HTTP/1.1 200 OK"
2026-04-29 15:53:21,866 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f/data?recursive=true&expand=false "HTTP/1.1 404 Not Found"
2026-04-29 15:53:21,955 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f?recursive=false&expand=false "HTTP/1.1 200 OK"
2026-04-29 15:53:22,052 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f?recursive=true&expand=false "HTTP/1.1 200 OK"
Resolving data files: 100% 99/99 [00:00<00:00, 353693.44it/s]
2026-04-29 15:53:22,247 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/dataset_infos.json "HTTP/1.1 404 Not Found"
Extracting TelecomTS: 100% 32000/32000 [02:17<00:00, 232.11it/s]
2026-04-29 15:55:40,383 [INFO] synapse.dataset.adapters.telecom_adapter: TelecomAdapter: loaded 32000 samples, shape=(32000, 128, 16), classes=3
2026-04-29 15:55:57,030 [WARNING] synapse.dataset.adapters.persistence: Failed to save modular prepared bundle: Object of type ndarray is not JSON serializable
2026-04-29 15:55:57,033 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
2026-04-29 15:55:57,058 [INFO] synapse.baselines.src.engine.train: Training backbone=mlp, params=4278723
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 15:55:57,101 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
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
Epoch 0/49:  99% 397/400 [00:05<00:00, 105.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 209.44batch/s]
  Validating:  86% 43/50 [00:00<00:00, 210.74batch/s]
Epoch 0  │ val_loss=0.3036  │ val_acc=0.9578
Metric val/loss improved. New best score: 0.304
Epoch 1/49:  98% 390/400 [00:03<00:00, 102.83batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 186.07batch/s]
  Validating:  78% 39/50 [00:00<00:00, 189.20batch/s]
Epoch 1  │ train_loss=0.1590  │ val_loss=0.1575  │ val_acc=0.9694
Metric val/loss improved by 0.146 >= min_delta = 0.0. New best score: 0.157
Epoch 2/49: 100% 398/400 [00:04<00:00, 75.50batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  24% 12/50 [00:00<00:00, 116.82batch/s]
  Validating:  50% 25/50 [00:00<00:00, 122.16batch/s]
  Validating:  76% 38/50 [00:00<00:00, 120.28batch/s]
Epoch 2  │ train_loss=0.0769  │ val_loss=0.0866  │ val_acc=0.9806
Metric val/loss improved by 0.071 >= min_delta = 0.0. New best score: 0.087
Epoch 3/49: 100% 399/400 [00:04<00:00, 99.74batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 173.57batch/s]
  Validating:  74% 37/50 [00:00<00:00, 182.68batch/s]
Epoch 3  │ train_loss=0.0560  │ val_loss=0.0570  │ val_acc=0.9844
Metric val/loss improved by 0.030 >= min_delta = 0.0. New best score: 0.057
Epoch 4/49: 100% 400/400 [00:03<00:00, 104.83batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 204.49batch/s]
  Validating:  84% 42/50 [00:00<00:00, 200.57batch/s]
Epoch 4  │ train_loss=0.0477  │ val_loss=0.0422  │ val_acc=0.9881
Metric val/loss improved by 0.015 >= min_delta = 0.0. New best score: 0.042
Epoch 5/49:  98% 394/400 [00:04<00:00, 104.93batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 186.69batch/s]
  Validating:  76% 38/50 [00:00<00:00, 171.12batch/s]
Epoch 5  │ train_loss=0.0369  │ val_loss=0.0359  │ val_acc=0.9897
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.036
Epoch 6/49:  98% 390/400 [00:03<00:00, 100.03batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 187.01batch/s]
  Validating:  78% 39/50 [00:00<00:00, 192.53batch/s]
Epoch 6  │ train_loss=0.0345  │ val_loss=0.0302  │ val_acc=0.9912
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.030
Epoch 7/49:  99% 395/400 [00:04<00:00, 78.78batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 142.06batch/s]
  Validating:  60% 30/50 [00:00<00:00, 142.82batch/s]
  Validating:  92% 46/50 [00:00<00:00, 145.88batch/s]
Epoch 7  │ train_loss=0.0291  │ val_loss=0.0278  │ val_acc=0.9906
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.028
Epoch 8/49: 100% 400/400 [00:04<00:00, 97.89batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  32% 16/50 [00:00<00:00, 157.38batch/s]
  Validating:  70% 35/50 [00:00<00:00, 175.03batch/s]
Epoch 8  │ train_loss=0.0266  │ val_loss=0.0268  │ val_acc=0.9906
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.027
Epoch 9/49:  98% 390/400 [00:03<00:00, 98.69batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 203.42batch/s]
  Validating:  84% 42/50 [00:00<00:00, 202.90batch/s]
Epoch 9  │ train_loss=0.0223  │ val_loss=0.0257  │ val_acc=0.9922
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.026
Epoch 10/49:  98% 393/400 [00:04<00:00, 93.37batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 190.47batch/s]
  Validating:  82% 41/50 [00:00<00:00, 197.10batch/s]
Epoch 10  │ train_loss=0.0219  │ val_loss=0.0277  │ val_acc=0.9925
Epoch 11/49:  99% 395/400 [00:03<00:00, 101.94batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 174.70batch/s]
  Validating:  76% 38/50 [00:00<00:00, 188.74batch/s]
Epoch 11  │ train_loss=0.0189  │ val_loss=0.0308  │ val_acc=0.9925
Epoch 12/49:  99% 395/400 [00:03<00:00, 103.32batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 199.23batch/s]
  Validating:  82% 41/50 [00:00<00:00, 203.24batch/s]
Epoch 12  │ train_loss=0.0202  │ val_loss=0.0333  │ val_acc=0.9925
Epoch 13/49: 100% 398/400 [00:04<00:00, 100.45batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 192.31batch/s]
  Validating:  80% 40/50 [00:00<00:00, 192.24batch/s]
Epoch 13  │ train_loss=0.0209  │ val_loss=0.0347  │ val_acc=0.9919
Epoch 14/49:  99% 397/400 [00:03<00:00, 102.25batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 194.35batch/s]
  Validating:  80% 40/50 [00:00<00:00, 192.53batch/s]
Epoch 14  │ train_loss=0.0127  │ val_loss=0.0358  │ val_acc=0.9919
Epoch 15/49:  98% 394/400 [00:03<00:00, 102.07batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 191.47batch/s]
  Validating:  80% 40/50 [00:00<00:00, 194.83batch/s]
Epoch 15  │ train_loss=0.0174  │ val_loss=0.0355  │ val_acc=0.9925
Epoch 16/49:  98% 391/400 [00:04<00:00, 99.75batch/s] 
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 208.30batch/s]
  Validating:  86% 43/50 [00:00<00:00, 205.44batch/s]
Epoch 16  │ train_loss=0.0159  │ val_loss=0.0357  │ val_acc=0.9922
Epoch 17/49:  99% 395/400 [00:03<00:00, 101.22batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  44% 22/50 [00:00<00:00, 211.20batch/s]
  Validating:  88% 44/50 [00:00<00:00, 213.36batch/s]
Epoch 17  │ train_loss=0.0140  │ val_loss=0.0372  │ val_acc=0.9919
Epoch 18/49:  99% 395/400 [00:03<00:00, 100.41batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 191.34batch/s]
  Validating:  82% 41/50 [00:00<00:00, 201.15batch/s]
Epoch 18  │ train_loss=0.0116  │ val_loss=0.0377  │ val_acc=0.9922
Epoch 19/49:  98% 392/400 [00:04<00:00, 98.92batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 186.32batch/s]
  Validating:  80% 40/50 [00:00<00:00, 195.16batch/s]
Epoch 19  │ train_loss=0.0135  │ val_loss=0.0378  │ val_acc=0.9931
Monitored metric val/loss did not improve in the last 10 records. Best score: 0.026. Signaling Trainer to stop.
2026-04-29 15:57:33,440 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_155320_z3_baseline_full/dataset_telecom/backbone_mlp/seed_42/checkpoints/last.ckpt
2026-04-29 15:57:34,033 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 15:57:34,034 [INFO] __main__:   telecom × mlp × seed=42: rollout_auc=9.7400, degradation_slope=0.0031
2026-04-29 15:57:34,034 [INFO] __main__:   telecom × mlp × seed=42: accuracy=0.9900, f1=0.9900
2026-04-29 15:57:34,036 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 15:57:34,036 [WARNING] synapse.dataset.adapters.persistence: Failed to load modular bundle from data/datasets/telecom/prepared/T256_D16_S43_R80_10: Expecting value: line 11 column 27 (char 212)
2026-04-29 15:57:34,037 [INFO] synapse.dataset.adapters.telecom_adapter: No prepared cache found for TelecomTS. Starting extraction...
2026-04-29 15:57:34,037 [INFO] synapse.dataset.adapters.telecom_adapter: Loading TelecomTS from HuggingFace: AliMaatouk/TelecomTS
2026-04-29 15:57:34,147 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-04-29 15:57:34,158 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/AliMaatouk/TelecomTS/01e44b1b75e9b229c71a801ccd55320c28669e7f/README.md "HTTP/1.1 200 OK"
2026-04-29 15:57:34,246 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 15:57:34,458 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/AliMaatouk/TelecomTS/AliMaatouk/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 15:57:34,545 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-29 15:57:34,693 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=AliMaatouk/TelecomTS "HTTP/1.1 200 OK"
2026-04-29 15:57:34,782 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f/data?recursive=true&expand=false "HTTP/1.1 404 Not Found"
Resolving data files: 100% 99/99 [00:00<00:00, 291598.38it/s]
2026-04-29 15:57:34,960 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/dataset_infos.json "HTTP/1.1 404 Not Found"
Extracting TelecomTS: 100% 32000/32000 [02:17<00:00, 232.82it/s]
2026-04-29 15:59:52,665 [INFO] synapse.dataset.adapters.telecom_adapter: TelecomAdapter: loaded 32000 samples, shape=(32000, 128, 16), classes=3
2026-04-29 16:00:11,050 [WARNING] synapse.dataset.adapters.persistence: Failed to save modular prepared bundle: Object of type ndarray is not JSON serializable
2026-04-29 16:00:11,054 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
2026-04-29 16:00:11,080 [INFO] synapse.baselines.src.engine.train: Training backbone=mlp, params=4278723
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:00:11,123 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
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
Epoch 0/49:  99% 395/400 [00:04<00:00, 83.21batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 143.44batch/s]
  Validating:  60% 30/50 [00:00<00:00, 139.15batch/s]
  Validating:  90% 45/50 [00:00<00:00, 140.11batch/s]
Epoch 0  │ val_loss=0.3098  │ val_acc=0.9572
Metric val/loss improved. New best score: 0.310
Epoch 1/49:  98% 394/400 [00:04<00:00, 102.54batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 204.36batch/s]
  Validating:  84% 42/50 [00:00<00:00, 202.53batch/s]
Epoch 1  │ train_loss=0.1604  │ val_loss=0.1375  │ val_acc=0.9822
Metric val/loss improved by 0.172 >= min_delta = 0.0. New best score: 0.138
Epoch 2/49:  99% 397/400 [00:04<00:00, 99.03batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 177.13batch/s]
  Validating:  74% 37/50 [00:00<00:00, 177.33batch/s]
Epoch 2  │ train_loss=0.0730  │ val_loss=0.0746  │ val_acc=0.9862
Metric val/loss improved by 0.063 >= min_delta = 0.0. New best score: 0.075
Epoch 3/49:  98% 392/400 [00:04<00:00, 85.56batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 148.72batch/s]
  Validating:  68% 34/50 [00:00<00:00, 167.83batch/s]
Epoch 3  │ train_loss=0.0524  │ val_loss=0.0550  │ val_acc=0.9881
Metric val/loss improved by 0.020 >= min_delta = 0.0. New best score: 0.055
Epoch 4/49:  98% 394/400 [00:04<00:00, 100.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  34% 17/50 [00:00<00:00, 166.96batch/s]
  Validating:  72% 36/50 [00:00<00:00, 177.19batch/s]
Epoch 4  │ train_loss=0.0496  │ val_loss=0.0473  │ val_acc=0.9903
Metric val/loss improved by 0.008 >= min_delta = 0.0. New best score: 0.047
Epoch 5/49:  99% 397/400 [00:04<00:00, 98.05batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 108.37batch/s]
  Validating:  46% 23/50 [00:00<00:00, 114.59batch/s]
  Validating:  72% 36/50 [00:00<00:00, 118.71batch/s]
  Validating:  98% 49/50 [00:00<00:00, 120.49batch/s]
Epoch 5  │ train_loss=0.0421  │ val_loss=0.0409  │ val_acc=0.9903
Metric val/loss improved by 0.006 >= min_delta = 0.0. New best score: 0.041
Epoch 6/49:  98% 394/400 [00:04<00:00, 100.01batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 177.51batch/s]
  Validating:  76% 38/50 [00:00<00:00, 185.96batch/s]
Epoch 6  │ train_loss=0.0309  │ val_loss=0.0383  │ val_acc=0.9897
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.038
Epoch 7/49:  99% 395/400 [00:03<00:00, 96.97batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 173.03batch/s]
  Validating:  72% 36/50 [00:00<00:00, 168.07batch/s]
Epoch 7  │ train_loss=0.0328  │ val_loss=0.0361  │ val_acc=0.9894
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.036
Epoch 8/49:  99% 395/400 [00:04<00:00, 70.84batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 186.12batch/s]
  Validating:  76% 38/50 [00:00<00:00, 176.96batch/s]
Epoch 8  │ train_loss=0.0267  │ val_loss=0.0350  │ val_acc=0.9906
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.035
Epoch 9/49:  99% 395/400 [00:03<00:00, 108.18batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 205.04batch/s]
  Validating:  86% 43/50 [00:00<00:00, 210.83batch/s]
Epoch 9  │ train_loss=0.0233  │ val_loss=0.0356  │ val_acc=0.9903
Epoch 10/49:  99% 397/400 [00:03<00:00, 101.31batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 193.04batch/s]
  Validating:  82% 41/50 [00:00<00:00, 201.28batch/s]
Epoch 10  │ train_loss=0.0175  │ val_loss=0.0351  │ val_acc=0.9912
Epoch 11/49: 100% 399/400 [00:04<00:00, 99.89batch/s] 
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 196.49batch/s]
  Validating:  80% 40/50 [00:00<00:00, 197.13batch/s]
Epoch 11  │ train_loss=0.0206  │ val_loss=0.0356  │ val_acc=0.9906
Epoch 12/49:  99% 396/400 [00:03<00:00, 103.58batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 185.31batch/s]
  Validating:  78% 39/50 [00:00<00:00, 189.70batch/s]
Epoch 12  │ train_loss=0.0214  │ val_loss=0.0360  │ val_acc=0.9912
Epoch 13/49: 100% 398/400 [00:03<00:00, 104.02batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 173.16batch/s]
  Validating:  78% 39/50 [00:00<00:00, 194.08batch/s]
Epoch 13  │ train_loss=0.0204  │ val_loss=0.0369  │ val_acc=0.9916
Epoch 14/49:  99% 395/400 [00:04<00:00, 101.25batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 195.11batch/s]
  Validating:  80% 40/50 [00:00<00:00, 189.42batch/s]
Epoch 14  │ train_loss=0.0152  │ val_loss=0.0375  │ val_acc=0.9916
Epoch 15/49: 100% 398/400 [00:03<00:00, 105.11batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 202.10batch/s]
  Validating:  86% 43/50 [00:00<00:00, 207.84batch/s]
Epoch 15  │ train_loss=0.0123  │ val_loss=0.0359  │ val_acc=0.9919
Epoch 16/49:  99% 396/400 [00:03<00:00, 101.42batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 205.77batch/s]
  Validating:  84% 42/50 [00:00<00:00, 204.57batch/s]
Epoch 16  │ train_loss=0.0137  │ val_loss=0.0364  │ val_acc=0.9909
Epoch 17/49: 100% 399/400 [00:04<00:00, 101.42batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 197.33batch/s]
  Validating:  80% 40/50 [00:00<00:00, 194.11batch/s]
Epoch 17  │ train_loss=0.0144  │ val_loss=0.0371  │ val_acc=0.9922
Epoch 18/49:  99% 396/400 [00:03<00:00, 101.27batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 195.57batch/s]
  Validating:  80% 40/50 [00:00<00:00, 190.89batch/s]
Epoch 18  │ train_loss=0.0137  │ val_loss=0.0366  │ val_acc=0.9925
Monitored metric val/loss did not improve in the last 10 records. Best score: 0.035. Signaling Trainer to stop.
2026-04-29 16:01:40,326 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_155320_z3_baseline_full/dataset_telecom/backbone_mlp/seed_43/checkpoints/last.ckpt
2026-04-29 16:01:40,837 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:01:40,838 [INFO] __main__:   telecom × mlp × seed=43: rollout_auc=9.8000, degradation_slope=0.0000
2026-04-29 16:01:40,838 [INFO] __main__:   telecom × mlp × seed=43: accuracy=0.9891, f1=0.9890
2026-04-29 16:01:40,840 [INFO] synapse.arch.data.data: Building dataloaders for dataset: telecom
2026-04-29 16:01:40,840 [INFO] synapse.dataset.adapters.telecom_adapter: No prepared cache found for TelecomTS. Starting extraction...
2026-04-29 16:01:40,840 [INFO] synapse.dataset.adapters.telecom_adapter: Loading TelecomTS from HuggingFace: AliMaatouk/TelecomTS
2026-04-29 16:01:40,955 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-04-29 16:01:40,966 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/AliMaatouk/TelecomTS/01e44b1b75e9b229c71a801ccd55320c28669e7f/README.md "HTTP/1.1 200 OK"
2026-04-29 16:01:41,055 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 16:01:41,281 [INFO] httpx: HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/AliMaatouk/TelecomTS/AliMaatouk/TelecomTS.py "HTTP/1.1 404 Not Found"
2026-04-29 16:01:41,370 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/.huggingface.yaml "HTTP/1.1 404 Not Found"
2026-04-29 16:01:41,488 [INFO] httpx: HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=AliMaatouk/TelecomTS "HTTP/1.1 200 OK"
2026-04-29 16:01:41,582 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/AliMaatouk/TelecomTS/tree/01e44b1b75e9b229c71a801ccd55320c28669e7f/data?recursive=true&expand=false "HTTP/1.1 404 Not Found"
Resolving data files: 100% 99/99 [00:00<00:00, 335408.80it/s]
2026-04-29 16:01:41,741 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/01e44b1b75e9b229c71a801ccd55320c28669e7f/dataset_infos.json "HTTP/1.1 404 Not Found"
Extracting TelecomTS: 100% 32000/32000 [02:25<00:00, 220.27it/s]
2026-04-29 16:04:07,336 [INFO] synapse.dataset.adapters.telecom_adapter: TelecomAdapter: loaded 32000 samples, shape=(32000, 128, 16), classes=3
2026-04-29 16:04:25,116 [WARNING] synapse.dataset.adapters.persistence: Failed to save modular prepared bundle: Object of type ndarray is not JSON serializable
2026-04-29 16:04:25,119 [INFO] synapse.arch.data.data: Dataset 'telecom': train=25600, val=3200, test=3200, input_dim=16, seq_len=256
2026-04-29 16:04:25,155 [INFO] synapse.baselines.src.engine.train: Training backbone=mlp, params=4278723
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-29 16:04:25,200 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
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
Epoch 0/49:  98% 392/400 [00:03<00:00, 97.28batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 191.91batch/s]
  Validating:  80% 40/50 [00:00<00:00, 195.87batch/s]
Epoch 0  │ val_loss=0.2776  │ val_acc=0.9672
Metric val/loss improved. New best score: 0.278
Epoch 1/49:  99% 397/400 [00:05<00:00, 77.31batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 128.40batch/s]
  Validating:  56% 28/50 [00:00<00:00, 140.33batch/s]
  Validating:  86% 43/50 [00:00<00:00, 139.19batch/s]
Epoch 1  │ train_loss=0.1571  │ val_loss=0.1240  │ val_acc=0.9750
Metric val/loss improved by 0.154 >= min_delta = 0.0. New best score: 0.124
Epoch 2/49: 100% 398/400 [00:04<00:00, 85.53batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 176.26batch/s]
  Validating:  72% 36/50 [00:00<00:00, 173.56batch/s]
Epoch 2  │ train_loss=0.0773  │ val_loss=0.0762  │ val_acc=0.9753
Metric val/loss improved by 0.048 >= min_delta = 0.0. New best score: 0.076
Epoch 3/49: 100% 400/400 [00:05<00:00, 65.76batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  20% 10/50 [00:00<00:00, 95.74batch/s]
  Validating:  40% 20/50 [00:00<00:00, 95.96batch/s]
  Validating:  62% 31/50 [00:00<00:00, 100.24batch/s]
  Validating:  86% 43/50 [00:00<00:00, 105.88batch/s]
Epoch 3  │ train_loss=0.0582  │ val_loss=0.0617  │ val_acc=0.9759
Metric val/loss improved by 0.015 >= min_delta = 0.0. New best score: 0.062
Epoch 4/49:  99% 395/400 [00:04<00:00, 83.52batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  34% 17/50 [00:00<00:00, 161.92batch/s]
  Validating:  70% 35/50 [00:00<00:00, 169.81batch/s]
Epoch 4  │ train_loss=0.0442  │ val_loss=0.0527  │ val_acc=0.9791
Metric val/loss improved by 0.009 >= min_delta = 0.0. New best score: 0.053
Epoch 5/49:  98% 393/400 [00:04<00:00, 82.30batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  22% 11/50 [00:00<00:00, 102.79batch/s]
  Validating:  46% 23/50 [00:00<00:00, 107.38batch/s]
  Validating:  68% 34/50 [00:00<00:00, 105.43batch/s]
  Validating:  90% 45/50 [00:00<00:00, 103.04batch/s]
Epoch 5  │ train_loss=0.0337  │ val_loss=0.0485  │ val_acc=0.9825
Metric val/loss improved by 0.004 >= min_delta = 0.0. New best score: 0.048
Epoch 6/49:  98% 394/400 [00:05<00:00, 90.80batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 187.84batch/s]
  Validating:  78% 39/50 [00:00<00:00, 194.14batch/s]
Epoch 6  │ train_loss=0.0315  │ val_loss=0.0461  │ val_acc=0.9862
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.046
Epoch 7/49:  98% 393/400 [00:04<00:00, 97.80batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 183.46batch/s]
  Validating:  76% 38/50 [00:00<00:00, 159.36batch/s]
Epoch 7  │ train_loss=0.0264  │ val_loss=0.0449  │ val_acc=0.9872
Metric val/loss improved by 0.001 >= min_delta = 0.0. New best score: 0.045
Epoch 8/49:  98% 391/400 [00:04<00:00, 96.29batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  34% 17/50 [00:00<00:00, 164.91batch/s]
  Validating:  70% 35/50 [00:00<00:00, 172.65batch/s]
Epoch 8  │ train_loss=0.0283  │ val_loss=0.0423  │ val_acc=0.9881
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.042
Epoch 9/49:  99% 396/400 [00:04<00:00, 80.19batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  34% 17/50 [00:00<00:00, 161.42batch/s]
  Validating:  68% 34/50 [00:00<00:00, 159.16batch/s]
Epoch 9  │ train_loss=0.0235  │ val_loss=0.0432  │ val_acc=0.9884
Epoch 10/49: 100% 399/400 [00:04<00:00, 70.68batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  26% 13/50 [00:00<00:00, 121.88batch/s]
  Validating:  52% 26/50 [00:00<00:00, 123.59batch/s]
  Validating:  78% 39/50 [00:00<00:00, 123.77batch/s]
Epoch 10  │ train_loss=0.0230  │ val_loss=0.0448  │ val_acc=0.9897
Epoch 11/49:  98% 392/400 [00:04<00:00, 89.89batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 173.25batch/s]
  Validating:  72% 36/50 [00:00<00:00, 167.03batch/s]
Epoch 11  │ train_loss=0.0209  │ val_loss=0.0462  │ val_acc=0.9897
Epoch 12/49:  98% 392/400 [00:04<00:00, 100.82batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 181.81batch/s]
  Validating:  76% 38/50 [00:00<00:00, 185.49batch/s]
Epoch 12  │ train_loss=0.0170  │ val_loss=0.0465  │ val_acc=0.9887
Epoch 13/49: 100% 400/400 [00:04<00:00, 69.42batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  36% 18/50 [00:00<00:00, 176.14batch/s]
  Validating:  76% 38/50 [00:00<00:00, 184.82batch/s]
Epoch 13  │ train_loss=0.0172  │ val_loss=0.0490  │ val_acc=0.9881
Epoch 14/49:  99% 396/400 [00:04<00:00, 94.08batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  30% 15/50 [00:00<00:00, 144.77batch/s]
  Validating:  66% 33/50 [00:00<00:00, 161.77batch/s]
  Validating: 100% 50/50 [00:00<00:00, 160.03batch/s]
Epoch 14  │ train_loss=0.0163  │ val_loss=0.0508  │ val_acc=0.9894
Epoch 15/49: 100% 398/400 [00:04<00:00, 103.67batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 180.48batch/s]
  Validating:  80% 40/50 [00:00<00:00, 195.60batch/s]
Epoch 15  │ train_loss=0.0144  │ val_loss=0.0500  │ val_acc=0.9897
Epoch 16/49:  98% 394/400 [00:04<00:00, 98.15batch/s] 
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  40% 20/50 [00:00<00:00, 193.82batch/s]
  Validating:  80% 40/50 [00:00<00:00, 184.14batch/s]
Epoch 16  │ train_loss=0.0158  │ val_loss=0.0503  │ val_acc=0.9897
Epoch 17/49: 100% 399/400 [00:03<00:00, 103.16batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  42% 21/50 [00:00<00:00, 199.54batch/s]
  Validating:  82% 41/50 [00:00<00:00, 193.17batch/s]
Epoch 17  │ train_loss=0.0110  │ val_loss=0.0530  │ val_acc=0.9897
Epoch 18/49:  99% 396/400 [00:03<00:00, 100.88batch/s]
  Validating:   0% 0/50 [00:00<?, ?batch/s]
  Validating:  38% 19/50 [00:00<00:00, 186.94batch/s]
  Validating:  80% 40/50 [00:00<00:00, 193.12batch/s]
Epoch 18  │ train_loss=0.0134  │ val_loss=0.0523  │ val_acc=0.9900
Monitored metric val/loss did not improve in the last 10 records. Best score: 0.042. Signaling Trainer to stop.
2026-04-29 16:06:03,285 [INFO] synapse.baselines.src.engine.train: Loaded best checkpoint from outputs/baselines/20260429_155320_z3_baseline_full/dataset_telecom/backbone_mlp/seed_44/checkpoints/last.ckpt
2026-04-29 16:06:03,988 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-29 16:06:03,989 [INFO] __main__:   telecom × mlp × seed=44: rollout_auc=9.7000, degradation_slope=0.0018
2026-04-29 16:06:03,989 [INFO] __main__:   telecom × mlp × seed=44: accuracy=0.9897, f1=0.9897
2026-04-29 16:06:03,990 [INFO] __main__:   telecom × mlp: mean_acc=0.9896 ± 0.0004 (3 seeds)
2026-04-29 16:06:03,990 [INFO] __main__: Dataset 'telecom' complete in 763.2s (1 backbones)
2026-04-29 16:06:03,991 [INFO] synapse.baselines.src.reporting.report: JSON report saved to outputs/baselines/20260429_155320_z3_baseline_full/cross_backbone/telecom_results.json
2026-04-29 16:06:03,991 [INFO] synapse.baselines.src.reporting.report: Markdown report saved to outputs/baselines/20260429_155320_z3_baseline_full/cross_backbone/telecom_summary.md
2026-04-29 16:06:05,106 [INFO] synapse.baselines.src.reporting.visualize: Saved accuracy plot to outputs/baselines/20260429_155320_z3_baseline_full/cross_backbone/telecom_accuracy.pdf
2026-04-29 16:06:05,657 [INFO] synapse.baselines.src.reporting.visualize: Saved learning curves to outputs/baselines/20260429_155320_z3_baseline_full/cross_backbone/telecom_learning.pdf
2026-04-29 16:06:05,799 [INFO] __main__: Rollout report saved for telecom
2026-04-29 16:06:05,799 [INFO] __main__: Cross-backbone reports saved to outputs/baselines/20260429_155320_z3_baseline_full/cross_backbone
2026-04-29 16:06:05,800 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-29 16:06:05,800 [INFO] __main__: EXPERIMENT COMPLETE: 1 datasets, 763.2s total
2026-04-29 16:06:05,800 [INFO] __main__: Output directory: outputs/baselines/20260429_155320_z3_baseline_full
2026-04-29 16:06:05,800 [INFO] __main__: ═══════════════════════════════════════════════════════════