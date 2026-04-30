2026-04-30 07:37:14,841 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-04-30 07:37:14,876 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 07:37:14,876 [INFO] __main__: Z3 Baseline Study: 1 datasets × 1 backbones × 3 seeds
2026-04-30 07:37:14,876 [INFO] __main__:   Datasets: ['photonic']
2026-04-30 07:37:14,877 [INFO] __main__:   Backbones: ['ptv3']
2026-04-30 07:37:14,877 [INFO] __main__:   Seeds: 3 (base=42)
2026-04-30 07:37:14,877 [INFO] __main__:   Device: cuda
2026-04-30 07:37:14,877 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 07:37:14,877 [INFO] __main__: 
▓▓▓ DATASET: PHOTONIC ▓▓▓

2026-04-30 07:37:14,877 [INFO] __main__: ═══ PHOTONIC × PTv3 (Geometric) ═══
2026-04-30 07:37:14,881 [INFO] synapse.arch.data.data: Building dataloaders for dataset: photonic
2026-04-30 07:37:14,881 [INFO] synapse.dataset.adapters.photonic_adapter: No prepared cache found for PhotonicTopology. Starting extraction...
2026-04-30 07:37:15,192 [INFO] synapse.dataset.adapters.photonic_adapter: Loading 2D photonic topology from HuggingFace: cgeorgiaw/2d-photonic-topology
2026-04-30 07:37:15,428 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/revision/main "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 11 files:   0% 0/11 [00:00<?, ?it/s]2026-04-30 07:37:15,527 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup10.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:15,555 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup11.jld2 "HTTP/1.1 302 Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-30 07:37:15,566 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup17.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:15,567 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup13.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:15,572 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup12.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:15,572 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup16.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:15,565 [WARNING] huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-30 07:37:15,576 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup14.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:15,599 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup15.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:15,676 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/42.7M [00:00<?, ?B/s]2026-04-30 07:37:15,680 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/70.1M [00:00<?, ?B/s]2026-04-30 07:37:15,681 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/100M [00:00<?, ?B/s] 2026-04-30 07:37:15,681 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/143M [00:00<?, ?B/s]2026-04-30 07:37:15,683 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/185M [00:00<?, ?B/s]2026-04-30 07:37:15,684 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/228M [00:00<?, ?B/s]2026-04-30 07:37:15,685 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/258M [00:00<?, ?B/s]2026-04-30 07:37:15,688 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):  38% 115M/301M [00:01<00:00, 329MB/s]
Fetching 11 files:   9% 1/11 [00:01<00:13,  1.31s/it]2026-04-30 07:37:16,807 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup2.jld2 "HTTP/1.1 302 Found"
Downloading (incomplete total...):  35% 115M/331M [00:01<00:00, 329MB/s]2026-04-30 07:37:16,851 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup6.jld2 "HTTP/1.1 302 Found"
2026-04-30 07:37:16,852 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup9.jld2 "HTTP/1.1 302 Found"
Downloading (incomplete total...):  50% 188M/378M [00:01<00:00, 199MB/s]
Downloading (incomplete total...):  76% 288M/378M [00:02<00:00, 271MB/s]
Downloading (incomplete total...):  92% 348M/378M [00:02<00:00, 324MB/s]
Fetching 11 files: 100% 11/11 [00:02<00:00,  3.96it/s]
Download complete: 100% 378M/378M [00:02<00:00, 324MB/s]                2026-04-30 07:37:18,224 [INFO] synapse.dataset.adapters.photonic_adapter: Downloaded photonic lattices to: /root/.cache/huggingface/hub/datasets--cgeorgiaw--2d-photonic-topology/snapshots/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad

Reading photonic JLD2:   0% 0/11 [00:00<?, ?it/s]
Reading photonic JLD2:   9% 1/11 [00:00<00:03,  3.26it/s]
Reading photonic JLD2:  18% 2/11 [00:00<00:02,  3.25it/s]
Reading photonic JLD2:  27% 3/11 [00:00<00:02,  3.13it/s]
Reading photonic JLD2:  36% 4/11 [00:01<00:02,  3.15it/s]
Reading photonic JLD2:  45% 5/11 [00:01<00:01,  3.14it/s]
Reading photonic JLD2:  55% 6/11 [00:01<00:01,  3.00it/s]
Reading photonic JLD2:  64% 7/11 [00:02<00:01,  3.05it/s]
Reading photonic JLD2:  73% 8/11 [00:02<00:00,  3.07it/s]
Reading photonic JLD2:  82% 9/11 [00:02<00:00,  2.97it/s]
Reading photonic JLD2:  91% 10/11 [00:03<00:00,  3.03it/s]
Reading photonic JLD2: 100% 11/11 [00:03<00:00,  3.07it/s]
2026-04-30 07:37:21,805 [INFO] synapse.dataset.adapters.photonic_adapter: Read 110000 photonic records from 11 JLD2 files

Extracting PhotonicTopology: 100% 110000/110000 [00:00<00:00, 1480863.66it/s]
2026-04-30 07:37:22,211 [INFO] synapse.dataset.adapters.photonic_adapter: PhotonicAdapter: loaded 110000 grids, shape=(110000, 8, 8, 8), classes=4
Download complete: 100% 378M/378M [00:12<00:00, 324MB/s]2026-04-30 07:37:29,699 [INFO] synapse.dataset.adapters.persistence: Saved modular prepared bundle to: data/datasets/photonic/prepared/T64_D10_C4_S42_R80_10
2026-04-30 07:37:29,739 [INFO] synapse.arch.data.data: Dataset 'photonic': train=88000, val=11000, test=11000, input_dim=10, seq_len=64
2026-04-30 07:37:29,818 [INFO] synapse.baselines.src.engine.train: Training backbone=ptv3, params=219332
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-04-30 07:37:29,873 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
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
Epoch 0 | val_loss=0.8999 | val_acc=0.5757
Metric val/loss improved. New best score: 0.900
Epoch 1 | train_loss=0.9775 | val_loss=0.8450 | val_acc=0.6175
Metric val/loss improved by 0.055 >= min_delta = 0.0. New best score: 0.845
Epoch 2 | train_loss=0.8613 | val_loss=0.8404 | val_acc=0.6202
Metric val/loss improved by 0.005 >= min_delta = 0.0. New best score: 0.840
Epoch 3 | train_loss=0.8498 | val_loss=0.8365 | val_acc=0.6234
Metric val/loss improved by 0.004 >= min_delta = 0.0. New best score: 0.837
Epoch 4 | train_loss=0.8434 | val_loss=0.8348 | val_acc=0.6234
Metric val/loss improved by 0.002 >= min_delta = 0.0. New best score: 0.835
Epoch 5 | train_loss=0.8390 | val_loss=0.8321 | val_acc=0.6225
Metric val/loss improved by 0.003 >= min_delta = 0.0. New best score: 0.832
Epoch 6 | train_loss=0.8344 | val_loss=0.8345 | val_acc=0.6233
Epoch 7 | train_loss=0.8291 | val_loss=0.8406 | val_acc=0.6218
Epoch 8 | train_loss=0.8262 | val_loss=0.8437 | val_acc=0.6207
Epoch 9 | train_loss=0.8229 | val_loss=0.8477 | val_acc=0.6208
Epoch 10 | train_loss=0.8213 | val_loss=0.8561 | val_acc=0.6204
Epoch 11 | train_loss=0.8186 | val_loss=0.8541 | val_acc=0.6201
Epoch 12 | train_loss=0.8174 | val_loss=0.8583 | val_acc=0.6179
Epoch 13 | train_loss=0.8163 | val_loss=0.8587 | val_acc=0.6195
Epoch 14 | train_loss=0.8145 | val_loss=0.8657 | val_acc=0.6181
Epoch 15 | train_loss=0.8129 | val_loss=0.8580 | val_acc=0.6175
Monitored metric val/loss did not improve in the last 10 records. Best score: 0.832. Signaling Trainer to stop.
2026-04-30 08:02:23,232 [INFO] synapse.baselines.src.engine.train: Loaded evaluation checkpoint from /content/topo_synapse/outputs/baselines/20260430_073714_z3_baseline_full/dataset_photonic/backbone_ptv3/seed_42/checkpoints/best-epoch=005-val/loss=0.8321.ckpt
2026-04-30 08:02:27,770 [INFO] synapse.baselines.src.engine.rollout: Rollout evaluation: 50 samples, 10 steps, noise_scale=0.100
2026-04-30 08:02:27,771 [INFO] __main__:   photonic × ptv3 × seed=42: rollout_auc=0.4830, degradation_slope=-0.2073
2026-04-30 08:02:27,771 [INFO] __main__:   photonic × ptv3 × seed=42: accuracy=0.6166, f1=0.5112
2026-04-30 08:02:27,771 [INFO] __main__:   photonic × ptv3: mean_acc=0.6166 ± 0.0000 (1 seeds)
2026-04-30 08:02:27,771 [INFO] __main__: Dataset 'photonic' complete in 1512.9s (1 backbones)
2026-04-30 08:02:27,772 [INFO] synapse.baselines.src.reporting.report: JSON report saved to outputs/baselines/20260430_073714_z3_baseline_full/cross_backbone/photonic_results.json
2026-04-30 08:02:27,772 [INFO] synapse.baselines.src.reporting.report: Markdown report saved to outputs/baselines/20260430_073714_z3_baseline_full/cross_backbone/photonic_summary.md
2026-04-30 08:02:28,503 [INFO] synapse.baselines.src.reporting.visualize: Saved accuracy plot to outputs/baselines/20260430_073714_z3_baseline_full/cross_backbone/photonic_accuracy.pdf
2026-04-30 08:02:28,649 [INFO] synapse.baselines.src.reporting.visualize: Saved learning curves to outputs/baselines/20260430_073714_z3_baseline_full/cross_backbone/photonic_learning.pdf
2026-04-30 08:02:28,768 [INFO] __main__: Rollout report saved for photonic
2026-04-30 08:02:28,768 [INFO] __main__: Cross-backbone reports saved to outputs/baselines/20260430_073714_z3_baseline_full/cross_backbone
2026-04-30 08:02:28,768 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-04-30 08:02:28,768 [INFO] __main__: EXPERIMENT COMPLETE: 1 datasets, 1512.9s total
2026-04-30 08:02:28,769 [INFO] __main__: Output directory: outputs/baselines/20260430_073714_z3_baseline_full
2026-04-30 08:02:28,769 [INFO] __main__: ═══════════════════════════════════════════════════════════
Download complete: 100% 378M/378M [25:13<00:00, 250kB/s]