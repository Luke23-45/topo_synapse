2026-05-01 17:09:58,760 [INFO] __main__: Loaded config: synapse/baselines/configs/experiment/full.yaml
2026-05-01 17:09:58,794 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-05-01 17:09:58,794 [INFO] __main__: Z3 Baseline Study: 1 datasets × 1 backbones × 3 seeds
2026-05-01 17:09:58,795 [INFO] __main__:   Datasets: ['photonic']
2026-05-01 17:09:58,795 [INFO] __main__:   Backbones: ['deep_hodge']
2026-05-01 17:09:58,795 [INFO] __main__:   Seeds: 3 (base=42)
2026-05-01 17:09:58,795 [INFO] __main__:   Device: cuda
2026-05-01 17:09:58,795 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-05-01 17:09:58,795 [INFO] __main__: 
▓▓▓ DATASET: PHOTONIC ▓▓▓

2026-05-01 17:09:58,795 [INFO] __main__: ═══ PHOTONIC × Deep Hodge (Proposed) ═══
2026-05-01 17:09:58,797 [INFO] synapse.arch.data.data: Building dataloaders for dataset: photonic
2026-05-01 17:09:58,798 [INFO] synapse.dataset.adapters.photonic_adapter: No prepared cache found for PhotonicTopology. Starting extraction...
2026-05-01 17:09:58,844 [INFO] synapse.dataset.adapters.photonic_adapter: Loading 2D photonic topology from HuggingFace: cgeorgiaw/2d-photonic-topology
2026-05-01 17:09:59,057 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/revision/main "HTTP/1.1 200 OK"
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
Fetching 11 files:   0% 0/11 [00:00<?, ?it/s]2026-05-01 17:09:59,153 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup10.jld2 "HTTP/1.1 302 Found"
2026-05-01 17:09:59,193 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup17.jld2 "HTTP/1.1 302 Found"
2026-05-01 17:09:59,194 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup14.jld2 "HTTP/1.1 302 Found"
2026-05-01 17:09:59,195 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup13.jld2 "HTTP/1.1 302 Found"
2026-05-01 17:09:59,196 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup11.jld2 "HTTP/1.1 302 Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-05-01 17:09:59,199 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup16.jld2 "HTTP/1.1 302 Found"
2026-05-01 17:09:59,198 [WARNING] huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-05-01 17:09:59,289 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup15.jld2 "HTTP/1.1 302 Found"
2026-05-01 17:09:59,292 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/42.7M [00:00<?, ?B/s]2026-05-01 17:09:59,295 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/85.4M [00:00<?, ?B/s]2026-05-01 17:09:59,296 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/115M [00:00<?, ?B/s] 2026-05-01 17:09:59,297 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/158M [00:00<?, ?B/s]2026-05-01 17:09:59,298 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
2026-05-01 17:09:59,299 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/231M [00:00<?, ?B/s]2026-05-01 17:09:59,377 [INFO] httpx: HTTP Request: GET https://huggingface.co/api/datasets/cgeorgiaw/2d-photonic-topology/xet-read-token/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad "HTTP/1.1 200 OK"
Downloading (incomplete total...):   0% 0.00/273M [00:01<?, ?B/s]2026-05-01 17:10:00,191 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup12.jld2 "HTTP/1.1 302 Found"
Downloading (incomplete total...):  38% 115M/301M [00:01<00:01, 102MB/s]2026-05-01 17:10:00,246 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup2.jld2 "HTTP/1.1 302 Found"
Downloading (incomplete total...):  35% 115M/331M [00:01<00:02, 97.2MB/s]2026-05-01 17:10:00,298 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup6.jld2 "HTTP/1.1 302 Found"
Downloading (incomplete total...):  32% 115M/361M [00:01<00:00, 578MB/s]2026-05-01 17:10:00,444 [INFO] httpx: HTTP Request: HEAD https://huggingface.co/datasets/cgeorgiaw/2d-photonic-topology/resolve/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad/lattices/lattices-planegroup9.jld2 "HTTP/1.1 302 Found"
Downloading (incomplete total...):  92% 348M/378M [00:07<00:00, 31.6MB/s]
Fetching 11 files: 100% 11/11 [00:07<00:00,  1.43it/s]
Download complete: 100% 378M/378M [00:07<00:00, 31.6MB/s]                2026-05-01 17:10:06,762 [INFO] synapse.dataset.adapters.photonic_adapter: Downloaded photonic lattices to: /root/.cache/huggingface/hub/datasets--cgeorgiaw--2d-photonic-topology/snapshots/8c828fec784dc2ed53d4c4121dd8978c61dfe8ad

Reading photonic JLD2:   0% 0/11 [00:00<?, ?it/s]
Reading photonic JLD2:   9% 1/11 [00:00<00:02,  3.34it/s]
Reading photonic JLD2:  18% 2/11 [00:00<00:02,  3.32it/s]
Reading photonic JLD2:  27% 3/11 [00:00<00:02,  3.27it/s]
Reading photonic JLD2:  36% 4/11 [00:01<00:02,  3.30it/s]
Reading photonic JLD2:  45% 5/11 [00:01<00:01,  3.30it/s]
Reading photonic JLD2:  55% 6/11 [00:01<00:01,  3.29it/s]
Reading photonic JLD2:  64% 7/11 [00:02<00:01,  3.21it/s]
Reading photonic JLD2:  73% 8/11 [00:02<00:01,  2.92it/s]
Reading photonic JLD2:  82% 9/11 [00:03<00:00,  2.61it/s]
Reading photonic JLD2:  91% 10/11 [00:03<00:00,  2.44it/s]
Reading photonic JLD2: 100% 11/11 [00:03<00:00,  2.78it/s]
2026-05-01 17:10:10,726 [INFO] synapse.dataset.adapters.photonic_adapter: Read 110000 photonic records from 11 JLD2 files

Extracting PhotonicTopology:   0% 0/110000 [00:00<?, ?it/s]
Extracting PhotonicTopology: 100% 110000/110000 [00:00<00:00, 784799.53it/s]
2026-05-01 17:10:11,317 [INFO] synapse.dataset.adapters.photonic_adapter: PhotonicAdapter: loaded 110000 grids, shape=(110000, 8, 8, 8), classes=4
Download complete: 100% 378M/378M [00:17<00:00, 31.6MB/s]2026-05-01 17:10:18,255 [INFO] synapse.dataset.adapters.persistence: Saved modular prepared bundle to: data/datasets/photonic/prepared/T64_D10_C4_S42_R80_10
2026-05-01 17:10:18,303 [INFO] synapse.arch.data.data: Dataset 'photonic': train=88000, val=11000, test=11000, input_dim=10, seq_len=64
2026-05-01 17:10:33,524 [INFO] synapse.baselines.src.engine.train: Training backbone=deep_hodge, params=240768
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
2026-05-01 17:10:33,575 [INFO] synapse.baselines.src.engine.train: Starting Lightning training: epochs=50, accelerator=gpu, precision=32
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/usr/local/lib/python3.12/dist-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type         ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ UnifiedModel │  240 K │ train │     0 │
└───┴───────┴──────────────┴────────┴───────┴───────┘
Trainable params: 240 K                                                         
Non-trainable params: 0                                                         
Total params: 240 K                                                             
Total estimated model params size (MB): 0                                       
Modules in train mode: 59                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
2026-05-01 17:10:36,445 [ERROR] __main__:   photonic × deep_hodge × seed=42 FAILED: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [2048, 3]], which is output 0 of AsStridedBackward0, is at version 19; expected version 18 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
Traceback (most recent call last):
  File "/content/topo_synapse/synapse/baselines/run_experiment.py", line 152, in _run_single
    train_state = train_backbone(
                  ^^^^^^^^^^^^^^^
  File "/content/topo_synapse/synapse/baselines/src/engine/train.py", line 211, in train_backbone
    trainer.fit(
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/call.py", line 49, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
    self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 1079, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/trainer.py", line 1123, in _run_stage
    self.fit_loop.run()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py", line 217, in run
    self.advance()
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/fit_loop.py", line 465, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 153, in run
    self.advance(data_fetcher)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 352, in advance
    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 192, in run
    self._optimizer_step(batch_idx, closure)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 270, in _optimizer_step
    call._call_lightning_module_hook(
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/call.py", line 177, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/core/module.py", line 1368, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/core/optimizer.py", line 154, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/strategies/strategy.py", line 239, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/plugins/precision/precision.py", line 123, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/lr_scheduler.py", line 166, in wrapper
    return func.__get__(opt, opt.__class__)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/optimizer.py", line 526, in wrapper
    out = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/optimizer.py", line 81, in _use_grad
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/optim/adam.py", line 227, in step
    loss = closure()
           ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/plugins/precision/precision.py", line 109, in _wrap_closure
    closure_result = closure()
                     ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 146, in __call__
    self._result = self.closure(*args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 140, in closure
    self._backward_fn(step_output.closure_loss)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 241, in backward_fn
    call._call_strategy_hook(self.trainer, "backward", loss, optimizer)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/trainer/call.py", line 329, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/strategies/strategy.py", line 213, in backward
    self.precision_plugin.backward(closure_loss, self.lightning_module, optimizer, *args, **kwargs)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/plugins/precision/precision.py", line 73, in backward
    model.backward(tensor, *args, **kwargs)
  File "/usr/local/lib/python3.12/dist-packages/pytorch_lightning/core/module.py", line 1137, in backward
    loss.backward(*args, **kwargs)
  File "/usr/local/lib/python3.12/dist-packages/torch/_tensor.py", line 630, in backward
    torch.autograd.backward(
  File "/usr/local/lib/python3.12/dist-packages/torch/autograd/__init__.py", line 364, in backward
    _engine_run_backward(
  File "/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py", line 865, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [2048, 3]], which is output 0 of AsStridedBackward0, is at version 19; expected version 18 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
2026-05-01 17:10:36,464 [INFO] __main__: Dataset 'photonic' complete in 37.7s (0 backbones)
2026-05-01 17:10:36,464 [INFO] __main__: ═══════════════════════════════════════════════════════════
2026-05-01 17:10:36,464 [INFO] __main__: EXPERIMENT COMPLETE: 0 datasets, 37.7s total
2026-05-01 17:10:36,464 [INFO] __main__: Output directory: outputs/baselines/20260501_170958_z3_baseline_full
2026-05-01 17:10:36,465 [INFO] __main__: ═══════════════════════════════════════════════════════════
Download complete: 100% 378M/378M [00:37<00:00, 10.0MB/s]