"""Z3 Baseline Study — Modular Baseline Comparison Framework.

Structure (mirrors legacy ``baselines/`` project):
    baselines/
    ├── base.py, registry.py          — shared interface + factory
    ├── configs/                      — per-model + experiment YAML configs
    ├── mlp/, tcn/, ptv3/, snn/       — dedicated model directories
    ├── src/                           — orchestration (config, engine, reporting)
    ├── experiments/                   — experiment runner
    ├── run_experiment.py              — end-to-end CLI runner
    └── test_overfit.py               — sanity check

Reuses from ``synapse.synapse``:
    - ``synapse.synapse.data``           — data pipeline + normalization
    - ``synapse.synapse.training``       — Lightning training engine
    - ``synapse.synapse.losses``         — loss functions
    - ``synapse.synapse.training.builders`` — model construction

Baselines
---------
- **MLP**  (``"mlp"``)  : Sanity check — flattened token MLP
- **TCN**  (``"tcn"``)  : Temporal — dilated causal convolutions
- **PTv3** (``"ptv3"``) : Geometric — point self-attention
- **SNN**  (``"snn"``)  : Topological — simplicial message passing
"""

from .base import BaselineBackbone
from .registry import create_backbone, list_available_backbones, register_backbone, registered_names

# Import backbone modules to trigger auto-registration
from .mlp import MLPBackbone as _MLP
from .tcn import TCNBackbone as _TCN
from .ptv3 import PTv3Backbone as _PTv3
from .snn import SNNBackbone as _SNN

__all__ = [
    "BaselineBackbone",
    "create_backbone",
    "list_available_backbones",
    "register_backbone",
    "registered_names",
]
