# SYNAPSE Z3

This directory contains the publication-ready topology-first refactor of SYNAPSE.

Structure:

- `synapse_arch/`: task-agnostic representation architecture
- `synapse_core/`: exact audit, selector, lift, and differentiable proxy logic
- `empirical/`: synthetic topology-sensitive datasets
- `evaluation/`: ablations, metrics, visualization
- `verification/`: invariance and stability checks
- `verification/scripts/`: run_verification, topology_audit, validation_runner
- `evaluation/scripts/`: evaluate, run_ablations, visualize
- `empirical/scripts/`: prepare_dataset, compute_proxy, run_suite
- `synapse/scripts/`: train, evaluate, infer, run_ablations, deploy_test
- `config/`: experiment configurations
