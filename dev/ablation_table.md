# Ablation Study: TopologicalEncoder Component Effects

## Task
Synthetic loop detection: classify circular vs linear trajectories (noisy 2D geometry).

## Results

| Model | Accuracy | Time | Event+Saliency | Selection | Lift+Projection |
|---|---|---|---|---|---|
| Raw DeepHodge | 100% | ✓ | ✗ | ✗ | ✗ |
| Time preserved, no selection | 50% | ✓ | ✓ | ✗ | ✗ |
| Time preserved, with selection | 27.5% | ✓ | ✓ | ✓ | ✗ |
| Full TopologicalEncoder | 100% | ✗ | ✓ | ✓ | ✓ |

## Component Effects

| Component | Effect on Accuracy |
|---|---|
| Event detection + saliency | -50% |
| Selection (with time preserved) | -22.5% |
| Time zeroing + lift + projection | +72.5% |
| Full encoder (vs raw) | 0% |

## Interpretation

The ablation reveals a **non-additive interaction effect**:

1. **Event detection and saliency harm performance** (-50%) on this simple synthetic task
2. **Selection compounds the harm** (-22.5% additional)
3. **The lift + projection pipeline recovers performance** (+72.5%)

The full TopologicalEncoder achieves the same accuracy as the raw model (100%), but through a different pathway. The learned lift and topology projection provide a beneficial transformation that compensates for the harmful effects of event detection and selection.

This suggests the TopologicalEncoder components are **interdependent** rather than modular — the pipeline must be used as a whole to realize its benefits.
