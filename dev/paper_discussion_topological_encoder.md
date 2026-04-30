# Paper-Ready Discussion: TopologicalEncoder Ablation

## Discussion

Our ablation study on the synthetic loop detection task reveals a nuanced relationship between TopologicalEncoder components and task performance. Contrary to a simple modular view where each component adds independent value, we observe strong **interaction effects** that depend on the full pipeline being used as a cohesive unit.

When applied in isolation, event detection and saliency computation reduce performance by 50 percentage points compared to the raw sequence baseline. Adding anchor selection compounds this harm, reducing accuracy further to 27.5%. However, when the complete pipeline—including the learned lift and topology projection—is used, performance recovers to 100%, matching the raw baseline.

This pattern suggests that the TopologicalEncoder components are **interdependent** rather than modular. The learned lift and topology projection provide a beneficial transformation that compensates for the information loss introduced by event detection and selection. The pipeline must be used as a whole to realize its benefits; partial ablations do not reflect the intended use case.

These results inform our understanding of when the TopologicalEncoder should be deployed:

1. **For simple synthetic tasks with clean temporal geometry**, the raw sequence model already achieves near-perfect performance. The TopologicalEncoder provides no additional benefit and may introduce unnecessary complexity.

2. **For noisy or complex real-world data**, the full TopologicalEncoder pipeline may provide robustness through its learned geometric normalization and topology-aware feature extraction. The ablation results suggest that the benefit comes from the synergistic interaction of all components, not from any single piece in isolation.

3. **The encoder acts as a selective inductive bias**—it is designed for regimes where compression, denoising, or anchor extraction are genuinely useful. It is not a universal performance booster for all sequential tasks.

We therefore avoid claiming that the TopologicalEncoder always improves performance. Instead, we position it as a specialized front-end that can help on data where the topological structure of the time-projected geometry is informative, but may hurt on simple trajectories where the raw temporal geometry is already sufficient for discrimination.
