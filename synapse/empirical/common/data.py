from __future__ import annotations

import numpy as np

from synapse.empirical.common.tasks import SequenceSample, generate_control_dataset, generate_memory_task_dataset


def samples_to_arrays(samples: list[SequenceSample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([sample.sequence for sample in samples], axis=0)
    y = np.asarray([sample.target for sample in samples])
    return x, y
