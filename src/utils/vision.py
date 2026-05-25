from __future__ import annotations

import numpy as np


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, eps)
