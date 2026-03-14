from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from .logging import logger


def set_global_seed(seed: Optional[int] = None) -> Optional[int]:
    """Установить глобальный seed для воспроизводимости.

    Устанавливает seed для random, numpy, torch (CPU + CUDA).

    Args:
        seed: значение seed (None — не устанавливать).

    Returns:
        Использованный seed или None.
    """
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Детерминизм для CUDA (может снизить производительность)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d", seed)
    return seed