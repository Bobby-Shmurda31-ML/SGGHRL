from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def obs_to_tensor(obs, device):
    """Конвертировать наблюдение в тензор с batch-размерностью.

    Поддерживает ndarray и dict of ndarrays.

    Args:
        obs: наблюдение.
        device: устройство PyTorch.

    Returns:
        Тензор или словарь тензоров.
    """
    if isinstance(obs, dict):
        return {
            k: torch.as_tensor(v).unsqueeze(0).float().to(device)
            for k, v in obs.items()
        }
    return torch.as_tensor(obs).unsqueeze(0).float().to(device)


def obs_add_batch_dim(obs):
    """Добавить ведущую batch-размерность к наблюдению.

    Поддерживает ndarray и dict of ndarrays.

    Args:
        obs: наблюдение.

    Returns:
        Наблюдение с добавленной размерностью.
    """
    if isinstance(obs, dict):
        return {k: np.expand_dims(v, 0) for k, v in obs.items()}
    return np.expand_dims(obs, 0)

def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Скопировать state_dict на CPU (detach + clone).

    Args:
        state_dict: словарь параметров PyTorch.

    Returns:
        Копия state_dict на CPU.
    """
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}