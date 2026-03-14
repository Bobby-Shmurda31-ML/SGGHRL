from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from ..logging import logger


def save_replay_buffer(buffer, path: str) -> int:
    """Сохранить replay buffer на диск в формате pickle.

    Args:
        buffer: replay buffer из SB3 (ReplayBuffer / DictReplayBuffer).
        path: путь к файлу.

    Returns:
        Количество сохранённых переходов.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    n = buffer.buffer_size if buffer.full else buffer.pos
    is_dict = isinstance(buffer.observations, dict)

    data = {
        "n_valid": n,
        "buffer_size": buffer.buffer_size,
        "pos": buffer.pos,
        "full": buffer.full,
        "is_dict": is_dict,
        "actions": buffer.actions[:n].copy(),
        "rewards": buffer.rewards[:n].copy(),
        "dones": buffer.dones[:n].copy(),
    }

    if is_dict:
        data["observations"] = {k: v[:n].copy() for k, v in buffer.observations.items()}
        data["next_observations"] = {k: v[:n].copy() for k, v in buffer.next_observations.items()}
    else:
        data["observations"] = buffer.observations[:n].copy()
        data["next_observations"] = buffer.next_observations[:n].copy()

    if hasattr(buffer, "timeouts"):
        data["timeouts"] = buffer.timeouts[:n].copy()

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("[Buffer] Saved %d transitions to '%s'", n, path)
    return n


def load_replay_buffer(buffer, path: str, append: bool = False) -> int:
    """Загрузить replay buffer с диска.

    Args:
        buffer: целевой replay buffer для заполнения.
        path: путь к файлу.
        append: если True — добавить к существующим данным,
            иначе — перезаписать.

    Returns:
        Количество загруженных переходов.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    n_saved = data["n_valid"]

    if append:
        start = buffer.buffer_size if buffer.full else buffer.pos
    else:
        start = 0
        buffer.full = False

    space = buffer.buffer_size - start
    n_to_load = min(n_saved, space)

    if n_to_load <= 0:
        logger.warning("[Buffer] No space left (pos=%d, capacity=%d)", start, buffer.buffer_size)
        return 0

    end = start + n_to_load

    if data.get("is_dict", False):
        for k in data["observations"]:
            if k in buffer.observations:
                buffer.observations[k][start:end] = data["observations"][k][:n_to_load]
                buffer.next_observations[k][start:end] = data["next_observations"][k][:n_to_load]
    else:
        buffer.observations[start:end] = data["observations"][:n_to_load]
        buffer.next_observations[start:end] = data["next_observations"][:n_to_load]

    buffer.actions[start:end] = data["actions"][:n_to_load]
    buffer.rewards[start:end] = data["rewards"][:n_to_load]
    buffer.dones[start:end] = data["dones"][:n_to_load]

    if hasattr(buffer, "timeouts") and "timeouts" in data:
        buffer.timeouts[start:end] = data["timeouts"][:n_to_load]

    buffer.pos = end % buffer.buffer_size
    buffer.full = end >= buffer.buffer_size

    logger.info("[Buffer] Loaded %d/%d transitions from '%s' (buffer: %d/%d)",
                n_to_load, n_saved, path, end, buffer.buffer_size)
    return n_to_load