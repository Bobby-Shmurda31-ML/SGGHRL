from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np

from ..core.base import BaseGoalExtractor


class GoalExtractor(BaseGoalExtractor):
    """Дефолтный экстрактор целей: весь вектор наблюдения = цель.

    Подходит для простых сред, где позиция агента — это всё наблюдение.
    Для сложных сред создайте свой подкласс BaseGoalExtractor.

    Args:
        obs_space: пространство наблюдений базовой среды.
    """

    def __init__(self, obs_space: gym.spaces.Box):
        self._dim = obs_space.shape[0]
        self._low = obs_space.low.astype(np.float32)
        self._high = obs_space.high.astype(np.float32)

    def extract_goal(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32)

    def goal_dim(self) -> int:
        return self._dim

    def goal_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._low, self._high

    def compute_distance(self, achieved: np.ndarray, desired: np.ndarray) -> float:
        return float(np.linalg.norm(achieved - desired))