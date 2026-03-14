from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

import numpy as np
import torch


@dataclass
class TimeMeter:
    """Счётчик FPS и ETA для тренировочного цикла.

    Attributes:
        start_time: время старта.
        last_time: время последнего тика.
        last_step: шаг последнего тика.
    """
    start_time: float
    last_time: float
    last_step: int

    @classmethod
    def start(cls) -> "TimeMeter":
        """Создать и запустить новый счётчик."""
        now = time.time()
        return cls(start_time=now, last_time=now, last_step=0)

    def tick(self, step: int, total_steps: int) -> Dict[str, Any]:
        """Обновить счётчик и вернуть метрики.

        Args:
            step: текущий шаг.
            total_steps: общее число шагов.

        Returns:
            Словарь с fps, eta (timedelta), elapsed (timedelta).
        """
        now = time.time()
        dt = now - self.last_time
        steps_delta = step - self.last_step
        fps = (steps_delta / dt) if dt > 0 else 0.0
        eta_sec = int((total_steps - step) / fps) if fps > 0 else 0

        self.last_time = now
        self.last_step = step

        return {
            "fps": fps,
            "eta": datetime.timedelta(seconds=eta_sec),
            "elapsed": datetime.timedelta(seconds=int(now - self.start_time)),
        }


@dataclass
class LinearEpsilonSchedule:
    """Линейное расписание epsilon для exploration.

    Args:
        epsilon_start: начальное значение.
        epsilon_min: минимальное значение.
        decay_steps: число шагов для линейного снижения
            (None — использовать total_steps).
    """
    epsilon_start: float = 0.5
    epsilon_min: float = 0.05
    decay_steps: Optional[int] = None

    def value(self, step: int, total_steps: int) -> float:
        """Текущее значение epsilon.

        Args:
            step: текущий шаг.
            total_steps: общее число шагов.

        Returns:
            Значение epsilon в [epsilon_min, epsilon_start].
        """
        decay_steps = self.decay_steps if self.decay_steps is not None else total_steps
        if decay_steps <= 0:
            return float(self.epsilon_min)

        if step < decay_steps:
            eps = self.epsilon_start - (step / decay_steps) * (self.epsilon_start - self.epsilon_min)
        else:
            eps = self.epsilon_min
        return float(np.clip(eps, self.epsilon_min, self.epsilon_start))