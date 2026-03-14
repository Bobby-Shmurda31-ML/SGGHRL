from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import gymnasium as gym
import numpy as np


class BaseGoalExtractor(ABC):
    """Базовый класс для извлечения и сравнения целей из наблюдений.

    Определяет как преобразовывать сырые наблюдения среды в векторы целей
    и как измерять расстояние между ними. Каждый SGGHRLAgent требует
    экземпляр этого класса.

    Обязательные методы для реализации:
        extract_goal, goal_dim, goal_bounds, compute_distance

    Методы с дефолтной реализацией (можно переопределить):
        inject_goal, is_success
    """

    @abstractmethod
    def extract_goal(self, obs: np.ndarray) -> np.ndarray:
        """Извлечь достигнутую цель из сырого наблюдения среды.

        Args:
            obs: сырое наблюдение из базовой среды.

        Returns:
            Вектор цели float32 размерности (goal_dim(),).
        """
        pass

    @abstractmethod
    def goal_dim(self) -> int:
        """Размерность вектора цели."""
        pass

    @abstractmethod
    def goal_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Границы значений цели (low, high).

        Returns:
            Кортеж из двух float32 массивов размерности (goal_dim(),).
        """
        pass

    @abstractmethod
    def compute_distance(self, achieved: np.ndarray, desired: np.ndarray) -> float:
        """Скалярное расстояние между двумя целями (>= 0).

        Args:
            achieved: достигнутая цель.
            desired: желаемая цель.

        Returns:
            Неотрицательное расстояние.
        """
        pass

    def inject_goal(self, raw_obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Вставить цель в сырое наблюдение.

        По умолчанию перезаписывает первые len(goal) элементов.

        Args:
            raw_obs: исходное наблюдение.
            goal: вектор цели для вставки.

        Returns:
            Модифицированное наблюдение.
        """
        result = raw_obs.copy()
        result[:len(goal)] = goal
        return result

    def is_success(self, achieved: np.ndarray, desired: np.ndarray, threshold: float) -> bool:
        """Проверить, достигнута ли цель.

        По умолчанию: compute_distance(achieved, desired) < threshold.

        Args:
            achieved: достигнутая цель.
            desired: желаемая цель.
            threshold: порог расстояния.

        Returns:
            True если цель достигнута.
        """
        return self.compute_distance(achieved, desired) < threshold


class BaseWorkerEnv(gym.Wrapper, ABC):
    """Низкоуровневая goal-conditioned среда для worker-политики.

    Worker получает подцель и должен достичь её за ограниченное
    число шагов. SGGHRLAgent создаёт экземпляр через kwargs:

        worker_env = WorkerEnvClass(
            env=base_env,
            goal_extractor=extractor,
            success_threshold=0.5,
        )

    Обязательные методы/свойства:
        set_goal — установить текущую подцель.
        get_success_rate — текущий success rate для логирования.
        last_obs (property) — последнее сырое наблюдение из базовой среды.
    """

    @abstractmethod
    def set_goal(self, goal: np.ndarray) -> None:
        """Установить текущую подцель для worker'а.

        Args:
            goal: вектор цели (выход GoalExtractor.extract_goal).
        """
        pass

    @abstractmethod
    def get_success_rate(self) -> float:
        """Текущий success rate в диапазоне [0.0, 1.0].

        Используется тренером для логирования и
        WorkerCurriculumCallback для продвижения по стадиям.
        """
        pass

    @property
    @abstractmethod
    def last_obs(self) -> np.ndarray:
        """Последнее сырое (unwrapped) наблюдение из базовой среды.

        Обновляется при каждом вызове reset() и step().
        Manager и граф состояний читают это свойство для отслеживания
        истинного состояния агента.
        """
        pass


class BaseManagerEnv(gym.Wrapper, ABC):
    """Высокоуровневая среда планирования для manager-политики.

    Каждое действие manager'а выбирает подцель. Среда внутри
    запускает worker'а на несколько шагов для её достижения,
    затем возвращает результирующее наблюдение и награду.

    SGGHRLAgent создаёт экземпляр через kwargs:

        manager_env = ManagerEnvClass(
            env=base_env,
            worker_env=worker_env,
            worker_model=worker_ppo,
            graph=state_graph,
            goal_extractor=extractor,
            max_worker_steps=15,
            success_threshold=0.5,
        )

    Обязательные свойства/методы:
        worker_env (property) — связанный экземпляр BaseWorkerEnv.
        success_threshold (property) — порог расстояния для достижения подцели.
        set_worker_model — замена worker-политики (при load_worker).
    """

    @property
    @abstractmethod
    def worker_env(self) -> BaseWorkerEnv:
        """Связанный экземпляр worker-среды."""
        pass

    @property
    @abstractmethod
    def success_threshold(self) -> float:
        """Порог расстояния, ниже которого подцель считается достигнутой."""
        pass

    @property
    @abstractmethod
    def final_goal(self) -> Optional[np.ndarray]:
        """Финальная цель текущего эпизода (вектор цели).

        Возвращает None если цель не определена.
        Устанавливается при reset(), используется
        GraphRewardShapingCallback для potential-based shaping.
        """
        pass

    @abstractmethod
    def set_worker_model(self, model: Any) -> None:
        """Заменить worker-политику.

        Вызывается из SGGHRLAgent.load_worker().

        Args:
            model: новая worker-модель (например PPO).
        """
        pass


@runtime_checkable
class CurriculumCapable(Protocol):
    """Протокол поддержки curriculum learning для worker-среды.

    Проверяется WorkerCurriculumCallback через isinstance().
    Реализуйте оба метода для автоматического продвижения по стадиям.
    """

    def set_curriculum_weights(self, weights: Dict[int, float]) -> None:
        """Установить веса сэмплирования целей по расстоянию.

        Args:
            weights: словарь {расстояние: вероятность}.
        """
        ...

    @property
    def curriculum_distance(self) -> float:
        """Текущая максимальная дистанция curriculum (для логов)."""
        ...


@runtime_checkable
class HERCapable(Protocol):
    """Протокол поддержки Hindsight Experience Replay для manager-среды.

    Проверяется HERCallback через isinstance().
    Без реализации HER использует fallback на GoalExtractor
    со sparse-наградой ±reward_scale.
    """

    def get_achieved_goal(self, raw_obs: np.ndarray) -> np.ndarray:
        """Извлечь достигнутую цель из сырого наблюдения для HER.

        Args:
            raw_obs: сырое наблюдение из базовой среды.

        Returns:
            Вектор достигнутой цели.
        """
        ...

    def compute_her_reward(self, start_raw: np.ndarray, end_raw: np.ndarray, new_goal: np.ndarray) -> Tuple[float, bool]:
        """Вычислить награду для HER-переразмеченного перехода.

        Args:
            start_raw: сырое наблюдение в начале перехода.
            end_raw: сырое наблюдение в конце перехода.
            new_goal: подменённая цель.

        Returns:
            Кортеж (reward, done).
        """
        ...

    def relabel_obs_for_her(self, obs: np.ndarray, raw_obs: np.ndarray, new_goal: np.ndarray) -> np.ndarray:
        """Переразметить обёрнутое наблюдение с подменённой целью.

        Args:
            obs: обёрнутое наблюдение (из observation_space manager'а).
            raw_obs: соответствующее сырое наблюдение.
            new_goal: новая цель для подмены.

        Returns:
            Переразмеченное наблюдение.
        """
        ...