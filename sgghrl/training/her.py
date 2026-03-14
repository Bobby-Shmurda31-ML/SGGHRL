from __future__ import annotations

import random
from typing import Optional, List, Tuple
import numpy as np

from ..core.base import BaseGoalExtractor, HERCapable


class HERBuffer:
    """Буфер Hindsight Experience Replay для manager'а.

    Собирает переходы в течение эпизода и генерирует
    дополнительные переходы с подменёнными целями.

    Если manager_env реализует HERCapable — используются его методы
    get_achieved_goal, compute_her_reward, relabel_obs_for_her.
    Иначе — fallback на GoalExtractor со sparse-наградой ±reward_scale.

    Args:
        manager_env: среда manager'а.
        goal_extractor: экстрактор целей.
        k_future: сколько будущих целей сэмплировать на каждый переход.
        strategy: стратегия HER — "future", "final" или "episode".
        reward_scale: масштаб HER-награды (при fallback).
    """

    def __init__(self,
                 manager_env,
                 goal_extractor: BaseGoalExtractor,
                 k_future: int = 4, strategy: str = "future",
                 reward_scale: float = 0.1
    ):
        self._her_capable = isinstance(manager_env, HERCapable)
        self.manager_env = manager_env
        self.goal_extractor = goal_extractor
        self.k_future = int(k_future)
        self.strategy = str(strategy)
        self.reward_scale = float(reward_scale)
        self.episode_buffer: List[dict] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        raw_obs: np.ndarray,
        next_raw_obs: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Добавить переход в эпизодный буфер.

        Args:
            obs: обёрнутое наблюдение (из observation_space manager'а).
            action: действие manager'а.
            next_obs: следующее обёрнутое наблюдение.
            raw_obs: сырое наблюдение до шага.
            next_raw_obs: сырое наблюдение после шага.
            reward: полученная награда.
            done: флаг завершения эпизода.
        """
        self.episode_buffer.append(
            {
                "obs": obs.copy(),
                "action": np.array(action).copy(),
                "next_obs": next_obs.copy(),
                "raw_obs": raw_obs.copy(),
                "next_raw_obs": next_raw_obs.copy(),
                "reward": float(reward),
                "done": bool(done),
            }
        )

    def clear(self):
        """Очистить эпизодный буфер. Вызывается в конце каждого эпизода."""
        self.episode_buffer = []

    def get_her_transitions(self) -> List[dict]:
        """Сгенерировать HER-переходы из текущего эпизодного буфера.

        Каждый переход — словарь с ключами:
            obs, action, next_obs, reward, done.

        Returns:
            Список переразмеченных переходов.
        """
        if len(self.episode_buffer) < 2:
            return []

        her_transitions: List[dict] = []

        if self.strategy == "final":
            final_raw = self.episode_buffer[-1]["next_raw_obs"]
            new_goal = self._get_achieved_goal(final_raw)
            for transition in self.episode_buffer:
                trans = self._relabel_transition(transition, new_goal)
                if trans is not None:
                    her_transitions.append(trans)

        elif self.strategy == "future":
            for idx in range(len(self.episode_buffer)):
                transition = self.episode_buffer[idx]
                future_indices = list(range(idx + 1, len(self.episode_buffer)))

                if not future_indices:
                    new_goal = self._get_achieved_goal(transition["next_raw_obs"])
                    trans = self._relabel_transition(transition, new_goal)
                    if trans is not None:
                        her_transitions.append(trans)
                    continue

                k = min(self.k_future, len(future_indices))
                selected = random.sample(future_indices, k)

                for future_idx in selected:
                    future_raw = self.episode_buffer[future_idx]["next_raw_obs"]
                    new_goal = self._get_achieved_goal(future_raw)
                    trans = self._relabel_transition(transition, new_goal)
                    if trans is not None:
                        her_transitions.append(trans)

        elif self.strategy == "episode":
            all_goals = [self._get_achieved_goal(t["next_raw_obs"]) for t in self.episode_buffer]
            for transition in self.episode_buffer:
                k = min(self.k_future, len(all_goals))
                selected_goals = random.sample(all_goals, k)
                for new_goal in selected_goals:
                    trans = self._relabel_transition(transition, new_goal)
                    if trans is not None:
                        her_transitions.append(trans)

        return her_transitions

    def _get_achieved_goal(self, raw_obs: np.ndarray) -> np.ndarray:
        if self._her_capable:
            return self.manager_env.get_achieved_goal(raw_obs)
        return self.goal_extractor.extract_goal(raw_obs)

    def _compute_her_reward(self, start_raw: np.ndarray, end_raw: np.ndarray,
                            new_goal: np.ndarray) -> Tuple[float, bool]:
        if self._her_capable:
            reward, done = self.manager_env.compute_her_reward(start_raw, end_raw, new_goal)
            return reward * self.reward_scale, done

        achieved = self._get_achieved_goal(end_raw)
        dist = self.goal_extractor.compute_distance(achieved, new_goal)
        threshold = getattr(self.manager_env, "success_threshold", 0.5)
        success = dist < threshold
        reward = self.reward_scale if success else -self.reward_scale
        return reward, bool(success)

    def _relabel_obs(self, obs: np.ndarray, raw_obs: np.ndarray, new_goal: np.ndarray) -> np.ndarray:
        if self._her_capable:
            return self.manager_env.relabel_obs_for_her(obs, raw_obs, new_goal)
        return obs

    def _relabel_transition(self, transition: dict, new_goal: np.ndarray) -> Optional[dict]:
        new_obs = self._relabel_obs(transition["obs"], transition["raw_obs"], new_goal)
        new_next_obs = self._relabel_obs(transition["next_obs"], transition["next_raw_obs"], new_goal)
        new_reward, new_done = self._compute_her_reward(transition["raw_obs"], transition["next_raw_obs"], new_goal)

        return {
            "obs": new_obs,
            "action": np.array(transition["action"]).copy(),
            "next_obs": new_next_obs,
            "reward": float(new_reward),
            "done": bool(new_done),
        }