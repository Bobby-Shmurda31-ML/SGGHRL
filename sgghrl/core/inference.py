from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Callable
from collections import deque

import numpy as np
import gymnasium as gym

from ..core.base import BaseGoalExtractor
from ..core.graph import StateGraph
from ..logging import logger


class InferenceStrategy(ABC):
    """Базовый класс стратегии выбора действия на инференсе.

    Определяет как agent выбирает подцель: чистая policy,
    граф-планировщик или их комбинация.
    """

    @abstractmethod
    def select_action(
        self,
        obs: np.ndarray,
        raw_obs: np.ndarray,
        manager_model,
        graph: StateGraph,
        goal_extractor: BaseGoalExtractor,
        manager_env,
    ) -> np.ndarray:
        """Выбрать действие manager'а.

        Args:
            obs: обёрнутое наблюдение (из observation_space manager'а).
            raw_obs: сырое наблюдение из базовой среды.
            manager_model: обученная SAC-модель manager'а.
            graph: граф состояний.
            goal_extractor: экстрактор целей.
            manager_env: среда manager'а.

        Returns:
            Действие manager'а (np.ndarray).
        """
        pass


class PolicyOnlyStrategy(InferenceStrategy):
    """Стратегия: чистая policy без использования графа.

    Эквивалентна текущему поведению agent.demo().
    """

    def select_action(self, obs, raw_obs, manager_model, graph,
                      goal_extractor, manager_env) -> np.ndarray:
        action, _ = manager_model.predict(obs, deterministic=True)
        return action


class GraphPlannerStrategy(InferenceStrategy):
    """Стратегия: граф генерирует кандидатов, critic оценивает.

    Граф предлагает стратегически осмысленные подцели
    (путь до финальной цели, соседи, frontier-узлы).
    SAC critic выбирает лучшего кандидата по Q(obs, action).

    Args:
        n_policy_samples: число сэмплов из policy для кандидатов.
        include_graph_neighbors: включать ли соседей по графу.
        include_path_nodes: сколько узлов с кратчайшего пути
            до финальной цели включать (0 — не включать).
        include_frontier: включать ли frontier-узлы.
        max_frontier_candidates: максимум frontier-кандидатов.
        goal_to_action_fn: имя метода manager_env для конвертации
            goal → action (None — использовать 'goal_to_action').
    """

    def __init__(
        self,
        n_policy_samples: int = 3,
        include_graph_neighbors: bool = True,
        include_path_nodes: int = 3,
        include_frontier: bool = False,
        max_frontier_candidates: int = 3,
        goal_to_action_fn: Optional[str] = None,
    ):
        self.n_policy_samples = int(n_policy_samples)
        self.include_graph_neighbors = bool(include_graph_neighbors)
        self.include_path_nodes = int(include_path_nodes)
        self.include_frontier = bool(include_frontier)
        self.max_frontier_candidates = int(max_frontier_candidates)
        self._goal_to_action_fn = goal_to_action_fn or "goal_to_action"

    def _goal_to_action(self, manager_env, goal: np.ndarray) -> Optional[np.ndarray]:
        """Конвертировать goal-координаты в action manager'а."""
        fn = getattr(manager_env, self._goal_to_action_fn, None)
        if fn is None:
            return None
        try:
            action = fn(goal)
            return np.asarray(action, dtype=np.float32)
        except Exception:
            return None

    def _get_candidates(
        self,
        raw_obs: np.ndarray,
        manager_model,
        graph: StateGraph,
        goal_extractor: BaseGoalExtractor,
        manager_env,
        obs: np.ndarray,
    ) -> List[np.ndarray]:
        """Собрать список кандидатов-действий."""
        candidates = []

        # 1) Policy samples
        for _ in range(self.n_policy_samples):
            action, _ = manager_model.predict(obs, deterministic=False)
            candidates.append(np.asarray(action, dtype=np.float32))

        # Deterministic policy sample
        action_det, _ = manager_model.predict(obs, deterministic=True)
        candidates.append(np.asarray(action_det, dtype=np.float32))

        current_key = graph.to_key(raw_obs)
        if current_key not in graph.nodes:
            return candidates

        # 2) Graph neighbors
        if self.include_graph_neighbors:
            adj = graph._get_adjacency()
            for neighbor_key in adj.get(current_key, []):
                node = graph.nodes.get(neighbor_key)
                if node is None:
                    continue
                goal = node["goal"]
                action = self._goal_to_action(manager_env, goal)
                if action is not None:
                    candidates.append(action)

        # 3) Nodes on shortest path to final goal
        if self.include_path_nodes > 0:
            final_goal = getattr(manager_env, "final_goal", None)
            if final_goal is not None:
                path_nodes = self._get_path_candidates(
                    raw_obs, final_goal, graph, goal_extractor
                )
                for goal in path_nodes[: self.include_path_nodes]:
                    action = self._goal_to_action(manager_env, goal)
                    if action is not None:
                        candidates.append(action)

        # 4) Frontier nodes
        if self.include_frontier:
            frontier_goals = self._get_frontier_candidates(
                raw_obs, graph
            )
            for goal in frontier_goals[: self.max_frontier_candidates]:
                action = self._goal_to_action(manager_env, goal)
                if action is not None:
                    candidates.append(action)

        return candidates

    def _get_path_candidates(
        self,
        raw_obs: np.ndarray,
        final_goal: np.ndarray,
        graph: StateGraph,
        goal_extractor: BaseGoalExtractor,
    ) -> List[np.ndarray]:
        """Получить узлы на кратчайшем пути до финальной цели."""
        start_key = graph.to_key(raw_obs)
        if start_key not in graph.nodes:
            return []

        goal_coord = tuple(
            np.round(final_goal / graph.discretization).astype(int)
        )
        matching_keys = graph._coord_index.get(goal_coord, set())
        if not matching_keys:
            return []

        adj = graph._get_adjacency()
        dist = graph._bfs_from(start_key, adj)

        best_goal_key = None
        best_dist = None
        for gk in matching_keys:
            d = dist.get(gk)
            if d is not None and (best_dist is None or d < best_dist):
                best_dist = d
                best_goal_key = gk

        if best_goal_key is None:
            return []

        # Reconstruct path via BFS (re-run with parent tracking)
        parent = {start_key: None}
        queue = deque([start_key])
        found = False
        while queue:
            current = queue.popleft()
            if current == best_goal_key:
                found = True
                break
            for neighbor in adj.get(current, []):
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)

        if not found:
            return []

        path_keys = []
        cur = best_goal_key
        while cur is not None:
            path_keys.append(cur)
            cur = parent[cur]
        path_keys.reverse()

        # Skip current node, return goals of path nodes
        result = []
        for key in path_keys[1:]:
            node = graph.nodes.get(key)
            if node is not None:
                result.append(node["goal"].copy())

        return result

    def _get_frontier_candidates(
        self,
        raw_obs: np.ndarray,
        graph: StateGraph,
    ) -> List[np.ndarray]:
        """Получить ближайшие frontier-узлы."""
        start_key = graph.to_key(raw_obs)
        if start_key not in graph.nodes:
            return []

        frontier = graph.get_frontier_nodes()
        if not frontier:
            return []

        adj = graph._get_adjacency()
        dist = graph._bfs_from(start_key, adj)

        frontier_with_dist = []
        for fk in frontier:
            d = dist.get(fk)
            if d is not None:
                frontier_with_dist.append((d, fk))

        frontier_with_dist.sort(key=lambda x: x[0])

        result = []
        for _, key in frontier_with_dist:
            node = graph.nodes.get(key)
            if node is not None:
                result.append(node["goal"].copy())

        return result

    def _evaluate_candidates(
        self,
        obs: np.ndarray,
        candidates: List[np.ndarray],
        manager_model,
    ) -> np.ndarray:
        """Оценить кандидатов через SAC critic и выбрать лучшего."""
        import torch

        if not candidates:
            action, _ = manager_model.predict(obs, deterministic=True)
            return action

        if len(candidates) == 1:
            return candidates[0]

        device = manager_model.device

        # Prepare observation batch
        if isinstance(obs, dict):
            obs_batch = {
                k: torch.as_tensor(
                    np.stack([v] * len(candidates)), dtype=torch.float32
                ).to(device)
                for k, v in obs.items()
            }
        else:
            obs_batch = torch.as_tensor(
                np.stack([obs] * len(candidates)), dtype=torch.float32
            ).to(device)

        actions_batch = torch.as_tensor(
            np.stack(candidates), dtype=torch.float32
        ).to(device)

        # Clip actions to action space bounds
        if hasattr(manager_model, "action_space"):
            low = torch.as_tensor(
                manager_model.action_space.low, dtype=torch.float32
            ).to(device)
            high = torch.as_tensor(
                manager_model.action_space.high, dtype=torch.float32
            ).to(device)
            actions_batch = torch.clamp(actions_batch, low, high)

        with torch.no_grad():
            q1, q2 = manager_model.critic(obs_batch, actions_batch)
            q_values = torch.min(q1, q2).squeeze()

        best_idx = int(q_values.argmax().item())
        return candidates[best_idx]

    def select_action(self, obs, raw_obs, manager_model, graph,
                      goal_extractor, manager_env) -> np.ndarray:
        candidates = self._get_candidates(
            raw_obs, manager_model, graph, goal_extractor, manager_env, obs
        )

        if not candidates:
            action, _ = manager_model.predict(obs, deterministic=True)
            return action

        return self._evaluate_candidates(obs, candidates, manager_model)