from __future__ import annotations

from typing import Optional, Dict, Tuple, Callable
import numpy as np
from collections import deque

from ..core.base import BaseGoalExtractor


class StateGraph:
    """Дискретизированный граф посещённых состояний.

    Квантует наблюдения в ячейки сетки с шагом discretization.
    Отслеживает количество посещений узлов и переходы между ними.
    При превышении max_nodes удаляется наименее посещённый узел.

    Args:
        goal_extractor: экстрактор целей (BaseGoalExtractor).
        discretization: размер ячейки сетки квантования.
        max_nodes: максимальное число узлов графа.
        context_fn: опциональная функция raw_obs -> hashable,
            результат добавляется к ключу узла (например состояние инвентаря).
        directed: Направленность графа. True — рёбра односторонние.
        valid_state_fn: Фильтр координат для frontier-определения. None — все валидны.
    """

    def __init__(
        self,
        goal_extractor: BaseGoalExtractor,
        discretization: float = 1.0,
        max_nodes: int = 5000,
        context_fn: Optional[Callable] = None,
        directed: bool = False,
        valid_state_fn: Optional[Callable[[tuple], bool]] = None
    ):
        self.goal_extractor = goal_extractor
        self.discretization = float(discretization)
        self.max_nodes = int(max_nodes)
        self.context_fn = context_fn
        self.directed = bool(directed)
        self.valid_state_fn = valid_state_fn

        self.nodes: Dict[tuple, dict] = {}
        self.edges: Dict[Tuple[tuple, tuple], dict] = {}
        self._shortest_path_cache: Dict[Tuple[tuple, tuple], int] = {}
        self._adj_cache: Optional[Dict[tuple, list]] = None
        self._frontier_cache: Optional[set] = None
        self._coord_index: Dict[tuple, set] = {}

    def reset(self):
        """Полностью очистить граф (узлы и рёбра)."""
        self.nodes.clear()
        self.edges.clear()
        self._shortest_path_cache.clear()
        self._invalidate_cache()
        self._coord_index.clear()

    def to_key(self, raw_obs: np.ndarray) -> tuple:
        goal = self.goal_extractor.extract_goal(raw_obs)
        coord = tuple(np.round(goal / self.discretization).astype(int))
        if self.context_fn is not None:
            return coord + (self.context_fn(raw_obs),)
        return coord

    def is_new_state(self, raw_obs: np.ndarray) -> bool:
        """Проверить, является ли состояние новым (не посещалось ранее).

        Args:
            raw_obs: сырое наблюдение.

        Returns:
            True если состояние ещё не в графе.
        """
        return self.to_key(raw_obs) not in self.nodes

    def visit_count(self, raw_obs: np.ndarray) -> int:
        """Количество посещений данного состояния.

        Args:
            raw_obs: сырое наблюдение.

        Returns:
            Число посещений (0 если состояние не в графе).
        """
        node = self.nodes.get(self.to_key(raw_obs))
        return node["visit_count"] if node else 0

    def add_state(self, raw_obs: np.ndarray) -> bool:
        """Добавить состояние в граф или увеличить счётчик посещений.

        Args:
            raw_obs: сырое наблюдение.

        Returns:
            True если узел новый, False если уже существовал.
        """
        key = self.to_key(raw_obs)
        is_new = self._add_state_no_invalidate(raw_obs, key)
        if is_new:
            self._invalidate_cache()  # один вызов
        return is_new

    def _add_state_no_invalidate(self, raw_obs, key) -> bool:
        if key in self.nodes:
            self.nodes[key]["visit_count"] += 1
            return False
        if len(self.nodes) >= self.max_nodes:
            self._remove_least_visited()
        self.nodes[key] = {
            "state": raw_obs.copy(),
            "goal": self.goal_extractor.extract_goal(raw_obs),
            "visit_count": 1,
        }
        dim = self.goal_extractor.goal_dim()
        coord = key[:dim]
        if coord not in self._coord_index:
            self._coord_index[coord] = set()
        self._coord_index[coord].add(key)
        return True

    def add_transition(self, start_raw: np.ndarray, end_raw: np.ndarray, reward: float):
        """Зарегистрировать переход между двумя состояниями.

        Автоматически добавляет оба состояния если их нет.
        Переход из состояния в само себя игнорируется.

        Args:
            start_raw: сырое наблюдение начального состояния.
            end_raw: сырое наблюдение конечного состояния.
            reward: полученная награда за переход.
        """
        k1, k2 = self.to_key(start_raw), self.to_key(end_raw)
        new1 = self._add_state_no_invalidate(start_raw, k1)
        new2 = self._add_state_no_invalidate(end_raw, k2)

        if k1 == k2:
            if new1 or new2:
                self._invalidate_cache()
            return

        is_new_edge = (k1, k2) not in self.edges
        if is_new_edge:
            self.edges[(k1, k2)] = {"count": 0, "total_reward": 0.0}

        edge = self.edges[(k1, k2)]
        edge["count"] += 1
        edge["total_reward"] += reward

        if new1 or new2 or is_new_edge:
            self._invalidate_cache()

    def avg_reward(self, start_raw: np.ndarray, end_raw: np.ndarray) -> Optional[float]:
        """Средняя награда за переход между двумя состояниями.

        Args:
            start_raw: начальное состояние.
            end_raw: конечное состояние.

        Returns:
            Средняя награда или None если переход не зарегистрирован.
        """
        edge = self.edges.get((self.to_key(start_raw), self.to_key(end_raw)))
        if edge is None or edge["count"] == 0:
            return None
        return edge["total_reward"] / edge["count"]

    def _invalidate_cache(self):
        self._shortest_path_cache.clear()
        self._adj_cache = None
        self._frontier_cache = None

    def _build_adjacency(self) -> Dict[tuple, list]:
        adj: Dict[tuple, set] = {k: set() for k in self.nodes}
        for (k1, k2) in self.edges:
            if k1 in self.nodes and k2 in self.nodes:
                adj[k1].add(k2)
                if not self.directed:
                    adj[k2].add(k1)
        return {k: list(v) for k, v in adj.items()}

    def _get_adjacency(self) -> Dict[tuple, list]:
        if self._adj_cache is None:
            self._adj_cache = self._build_adjacency()
        return self._adj_cache

    def _bfs_from(self, start_key: tuple, adj: Dict[tuple, list]) -> Dict[tuple, int]:
        """BFS из одного узла, возвращает {key: distance}."""
        dist = {start_key: 0}
        queue = deque([start_key])
        while queue:
            current = queue.popleft()
            for neighbor in adj.get(current, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def shortest_path_distance(self, raw_obs_start: np.ndarray,
                                raw_obs_end: np.ndarray) -> Optional[int]:
        """Кратчайший путь между двумя состояниями в графе.

        Args:
            raw_obs_start: сырое наблюдение начала.
            raw_obs_end: сырое наблюдение конца.

        Returns:
            Длина пути в рёбрах или None если путь не найден.
        """
        k1 = self.to_key(raw_obs_start)
        k2 = self.to_key(raw_obs_end)

        if k1 not in self.nodes or k2 not in self.nodes:
            return None
        if k1 == k2:
            return 0

        cached = self._shortest_path_cache.get((k1, k2))
        if cached is not None:
            return cached

        adj = self._get_adjacency()
        dist = self._bfs_from(k1, adj)

        for end_key, d in dist.items():
            self._shortest_path_cache[(k1, end_key)] = d

        return dist.get(k2)

    def shortest_path_to_goal(self, raw_obs: np.ndarray,
                               goal: np.ndarray) -> Optional[int]:
        """Кратчайший путь от состояния до ячейки, содержащей goal.

        Args:
            raw_obs: текущее сырое наблюдение.
            goal: вектор цели.

        Returns:
            Длина пути или None.
        """
        start_key = self.to_key(raw_obs)
        goal_coord = tuple(np.round(goal / self.discretization).astype(int))

        if start_key not in self.nodes:
            return None

        matching_keys = self._coord_index.get(goal_coord, set())
        if not matching_keys:
            return None

        adj = self._get_adjacency()
        dist = self._bfs_from(start_key, adj)

        best_dist = None
        for goal_key in matching_keys:
            d = dist.get(goal_key)
            if d is not None and (best_dist is None or d < best_dist):
                best_dist = d
        return best_dist

    def _remove_least_visited(self):
        if not self.nodes:
            return

        adj = self._get_adjacency()
        leaves = [k for k in self.nodes if len(adj.get(k, [])) <= 1]

        if leaves:
            victim = min(leaves, key=lambda k: self.nodes[k]["visit_count"])
        else:
            victim = min(self.nodes, key=lambda k: self.nodes[k]["visit_count"])

        del self.nodes[victim]
        self.edges = {e: v for e, v in self.edges.items() if victim not in e}

        dim = self.goal_extractor.goal_dim()
        coord = victim[:dim]
        if coord in self._coord_index:
            self._coord_index[coord].discard(victim)
            if not self._coord_index[coord]:
                del self._coord_index[coord]

        self._invalidate_cache()

    def _neighbor_coords(self, key: tuple) -> list:
        dim = self.goal_extractor.goal_dim()
        coord = key[:dim]
        neighbors = []
        for axis in range(dim):
            for delta in (-1, 1):
                neighbor = list(coord)
                neighbor[axis] += delta
                n_key = tuple(neighbor)
                if self.context_fn is not None:
                    n_key = n_key + key[dim:]
                neighbors.append(n_key)
        return neighbors

    def is_frontier(self, raw_obs: np.ndarray) -> bool:
        """Проверить, является ли состояние frontier-узлом.

        Frontier — узел, у которого хотя бы один сосед
        по сетке не присутствует в графе.

        Args:
            raw_obs: сырое наблюдение.

        Returns:
            True если узел на границе исследованного пространства.
        """
        key = self.to_key(raw_obs)
        if key not in self.nodes:
            return False
        dim = self.goal_extractor.goal_dim()
        for n_key in self._neighbor_coords(key):
            coord = n_key[:dim]
            if n_key not in self.nodes and self._is_valid_coord(coord):
                return True
        return False

    def get_frontier_nodes(self) -> set:
        """Множество ключей frontier-узлов (кэшируется).

        Returns:
            set of tuple — ключи узлов на границе.
        """
        if self._frontier_cache is not None:
            return self._frontier_cache
        dim = self.goal_extractor.goal_dim()
        frontier = set()
        for key in self.nodes:
            for n_key in self._neighbor_coords(key):
                coord = n_key[:dim]
                if n_key not in self.nodes and self._is_valid_coord(coord):
                    frontier.add(key)
                    break
        self._frontier_cache = frontier
        return frontier

    def distance_to_frontier(self, raw_obs: np.ndarray) -> Optional[int]:
        """Кратчайший путь от состояния до ближайшего frontier-узла.

        Args:
            raw_obs: сырое наблюдение.

        Returns:
            Расстояние в рёбрах или None если путь не найден.
            0 если сам узел — frontier.
        """
        key = self.to_key(raw_obs)
        if key not in self.nodes:
            return None

        frontier = self.get_frontier_nodes()
        if not frontier:
            return None

        if key in frontier:
            return 0

        adj = self._get_adjacency()
        visited = {key}
        queue = deque([(key, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in adj.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if neighbor in frontier:
                    return dist + 1
                queue.append((neighbor, dist + 1))
        return None

    def _is_valid_coord(self, coord: tuple) -> bool:
        """Проверить, может ли координата быть реальным состоянием."""
        low, high = self.goal_extractor.goal_bounds()
        dim = self.goal_extractor.goal_dim()
        for i in range(dim):
            val = coord[i] * self.discretization
            if val < low[i] or val > high[i]:
                return False
        if self.valid_state_fn is not None:
            return self.valid_state_fn(coord)
        return True

    def bfs_distances(self, raw_obs: np.ndarray) -> Dict[tuple, int]:
        """BFS-расстояния от состояния до всех достижимых узлов."""
        key = self.to_key(raw_obs)
        if key not in self.nodes:
            return {}
        adj = self._get_adjacency()
        return self._bfs_from(key, adj)

    def rebuild_index(self):
        """Перестроить _coord_index и инвалидировать кэши после загрузки."""
        self._coord_index.clear()
        dim = self.goal_extractor.goal_dim()
        for key in self.nodes:
            coord = key[:dim]
            if coord not in self._coord_index:
                self._coord_index[coord] = set()
            self._coord_index[coord].add(key)
        self._invalidate_cache()

    def get_reward_sources(self, min_abs_reward: float = 0.5) -> Dict[tuple, float]:
        """Узлы со значимыми наградами (из входящих рёбер).

        Для каждого узла берётся максимальная по модулю средняя
        награда среди всех входящих рёбер. Используется для
        GraphValueShapingCallback.

        Args:
            min_abs_reward: минимальный |avg_reward| для включения.

        Returns:
            Словарь {ключ_узла: средняя_награда}.
        """
        node_rewards: Dict[tuple, float] = {}
        for (k1, k2), edge in self.edges.items():
            if edge["count"] == 0 or k2 not in self.nodes:
                continue
            avg_r = edge["total_reward"] / edge["count"]
            current = node_rewards.get(k2, 0.0)
            if abs(avg_r) > abs(current):
                node_rewards[k2] = avg_r
        return {k: v for k, v in node_rewards.items() if abs(v) >= min_abs_reward}