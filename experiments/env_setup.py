import gymnasium as gym
import numpy as np
import random
import collections
from typing import Optional, Dict, Tuple, List
import pygame
import matplotlib.pyplot as plt
from sgghrl.core.base import BaseWorkerEnv, BaseManagerEnv
from sgghrl.core.goals import GoalExtractor


class DungeonWorkerEnv(BaseWorkerEnv):
    """Worker для DungeonHeist - навигация + избегание."""

    SUCCESS_REWARD_BASE = 1.0
    SUCCESS_REWARD_POWER = 0.5
    GOAL_TIMEOUT_PENALTY = -0.5
    STEP_PENALTY = -0.1
    GOAL_TIMEOUT_MULTIPLIER = 3
    GOAL_TIMEOUT_OFFSET = 5
    WALL_BUMP_PENALTY = -0.15

    def __init__(self, env: gym.Env, goal_extractor: GoalExtractor,
                 success_threshold: float = 0.8, view_radius: int = 5):
        super().__init__(env)
        self.goal_extractor = goal_extractor
        self.grid_size = float(max(env.grid_size))
        self.success_threshold = success_threshold
        self.view_radius = view_radius
        self._frame_buffer = None

        self.goal = np.zeros(2, dtype=np.float32)

        view_size = (2 * view_radius + 1) ** 2
        obs_dim = 2 + 1 + view_size + 2 + 1
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.curriculum_distance = view_radius
        self.curriculum_min_distance = 0.0
        self.max_curriculum_distance = view_radius
        self._distance_weights = None
        self._steps_since_goal = 0

        self._distance_cache = {}
        self._cache_valid = False
        self.recent_successes = collections.deque(maxlen=100)
        self.continuous_goals = False
        self._last_obs = None

    @property
    def last_obs(self):
        return self._last_obs

    def _get_distances_from(self, start: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """BFS от одной клетки — ленивый с кэшированием."""
        if start in self._distance_cache:
            return self._distance_cache[start]

        distances = {start: 0}
        queue = collections.deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                        and (nx, ny) not in self.env.obstacles
                        and (nx, ny) not in distances):
                    distances[(nx, ny)] = distances[(x, y)] + 1
                    queue.append((nx, ny))

        self._distance_cache[start] = distances
        return distances

    def _sample_new_goal(self):
        self._steps_since_goal = 0

        agent_pos = self._last_obs[:2]
        agent_cell = (int(agent_pos[0]), int(agent_pos[1]))

        agent_dists = self._get_distances_from(agent_cell)

        if self._distance_weights is not None:
            cells_by_dist = {}
            for pos, d in agent_dists.items():
                if pos != agent_cell and d in self._distance_weights:
                    cells_by_dist.setdefault(d, []).append(pos)

            available_dists = [d for d in self._distance_weights if d in cells_by_dist]

            if available_dists:
                dist_probs = np.array([self._distance_weights[d] for d in available_dists])
                dist_probs /= dist_probs.sum()
                chosen_dist = np.random.choice(available_dists, p=dist_probs)
                self.goal = np.array(random.choice(cells_by_dist[chosen_dist]), dtype=np.float32)
                return

        # по BFS-расстоянию, НЕ по Chebyshev
        valid_goals = [
            pos for pos, d in agent_dists.items()
            if pos != agent_cell
               and self.curriculum_min_distance <= d <= self.curriculum_distance
        ]

        if not valid_goals:
            nearby = [pos for pos, d in agent_dists.items()
                      if pos != agent_cell and d <= max(self.curriculum_distance, 3)]
            valid_goals = nearby if nearby else [agent_cell]

        self.goal = np.array(random.choice(valid_goals), dtype=np.float32)

    def _get_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        agent_pos = raw_obs[:2]
        enemy_pos = raw_obs[12:14]
        enemy_phase = raw_obs[15]

        delta = self.goal - agent_pos
        dist = np.linalg.norm(delta)
        max_dist = self.grid_size * np.sqrt(2)

        if dist > 1e-6:
            direction = delta / dist
        else:
            direction = np.zeros(2, dtype=np.float32)

        norm_dist = np.clip(dist / max_dist, 0, 1) * 2 - 1

        local_grid = self._get_local_grid(agent_pos)

        enemy_delta = enemy_pos - agent_pos
        enemy_rel = np.clip(enemy_delta / (self.grid_size * 0.5), -1, 1)

        return np.concatenate([
            direction,
            [norm_dist],
            local_grid,
            enemy_rel,
            [enemy_phase * 2 - 1]
        ]).astype(np.float32)

    def _get_local_grid(self, agent_pos: np.ndarray) -> np.ndarray:
        ax, ay = int(round(agent_pos[0])), int(round(agent_pos[1]))
        r = self.view_radius
        size = int(self.grid_size)
        grid = np.ones((2 * r + 1) ** 2, dtype=np.float32)

        idx = 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = ax + dx, ay + dy
                if (0 <= nx < size and 0 <= ny < size
                        and (nx, ny) not in self.env.obstacles):
                    grid[idx] = -1.0
                idx += 1

        return grid

    def set_curriculum_weights(self, weights: dict):
        self._distance_weights = weights
        if weights:
            self.curriculum_distance = max(weights.keys())
            self.curriculum_min_distance = min(weights.keys())

    def set_curriculum_distance(self, distance: float, min_distance: float = 0.0):
        self.curriculum_distance = np.clip(distance, 1.0, self.max_curriculum_distance)
        self.curriculum_min_distance = min_distance
        self._distance_weights = None

    def get_success_rate(self) -> float:
        """Текущий success rate."""
        if len(self.recent_successes) == 0:
            return 0.0
        return np.mean(self.recent_successes)

    def set_goal(self, goal: np.ndarray):
        self.goal = np.array(goal, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._distance_cache.clear()
        self._sample_new_goal()
        self._steps_since_goal = 0
        return self._get_obs(obs), info

    def step(self, action):
        old_pos = self.last_obs[:2].copy()
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        new_pos = obs[:2]
        hit_wall = np.array_equal(old_pos, new_pos)

        achieved = self.goal_extractor.extract_goal(obs)
        dist = np.linalg.norm(achieved - self.goal)
        is_success = dist < self.success_threshold

        self._steps_since_goal += 1

        max_steps_per_goal = int(self.curriculum_distance * self.GOAL_TIMEOUT_MULTIPLIER) + self.GOAL_TIMEOUT_OFFSET
        goal_timeout = self._steps_since_goal >= max_steps_per_goal

        if is_success:
            reward = self.SUCCESS_REWARD_BASE / self._steps_since_goal ** self.SUCCESS_REWARD_POWER
        elif goal_timeout:
            reward = self.GOAL_TIMEOUT_PENALTY
        else:
            reward = self.STEP_PENALTY

        if hit_wall:
            reward += self.WALL_BUMP_PENALTY

        info['is_success'] = bool(is_success)
        info['env_terminated'] = terminated
        info['env_truncated'] = truncated

        if self.continuous_goals and (is_success or goal_timeout) and not terminated and not truncated:
            self.recent_successes.append(1 if is_success else 0)
            self._sample_new_goal()
            return self._get_obs(obs), reward, False, False, info

        worker_done = is_success
        env_done = (terminated or truncated) if self.continuous_goals else truncated

        if worker_done or env_done:
            self.recent_successes.append(1 if is_success else 0)

        self.env.render_overlays = []
        self.env.render_overlays.append({
            "pos": [round(x) for x in self.goal],
            "color": (0, 255, 100),
            "shape": "cross",
            "size": self.env.cell_size // 4,
        })

        info['raw_env_reward'] = float(env_reward)

        return self._get_obs(obs), reward, bool(worker_done), env_done, info

class DungeonManagerEnv(BaseManagerEnv):
    """Manager для DungeonHeist - стратегическое планирование."""

    WORKER_REWARD_SCALE = 0
    KEY_REWARD = 5.0
    GEM_REWARD = 10.0
    GOLD_REWARD = 4.0
    WIN_REWARD = 15.0
    TRAP_PENALTY = -2
    ENEMY_PENALTY = -0.3
    SUBGOAL_FAIL_PENALTY = -0.2
    PENALTY_PER_STEP = -0.04
    HER_REACHED_REWARD = 0.5
    HER_MISSED_PENALTY = -0.5

    def __init__(self, env, worker_env, worker_model, graph, goal_extractor,
                 max_worker_steps: int = 20, success_threshold: float = 0.8):
        super().__init__(env)
        self._worker_env = worker_env
        self._worker_model = worker_model
        self._success_threshold = success_threshold
        self.graph = graph
        self.goal_extractor = goal_extractor
        self._max_worker_steps = max_worker_steps
        self.current_subgoal = None
        self._consecutive_fails = 0
        self._actual_budget = max_worker_steps
        self.MAX_BUDGET = 50

        self.subgoal_history_len = 4
        self._recent_subgoals = collections.deque(maxlen=self.subgoal_history_len)
        self._steps_since_new_state = 0
        self._episode_step = 0

        self.grid_size = max(env.grid_size)
        goal_low, goal_high = goal_extractor.goal_bounds()
        self.goal_low = goal_low
        self.goal_high = goal_high
        self.worker_view_radius = getattr(worker_env, 'view_radius', 3)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.manager_view_radius = 5
        grid_dim = (2 * self.manager_view_radius + 1) ** 2
        positions_dim = 6 * 2
        enemy_dim = self._worker_env.env.n_enemies * 3
        task_dim = 4
        feedback_dim = 1
        history_dim = self.subgoal_history_len * 2
        novelty_dim = 3
        exploration_dir_dim = 2
        exploration_dist_dim = 1
        norm_budget_dim = 1
        sector_dim = 8
        obs_dim = positions_dim + enemy_dim + task_dim + grid_dim + feedback_dim + sector_dim
        obs_dim += history_dim + novelty_dim + exploration_dir_dim + exploration_dist_dim + norm_budget_dim

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._final_goal = np.zeros(2, dtype=np.float32)
        self._had_key_before = False

    def _get_frontier_direction(self, raw_obs):
        agent_pos = raw_obs[:2]
        key = self.graph.to_key(raw_obs)

        frontier = self.graph.get_frontier_nodes()
        if not frontier or key not in self.graph.nodes:
            return np.zeros(2, dtype=np.float32), 1.0  # "не знаю"

        # BFS-расстояния от агента
        adj = self.graph._get_adjacency()
        dist = self.graph._bfs_from(key, adj)

        # Ближайший frontier-узел
        best_key = None
        best_dist = float('inf')
        for fk in frontier:
            d = dist.get(fk)
            if d is not None and d < best_dist:
                best_dist = d
                best_key = fk

        if best_key is None:
            return np.zeros(2, dtype=np.float32), 1.0

        # Направление в пространстве координат
        target = np.array(best_key[:2], dtype=np.float32) * self.graph.discretization
        delta = target - agent_pos
        norm = np.linalg.norm(delta)

        if norm < 1e-6:
            direction = np.zeros(2, dtype=np.float32)
        else:
            direction = delta / norm

        norm_dist = np.clip(best_dist / 20.0, 0, 1) * 2 - 1

        return direction, norm_dist

    @property
    def worker_env(self):
        return self._worker_env

    @property
    def success_threshold(self):
        return self._success_threshold

    @property
    def final_goal(self) -> Optional[np.ndarray]:
        return self._final_goal

    def set_worker_model(self, model):
        self._worker_model = model

    def _get_sector_novelty(self, agent_pos: np.ndarray) -> np.ndarray:
        ax, ay = agent_pos[0], agent_pos[1]
        r = self.manager_view_radius

        sector_novelty = np.zeros(8, dtype=np.float32)
        sector_count = np.zeros(8, dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = int(round(ax)) + dx, int(round(ay)) + dy
                if (nx, ny) in self.env.obstacles:
                    continue

                angle = np.arctan2(-dy, dx)
                sector = int((angle + np.pi) / (np.pi / 4)) % 8

                coord = (nx, ny)
                if coord not in self.graph._coord_index:
                    novelty = 1.0
                else:
                    total_vc = sum(
                        self.graph.nodes[k]["visit_count"]
                        for k in self.graph._coord_index[coord]
                        if k in self.graph.nodes
                    )
                    novelty = 1.0 / (1.0 + total_vc)

                sector_novelty[sector] += novelty
                sector_count[sector] += 1.0

        mask = sector_count > 0
        sector_novelty[mask] /= sector_count[mask]

        mean = sector_novelty.mean()
        spread = sector_novelty.max() - sector_novelty.min()

        if spread > 1e-6:
            sector_novelty = (sector_novelty - mean) / spread
            sector_novelty = np.clip(sector_novelty * 2, -1, 1)
        else:
            sector_novelty[:] = 0.0

        return sector_novelty

    def _get_obs(self, raw_obs) -> np.ndarray:
        def norm_pos(pos):
            if pos[0] < 0:
                return np.array([-1.0, -1.0])
            return (pos / self.grid_size) * 2 - 1

        local_grid = self._get_local_grid(raw_obs[:2])

        enemy_features = []
        for e in self.env.enemies:
            pos, phase = self.env._enemy_pos_at(e)
            delta = pos - raw_obs[:2]
            rel = np.clip(delta / (self.grid_size * 0.5), -1, 1)
            enemy_features.extend([rel[0], rel[1], phase * 2 - 1])

        while len(enemy_features) < self._worker_env.env.n_enemies * 3:
            enemy_features.extend([0.0, 0.0, 0.0])
        enemy_features = enemy_features[:self._worker_env.env.n_enemies * 3]

        task_phase = []
        task_phase.append(1.0 if raw_obs[14] > 0.5 else -1.0)
        task_phase.append(-1.0 if raw_obs[6] < 0 else 1.0)
        task_phase.append(-1.0 if raw_obs[8] < 0 else 1.0)
        task_phase.append(-1.0 if raw_obs[10] < 0 else 1.0)

        agent_pos = raw_obs[:2]
        history_features = []
        for i in range(self.subgoal_history_len):
            if i < len(self._recent_subgoals):
                sg = self._recent_subgoals[-(i + 1)]
                delta = sg - agent_pos
                rel = np.clip(delta / self.worker_view_radius, -1, 1)
                history_features.extend([rel[0], rel[1]])
            else:
                history_features.extend([0.0, 0.0])

        visit_count = self.graph.visit_count(raw_obs)
        norm_vc = np.clip(1.0 / (visit_count + 1), 0, 1) * 2 - 1

        norm_since_new = np.clip(self._steps_since_new_state / 20.0, 0, 1) * 2 - 1

        max_episode_steps = 1000
        episode_progress = np.clip(self._episode_step / max_episode_steps, 0, 1) * 2 - 1

        novelty_features = [norm_vc, norm_since_new, episode_progress]
        frontier_dir, frontier_dist = self._get_frontier_direction(raw_obs)

        norm_budget = np.clip(
            self._actual_budget / self.MAX_BUDGET, 0, 1
        ) * 2 - 1
        sector_novelty = self._get_sector_novelty(raw_obs[:2])

        return np.concatenate([
            norm_pos(raw_obs[0:2]),
            norm_pos(raw_obs[2:4]),
            norm_pos(raw_obs[4:6]),
            norm_pos(raw_obs[6:8]),
            norm_pos(raw_obs[8:10]),
            norm_pos(raw_obs[10:12]),
            enemy_features,
            task_phase,
            local_grid,
            [np.clip(self._consecutive_fails / 10.0, 0, 1) * 2 - 1],
            history_features,
            novelty_features,
            frontier_dir,
            [frontier_dist],
            [norm_budget],
            sector_novelty
        ]).astype(np.float32)

    def _get_local_grid(self, agent_pos: np.ndarray) -> np.ndarray:
        ax, ay = int(round(agent_pos[0])), int(round(agent_pos[1]))
        r = self.manager_view_radius
        size = self.grid_size
        grid_w = 2 * r + 1

        grid = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = ax + dx, ay + dy
                if not (0 <= nx < size and 0 <= ny < size):
                    grid.append(1.0)  # стена
                elif (nx, ny) in self.env.obstacles:
                    grid.append(1.0)  # стена
                elif (nx, ny) in self.env.trap_positions:
                    grid.append(0.5)  # трап
                else:
                    grid.append(-1.0)  # свободно

        # Поверх отметить врагов
        for e in self.env.enemies:
            pos, _ = self.env._enemy_pos_at(e)
            ex = int(round(pos[0])) - ax + r
            ey = int(round(pos[1])) - ay + r
            if 0 <= ex < grid_w and 0 <= ey < grid_w:
                grid[ey * grid_w + ex] = 0.75  # враг

        return np.array(grid, dtype=np.float32)

    def reset(self, **kwargs):
        # self.graph.reset()
        self._recent_subgoals.clear()
        self._consecutive_fails = 0
        self.current_subgoal = None
        self._steps_since_new_state = 0
        self._episode_step = 0
        _, info = self._worker_env.reset(**kwargs)
        raw_obs = self._worker_env.last_obs
        self._update_final_goal(raw_obs)
        self._had_key_before = False
        return self._get_obs(raw_obs), info

    def _update_final_goal(self, raw_obs: np.ndarray):
        has_key = raw_obs[14] > 0.5
        if has_key:
            self._final_goal = raw_obs[4:6].astype(np.float32)  # door
        else:
            self._final_goal = raw_obs[2:4].astype(np.float32)  # key

    def scale_action_to_goal(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1, 1)
        agent_pos = self._worker_env.last_obs[:2]
        goal = agent_pos + action * self.worker_view_radius
        return np.clip(goal, self.goal_low, self.goal_high)

    def goal_to_action(self, goal: np.ndarray) -> np.ndarray:
        agent_pos = self._worker_env.last_obs[:2]
        offset = goal - agent_pos
        action = offset / max(self.worker_view_radius, 1e-8)
        return np.clip(action, -1, 1).astype(np.float32)

    def _snap_to_reachable(self, goal):
        agent_pos = self._worker_env.last_obs[:2]
        agent_cell = (int(round(agent_pos[0])), int(round(agent_pos[1])))
        agent_dists = self._worker_env._get_distances_from(agent_cell)
        effective_range = min(self._max_worker_steps, int(self._worker_env.curriculum_distance))

        best_pos = None
        best_euclidean = float('inf')

        for cell, bfs_d in agent_dists.items():
            if bfs_d < 1 or bfs_d > effective_range:
                continue
            d = np.linalg.norm(goal - np.array(cell, dtype=np.float32))
            if d < best_euclidean:
                best_euclidean = d
                best_pos = np.array(cell, dtype=np.float32)

        if best_pos is None:
            for cell, bfs_d in sorted(agent_dists.items(), key=lambda x: x[1]):
                if bfs_d >= 1:
                    return np.array(cell, dtype=np.float32)
            return agent_pos.copy()

        return best_pos

    def _snap_to_free(self, goal: np.ndarray) -> np.ndarray:
        """Сдвинуть цель к ближайшей свободной клетке."""
        agent_pos = self._worker_env.last_obs[:2]
        agent_cell = (int(round(agent_pos[0])), int(round(agent_pos[1])))
        dist_map = self._worker_env._get_distances_from(agent_cell)

        best_pos = goal.copy()
        best_dist = float('inf')
        for cell in dist_map.keys():
            d = np.linalg.norm(goal - np.array(cell, dtype=np.float32))
            if d < best_dist:
                best_dist = d
                best_pos = np.array(cell, dtype=np.float32)
        return best_pos

    def _find_reachable_towards(self, agent_cell, target):
        """Найти ближайшую к target достижимую клетку."""
        best_pos = np.array(agent_cell, dtype=np.float32)
        best_score = float('inf')
        dist_from_agent = self._worker_env._get_distances_from(agent_cell)

        for cell, bfs_d in dist_from_agent.items():
            if bfs_d <= self.worker_view_radius:
                d_to_target = np.linalg.norm(np.array(cell) - target)
                if d_to_target < best_score:
                    best_score = d_to_target
                    best_pos = np.array(cell, dtype=np.float32)
        return best_pos

    def step(self, action):
        subgoal = self.scale_action_to_goal(action)
        subgoal = self._snap_to_reachable(subgoal)
        self.current_subgoal = subgoal.copy()
        self._worker_env.set_goal(subgoal)

        self._episode_step += 1
        self._actual_budget = self._max_worker_steps

        start_raw_obs = self._worker_env.last_obs.copy()
        self.graph.add_state(start_raw_obs)

        steps_taken = 0
        env_terminated = False
        env_truncated = False
        reached_subgoal = False

        total_worker_reward = 0.0
        raw_env_reward_total = 0.0
        trap_hits = 0
        enemy_hits = 0
        obs_worker = self._worker_env._get_obs(start_raw_obs)

        for _ in range(self._max_worker_steps):
            steps_taken += 1
            action_w, _ = self._worker_model.predict(obs_worker, deterministic=True)
            if isinstance(self._worker_env.action_space, gym.spaces.Discrete):
                action_w = int(action_w)

            next_obs, r_w, d_w, trunc_w, info_w = self._worker_env.step(action_w)
            total_worker_reward += float(r_w)
            raw_env_reward_total += info_w.get('raw_env_reward', 0.0)

            self.env.render()

            if info_w.get('hit_trap', False): trap_hits += 1
            if info_w.get('hit_enemy', False): enemy_hits += 1
            if info_w.get('env_terminated', False): env_terminated = True
            if info_w.get('env_truncated', False): env_truncated = True
            if d_w: reached_subgoal = True

            obs_worker = next_obs
            if d_w or env_terminated or env_truncated:
                break
            self.graph.add_state(self._worker_env.last_obs)

        if reached_subgoal:
            self._consecutive_fails = 0
        else:
            self._consecutive_fails += 1

        end_raw_obs = self._worker_env.last_obs
        end_agent_pos = end_raw_obs[:2]

        is_new = self.graph.is_new_state(end_raw_obs)
        vc = self.graph.visit_count(end_raw_obs)

        self._update_final_goal(end_raw_obs)

        if is_new:
            self._steps_since_new_state = 0
        else:
            self._steps_since_new_state += 1

        self._recent_subgoals.append(subgoal.copy())

        has_key = end_raw_obs[14] > 0.5
        door_pos = end_raw_obs[4:6]
        at_door = np.linalg.norm(end_agent_pos - door_pos) < self._success_threshold

        reward = total_worker_reward * self.WORKER_REWARD_SCALE

        event_reward = 0.0

        if has_key and not self._had_key_before:
            reward += self.KEY_REWARD
            event_reward += self.KEY_REWARD
            self._had_key_before = True

        gem_before, gem_after = start_raw_obs[6:8], end_raw_obs[6:8]
        if gem_before[0] >= 0 and gem_after[0] < 0:
            reward += self.GEM_REWARD
            event_reward += self.GEM_REWARD

        for i in range(2):
            gb = start_raw_obs[8 + i * 2: 10 + i * 2]
            ga = end_raw_obs[8 + i * 2: 10 + i * 2]
            if gb[0] >= 0 and ga[0] < 0:
                reward += self.GOLD_REWARD
                event_reward += self.GOLD_REWARD

        if at_door and has_key:
            reward += self.WIN_REWARD
            event_reward += self.WIN_REWARD
            env_terminated = True

        reward += trap_hits * self.TRAP_PENALTY
        reward += enemy_hits * self.ENEMY_PENALTY
        event_reward += trap_hits * self.TRAP_PENALTY
        event_reward += enemy_hits * self.ENEMY_PENALTY

        if not reached_subgoal:
            reward += self.SUBGOAL_FAIL_PENALTY

        reward += self.PENALTY_PER_STEP * steps_taken

        self.graph.add_transition(start_raw_obs, end_raw_obs, event_reward)

        step_info = {
            'steps_taken': steps_taken, 'reached_subgoal': reached_subgoal,
            'has_key': has_key, 'trap_hits': trap_hits, 'enemy_hits': enemy_hits,
            'exploration_new': int(is_new), 'visit_count': vc,
            'steps_since_new': self._steps_since_new_state,
            'raw_env_reward': raw_env_reward_total
        }
        return self._get_obs(end_raw_obs), reward, env_terminated, env_truncated, step_info

    def get_achieved_goal(self, raw_obs: np.ndarray) -> np.ndarray:
        return raw_obs[:2].copy()

    def compute_her_reward(self, start_raw: np.ndarray, end_raw: np.ndarray,
                           new_goal: np.ndarray) -> Tuple[float, bool]:
        """HER reward — простой sparse signal, масштабирование делает HERBuffer."""
        agent_pos = end_raw[:2]
        dist = np.linalg.norm(agent_pos - new_goal)
        reached = dist < self._success_threshold

        if reached:
            return self.HER_REACHED_REWARD, False
        else:
            return self.HER_MISSED_PENALTY, False

    def relabel_obs_for_her(self, obs: np.ndarray, raw_obs: np.ndarray,
                            new_goal: np.ndarray) -> np.ndarray:
        has_key = raw_obs[14] > 0.5
        new_obs = obs.copy()
        goal_norm = (new_goal / self.grid_size) * 2 - 1

        if not has_key:
            new_obs[2:4] = goal_norm
        else:
            new_obs[4:6] = goal_norm

        return new_obs

class DungeonHeistHardEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    STEP_PENALTY = -0.04
    TRAP_PENALTY = -2
    ENEMY_PENALTY = -0.3
    KEY_REWARD = 5
    GEM_REWARD = 10
    GOLD_REWARD = 4
    WIN_REWARD = 15

    def __init__(
        self,
        size: int = 23,
        max_steps: int = 1400,
        enemy_speed: float = 0.75,
        render_mode: Optional[str] = None,
        maze_braid_prob: float = 0.06,
        traps_count: int = 16,
        enemy_hit_radius: float = 1.35,
        n_enemies: int = 3,
        enemy_speed_jitter: float = 0.25,
        enemy_waypoints: int = 10,
        cell_size: int = 32,
        dungeon_seed: int = 42,
        maze: bool = True
    ):
        super().__init__()
        assert size >= 15
        if size % 2 == 0:
            size += 1

        self._dungeon_seed = dungeon_seed
        self.maze = maze
        self.grid_size = (size, size)
        self.max_steps = max_steps
        self.enemy_speed = float(enemy_speed)
        self.enemy_hit_radius = float(enemy_hit_radius)
        self.render_mode = render_mode
        self.render_overlays = []

        self.cell_size = cell_size
        self.window = None
        self.clock = None

        self.maze_braid_prob = float(maze_braid_prob)
        self.traps_count = int(traps_count)
        self.n_enemies = int(n_enemies)
        self.enemy_speed_jitter = float(enemy_speed_jitter)
        self.enemy_waypoints = int(enemy_waypoints)

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=float(size), shape=(16,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self.moves = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

        self.door_pos = np.array([2.0, 2.0], dtype=np.float32)
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.key_pos = np.array([-1.0, -1.0], dtype=np.float32)
        self.gem_pos = np.array([-1.0, -1.0], dtype=np.float32)
        self.gold_pos = [
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([-1.0, -1.0], dtype=np.float32),
        ]

        self.has_key = False
        self.has_gem = False
        self.gold_collected = [False, False]
        self.current_step = 0

        self.enemies: List[dict] = []
        self.enemy_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.enemy_phase = 0.0

        self.trap_positions: set = set()
        self.obstacles: set = set()
        self.rooms: Dict[str, List[Tuple[int, int]]] = {}
        self.passages: List[Tuple[int, int]] = []
        self._protected_cells: set = set()

        self._frame_buffer = None
        self._generate_dungeon()

    def _generate_dungeon(self):
        py_state = random.getstate()
        np_state = np.random.get_state()
        random.seed(self._dungeon_seed)
        np.random.seed(self._dungeon_seed)

        try:
            size = self.grid_size[0]
            mid = size // 2

            self.obstacles = set()

            for i in range(size):
                self.obstacles.update([(i, 0), (i, size - 1), (0, i), (size - 1, i)])

            for j in range(size):
                self.obstacles.add((mid, j))
            for i in range(size):
                self.obstacles.add((i, mid))

            self.passages = [
                (mid, mid // 2),
                (mid, mid + mid // 2),
                (mid // 2, mid),
                (mid + mid // 2, mid),
            ]
            for p in self.passages:
                self.obstacles.discard(p)

            TL = (1, mid - 1, 1, mid - 1)
            TR = (mid + 1, size - 2, 1, mid - 1)
            BL = (1, mid - 1, mid + 1, size - 2)
            BR = (mid + 1, size - 2, mid + 1, size - 2)

            tl_e1 = (mid - 1, mid // 2)
            tl_e2 = (mid // 2, mid - 1)
            tr_e1 = (mid + 1, mid // 2)
            tr_e2 = (mid + mid // 2, mid - 1)
            bl_e1 = (mid // 2, mid + 1)
            bl_e2 = (mid - 1, mid + mid // 2)
            br_e1 = (mid + mid // 2, mid + 1)
            br_e2 = (mid + 1, mid + mid // 2)

            if self.maze:
                self._fill_region_with_walls(*TL)
                self._fill_region_with_walls(*TR)
                self._fill_region_with_walls(*BL)
                self._fill_region_with_walls(*BR)

                self._carve_maze_in_quadrant(*TL, start_cell=tl_e1, braid_prob=self.maze_braid_prob)
                self._carve_maze_in_quadrant(*TR, start_cell=tr_e1, braid_prob=self.maze_braid_prob)
                self._carve_maze_in_quadrant(*BL, start_cell=bl_e1, braid_prob=self.maze_braid_prob)
                self._carve_maze_in_quadrant(*BR, start_cell=br_e1, braid_prob=self.maze_braid_prob)

            for (a, b, bounds) in [
                (tl_e1, tl_e2, TL),
                (tr_e1, tr_e2, TR),
                (bl_e1, bl_e2, BL),
                (br_e1, br_e2, BR),
            ]:
                self._open_cell(a)
                self._open_cell(b)
                self._carve_corridor(a, b, bounds)

            door_cell = (int(self.door_pos[0]), int(self.door_pos[1]))
            self._open_cell(door_cell)
            self._carve_corridor(door_cell, tl_e2, TL)

            self._protected_cells = set(self.passages)
            self._protected_cells.update([
                tl_e1, tl_e2, tr_e1, tr_e2,
                bl_e1, bl_e2, br_e1, br_e2,
                door_cell,
            ])
            for px, py in list(self.passages):
                for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    self._protected_cells.add((px + ddx, py + ddy))

            self.rooms = {
                "start": self._free_cells_in_bounds(*TL),
                "traps": self._free_cells_in_bounds(*TR),
                "patrol": self._free_cells_in_bounds(*BL),
                "vault": self._free_cells_in_bounds(*BR),
            }
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

        self._spawn_traps()
        self._build_enemies_global_paths()

    def _fill_region_with_walls(self, x0, x1, y0, y1):
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.obstacles.add((x, y))

    def _open_cell(self, cell: Tuple[int, int]):
        self.obstacles.discard((int(cell[0]), int(cell[1])))

    def _nearest_odd_cell(self, cell: Tuple[int, int], bounds) -> Tuple[int, int]:
        x0, x1, y0, y1 = bounds
        x, y = int(cell[0]), int(cell[1])
        x = int(np.clip(x, x0, x1))
        y = int(np.clip(y, y0, y1))
        if x % 2 == 0:
            x = x + 1 if x + 1 <= x1 else x - 1
        if y % 2 == 0:
            y = y + 1 if y + 1 <= y1 else y - 1
        x = int(np.clip(x, x0, x1))
        y = int(np.clip(y, y0, y1))
        return x, y

    def _carve_maze_in_quadrant(self, x0, x1, y0, y1,
                                 start_cell: Tuple[int, int],
                                 braid_prob: float = 0.05):
        bounds = (x0, x1, y0, y1)
        sx, sy = self._nearest_odd_cell(start_cell, bounds)
        self._open_cell((sx, sy))

        stack = [(sx, sy)]
        visited = {(sx, sy)}
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        def in_bounds(xx, yy):
            return (x0 <= xx <= x1) and (y0 <= yy <= y1)

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if in_bounds(nx, ny) and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))

            if not neighbors:
                stack.pop()
                continue

            nx, ny, dx, dy = random.choice(neighbors)
            wx, wy = cx + dx // 2, cy + dy // 2
            self._open_cell((wx, wy))
            self._open_cell((nx, ny))
            visited.add((nx, ny))
            stack.append((nx, ny))

        if braid_prob > 0:
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    if (x, y) in self.obstacles:
                        free_n = sum(
                            1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                            if (x + dx, y + dy) not in self.obstacles
                        )
                        if free_n >= 2 and random.random() < braid_prob:
                            self._open_cell((x, y))

    def _carve_corridor(self, a: Tuple[int, int], b: Tuple[int, int], bounds):
        x0, x1, y0, y1 = bounds
        x, y = int(a[0]), int(a[1])
        bx, by = int(b[0]), int(b[1])
        self._open_cell((x, y))

        while x != bx:
            x += 1 if bx > x else -1
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._open_cell((x, y))
        while y != by:
            y += 1 if by > y else -1
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._open_cell((x, y))

    def _free_cells_in_bounds(self, x0, x1, y0, y1) -> List[Tuple[int, int]]:
        return [
            (x, y)
            for x in range(x0, x1 + 1)
            for y in range(y0, y1 + 1)
            if (x, y) not in self.obstacles
        ]

    def _all_free_cells(self) -> List[Tuple[int, int]]:
        size = self.grid_size[0]
        return [
            (x, y) for x in range(size) for y in range(size)
            if (x, y) not in self.obstacles
        ]

    def _bfs_dist(self, start: Tuple[int, int],
                  allowed: Optional[set] = None) -> Dict[Tuple[int, int], int]:
        q = collections.deque([start])
        dist = {start: 0}
        while q:
            x, y = q.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                cell = (x + dx, y + dy)
                if cell in dist or cell in self.obstacles:
                    continue
                if allowed is not None and cell not in allowed:
                    continue
                dist[cell] = dist[(x, y)] + 1
                q.append(cell)
        return dist

    def _pick_deep_cell(self, room_cells: List[Tuple[int, int]],
                        entrance: Tuple[int, int],
                        quantile: float = 0.85) -> Tuple[int, int]:
        if not room_cells:
            return (int(entrance[0]), int(entrance[1]))

        room_set = set(room_cells)
        ent = entrance if entrance in room_set else random.choice(room_cells)

        dist = self._bfs_dist(ent, allowed=room_set)
        if not dist:
            return random.choice(room_cells)

        dvals = np.array(list(dist.values()), dtype=np.int32)
        thr = int(np.quantile(dvals, quantile))
        deep = [c for c, d in dist.items() if d >= thr]
        return random.choice(deep) if deep else random.choice(room_cells)

    def _pick_unique_deep_cells(
        self,
        room_cells: List[Tuple[int, int]],
        entrance: Tuple[int, int],
        count: int,
        quantile: float = 0.55,
        exclude: Optional[set] = None,
    ) -> List[Tuple[int, int]]:
        """Выбрать count УНИКАЛЬНЫХ глубоких клеток, избегая exclude."""
        if not room_cells:
            return []

        room_set = set(room_cells)
        ent = entrance if entrance in room_set else random.choice(room_cells)

        dist = self._bfs_dist(ent, allowed=room_set)
        if not dist:
            return []

        dvals = np.array(list(dist.values()), dtype=np.int32)
        thr = int(np.quantile(dvals, quantile))
        candidates = [c for c, d in dist.items() if d >= thr]

        if exclude:
            candidates = [c for c in candidates if c not in exclude]

        # Если кандидатов меньше count, снижаем порог
        if len(candidates) < count:
            lower_thr = int(np.quantile(dvals, max(0.1, quantile - 0.3)))
            extra = [c for c, d in dist.items()
                     if d >= lower_thr and c not in set(candidates)]
            if exclude:
                extra = [c for c in extra if c not in exclude]
            candidates.extend(extra)

        # Если всё ещё мало — берём все свободные (кроме exclude)
        if len(candidates) < count:
            all_available = [c for c in room_cells if c not in set(candidates)]
            if exclude:
                all_available = [c for c in all_available if c not in exclude]
            candidates.extend(all_available)

        random.shuffle(candidates)
        return candidates[:count]

    def _shortest_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                       allowed: Optional[set] = None) -> Optional[List[Tuple[int, int]]]:
        if start == goal:
            return [start]
        q = collections.deque([start])
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        while q:
            x, y = q.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                cell = (x + dx, y + dy)
                if cell in prev or cell in self.obstacles:
                    continue
                if allowed is not None and cell not in allowed:
                    continue
                prev[cell] = (x, y)
                if cell == goal:
                    q.clear()
                    break
                q.append(cell)
        if goal not in prev:
            return None
        cur = goal
        path = [cur]
        while prev[cur] is not None:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def _build_enemies_global_paths(self):
        self.enemies = []
        base_rng = random.Random(12345)

        for _ in range(self.n_enemies):
            rng = random.Random(base_rng.randint(0, 10 ** 9))
            path = self._build_one_global_enemy_path(rng)
            self.enemies.append({
                "path": path,
                "progress": 0.0,
                "speed": self.enemy_speed,
            })

    def _build_one_global_enemy_path(self, rng: random.Random) -> List[Tuple[int, int]]:
        free_all = self._all_free_cells()
        free_set = set(free_all)
        if not free_all:
            return [(1, 1)]

        door_cell = (int(self.door_pos[0]), int(self.door_pos[1]))
        if door_cell not in free_set:
            door_cell = rng.choice(free_all)

        size = self.grid_size[0]
        mid = size // 2
        entrances = [
            (mid - 1, mid // 2),
            (mid + 1, mid // 2),
            (mid // 2, mid + 1),
            (mid + 1, mid + mid // 2),
        ]
        entrances = [c for c in entrances if c in free_set]
        if not entrances:
            entrances = [door_cell]

        waypoints: List[Tuple[int, int]] = [door_cell]

        for name in ["start", "traps", "patrol", "vault"]:
            room = self.rooms.get(name, [])
            if room:
                ent = rng.choice(entrances)
                waypoints.append(self._pick_deep_cell(room, ent, quantile=0.90))

        anchor = rng.choice(free_all)
        dist = self._bfs_dist(anchor, allowed=free_set)
        if dist:
            dvals = np.array(list(dist.values()), dtype=np.int32)
            thr = int(np.quantile(dvals, 0.85))
            far_cells = [c for c, d in dist.items() if d >= thr]
            rng.shuffle(far_cells)
            extra = far_cells[: max(0, self.enemy_waypoints - len(waypoints))]
            waypoints.extend(extra)

        while len(waypoints) < max(6, self.enemy_waypoints):
            waypoints.append(rng.choice(free_all))

        middle = waypoints[1:]
        rng.shuffle(middle)
        waypoints = [waypoints[0]] + middle

        path: List[Tuple[int, int]] = []
        for i in range(len(waypoints)):
            a = waypoints[i]
            b = waypoints[(i + 1) % len(waypoints)]
            seg = self._shortest_path(a, b, allowed=free_set)
            if seg is None:
                continue
            if path and seg[0] == path[-1]:
                path.extend(seg[1:])
            else:
                path.extend(seg)

        if len(path) < 4:
            shuffled = free_all[:]
            rng.shuffle(shuffled)
            path = shuffled[:200]

        return path

    def _enemy_pos_at(self, enemy: dict) -> Tuple[np.ndarray, float]:
        path = enemy["path"]
        n = len(path)
        if n == 0:
            return np.array([0.0, 0.0], dtype=np.float32), 0.0

        prog = enemy["progress"] % n
        i0 = int(prog)
        t = float(prog - i0)
        i1 = (i0 + 1) % n

        x0, y0 = path[i0]
        x1, y1 = path[i1]
        pos = np.array([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], dtype=np.float32)
        phase = float(prog / n)
        return pos, phase

    def _update_enemies(self):
        for e in self.enemies:
            n = len(e["path"])
            if n > 0:
                e["progress"] = (e["progress"] + e["speed"]) % n
        self._update_nearest_enemy_cache()

    def _update_nearest_enemy_cache(self):
        best_d = float("inf")
        best_pos = np.array([0.0, 0.0], dtype=np.float32)
        best_phase = 0.0

        for e in self.enemies:
            pos, phase = self._enemy_pos_at(e)
            d = float(np.linalg.norm(self.agent_pos - pos))
            if d < best_d:
                best_d = d
                best_pos = pos
                best_phase = phase

        self.enemy_pos = best_pos.astype(np.float32)
        self.enemy_phase = float(best_phase)

    def _check_enemy_hit(self) -> bool:
        for e in self.enemies:
            pos, _ = self._enemy_pos_at(e)
            if np.linalg.norm(self.agent_pos - pos) < self.enemy_hit_radius:
                return True
        return False

    def _spawn_traps(self):
        """Расставить traps_count УНИКАЛЬНЫХ ловушек в traps-комнате."""
        traps_room = self.rooms["traps"]
        self.trap_positions = set()

        if not traps_room:
            return

        size = self.grid_size[0]
        mid = size // 2
        entrance_tr = (mid + 1, mid // 2)

        trap_cells = self._pick_unique_deep_cells(
            room_cells=traps_room,
            entrance=entrance_tr,
            count=self.traps_count,
            quantile=0.35,  # ниже порог — больше пул кандидатов
            exclude=self._protected_cells,
        )

        self.trap_positions = set(trap_cells)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # self._generate_dungeon()

        size = self.grid_size[0]
        mid = size // 2

        # Ловушки
        # self._spawn_traps()

        # Агент далеко от двери
        start_room = self.rooms["start"]
        if start_room:
            door_cell = (int(self.door_pos[0]), int(self.door_pos[1]))
            if door_cell in set(start_room):
                dist = self._bfs_dist(door_cell, allowed=set(start_room))
                if dist:
                    dvals = np.array(list(dist.values()), dtype=np.int32)
                    thr = int(np.quantile(dvals, 0.80))
                    far = [c for c, d in dist.items() if d >= thr]
                    self.agent_pos = np.array(
                        random.choice(far if far else start_room), dtype=np.float32
                    )
                else:
                    self.agent_pos = np.array(random.choice(start_room), dtype=np.float32)
            else:
                self.agent_pos = np.array(random.choice(start_room), dtype=np.float32)
        else:
            self.agent_pos = np.array([1.0, 1.0], dtype=np.float32)

        # Ключ глубоко в vault
        vault_room = self.rooms["vault"]
        entrance_br = (mid + 1, mid + mid // 2)
        if vault_room:
            kcell = self._pick_deep_cell(vault_room, entrance_br, quantile=0.90)
            self.key_pos = np.array(kcell, dtype=np.float32)
        else:
            self.key_pos = np.array([mid + 2.0, mid + 2.0], dtype=np.float32)

        # Гем глубоко в traps, не на ловушке
        traps_room = self.rooms["traps"]
        if traps_room:
            entrance_tr = (mid + 1, mid // 2)
            safe = self._pick_unique_deep_cells(
                room_cells=traps_room,
                entrance=entrance_tr,
                count=1,
                quantile=0.85,
                exclude=self.trap_positions | self._protected_cells,
            )
            if safe:
                self.gem_pos = np.array(safe[0], dtype=np.float32)
            else:
                fallback = [c for c in traps_room if c not in self.trap_positions]
                self.gem_pos = np.array(
                    random.choice(fallback) if fallback else random.choice(traps_room),
                    dtype=np.float32,
                )
        else:
            self.gem_pos = np.array([-1.0, -1.0], dtype=np.float32)

        # Золото в vault, не рядом с ключом
        self.gold_pos = []
        if vault_room:
            exclude_gold = {tuple(self.key_pos.astype(int))}
            # клетки достаточно далеко от ключа
            vault_far = [
                c for c in vault_room
                if np.linalg.norm(np.array(c, dtype=np.float32) - self.key_pos) > 3.0
                   and c not in exclude_gold
            ]
            if len(vault_far) < 2:
                vault_far = [c for c in vault_room if c not in exclude_gold]

            if len(vault_far) >= 2:
                picks = random.sample(vault_far, 2)
            elif len(vault_far) == 1:
                picks = vault_far[:]
            else:
                picks = random.sample(vault_room, min(2, len(vault_room)))

            for c in picks:
                self.gold_pos.append(np.array(c, dtype=np.float32))

        while len(self.gold_pos) < 2:
            self.gold_pos.append(np.array([-1.0, -1.0], dtype=np.float32))

        # Флаги
        self.has_key = False
        self.has_gem = False
        self.gold_collected = [False, False]

        # Враги
        for e in self.enemies:
            n = len(e["path"])
            e["progress"] = random.random() * max(1, n)
            jitter = (random.random() * 2 - 1) * self.enemy_speed_jitter
            e["speed"] = max(0.05, self.enemy_speed * (1.0 + jitter))

        self._update_nearest_enemy_cache()

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        gem_pos = self.gem_pos if not self.has_gem else np.array([-1.0, -1.0], dtype=np.float32)
        gold1 = self.gold_pos[0] if not self.gold_collected[0] else np.array([-1.0, -1.0], dtype=np.float32)
        gold2 = self.gold_pos[1] if not self.gold_collected[1] else np.array([-1.0, -1.0], dtype=np.float32)
        key_pos = self.key_pos if not self.has_key else np.array([-1.0, -1.0], dtype=np.float32)

        return np.concatenate([
            self.agent_pos,                          # 0:2
            key_pos,                                 # 2:4
            self.door_pos,                           # 4:6
            gem_pos,                                 # 6:8
            gold1,                                   # 8:10
            gold2,                                   # 10:12
            self.enemy_pos,                          # 12:14
            [1.0 if self.has_key else 0.0],          # 14
            [self.enemy_phase],                      # 15
        ]).astype(np.float32)

    def step(self, action: int):
        self.current_step += 1
        reward = self.STEP_PENALTY

        dx, dy = self.moves[int(action)]
        next_pos = self.agent_pos + np.array([dx, dy], dtype=np.float32)
        next_cell = (int(next_pos[0]), int(next_pos[1]))
        size = self.grid_size[0]

        if (0 <= next_pos[0] < size and 0 <= next_pos[1] < size
                and next_cell not in self.obstacles):
            self.agent_pos = next_pos

        self._update_enemies()

        agent_cell = (int(self.agent_pos[0]), int(self.agent_pos[1]))

        hit_trap = agent_cell in self.trap_positions
        if hit_trap:
            reward += self.TRAP_PENALTY

        hit_enemy = self._check_enemy_hit()
        if hit_enemy:
            reward += self.ENEMY_PENALTY

        picked_key = False
        if not self.has_key and self.key_pos[0] >= 0:
            if np.linalg.norm(self.agent_pos - self.key_pos) < 0.8:
                self.has_key = True
                picked_key = True
                reward += self.KEY_REWARD

        picked_gem = False
        if not self.has_gem and self.gem_pos[0] >= 0:
            if np.linalg.norm(self.agent_pos - self.gem_pos) < 0.8:
                self.has_gem = True
                picked_gem = True
                reward += self.GEM_REWARD

        picked_gold = [False, False]
        for i, (collected, pos) in enumerate(zip(self.gold_collected, self.gold_pos)):
            if not collected and pos[0] >= 0:
                if np.linalg.norm(self.agent_pos - pos) < 0.8:
                    self.gold_collected[i] = True
                    picked_gold[i] = True
                    reward += self.GOLD_REWARD

        at_door = np.linalg.norm(self.agent_pos - self.door_pos) < 0.8
        terminated = bool(at_door and self.has_key)
        if terminated:
            reward += self.WIN_REWARD

        truncated = self.current_step >= self.max_steps

        info = {
            "has_key": self.has_key,
            "has_gem": self.has_gem,
            "gold_collected": sum(self.gold_collected),
            "hit_trap": hit_trap,
            "hit_enemy": hit_enemy,
            "picked_key": picked_key,
            "picked_gem": picked_gem,
            "picked_gold": picked_gold,
            "at_door": at_door,
        }

        # self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
        if self.window is None:
            pygame.init()
            size = (self.grid_size[0] * self.cell_size, self.grid_size[1] * self.cell_size)
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(size)
            else:
                self.window = pygame.Surface(size)
            self.clock = pygame.time.Clock()

        cs = self.cell_size
        pad1 = max(1, cs // 8)
        pad2 = max(1, cs // 5)
        pad3 = max(2, cs // 4)

        self.window.fill((40, 40, 50))
        if self.render_mode == "human":
            pygame.event.pump()

        for obs in self.obstacles:
            pygame.draw.rect(
                self.window, (80, 80, 90),
                (obs[0] * cs, obs[1] * cs, cs, cs),
            )

        for trap in self.trap_positions:
            pygame.draw.rect(
                self.window, (150, 50, 50),
                (trap[0] * cs + pad1, trap[1] * cs + pad1,
                 cs - 2 * pad1, cs - 2 * pad1),
            )

        pygame.draw.rect(
            self.window, (139, 90, 43),
            (int(self.door_pos[0] * cs) + pad2,
             int(self.door_pos[1] * cs) + pad2,
             cs - 2 * pad2, cs - 2 * pad2),
        )

        if not self.has_key and self.key_pos[0] >= 0:
            pygame.draw.circle(
                self.window, (255, 215, 0),
                (int(self.key_pos[0] * cs + cs // 2),
                 int(self.key_pos[1] * cs + cs // 2)),
                max(2, cs // 4),
            )

        if not self.has_gem and self.gem_pos[0] >= 0:
            cx = int(self.gem_pos[0] * cs + cs // 2)
            cy = int(self.gem_pos[1] * cs + cs // 2)
            hs = max(2, cs // 2 - pad2)
            pygame.draw.polygon(self.window, (0, 255, 255), [
                (cx, cy - hs), (cx + hs, cy), (cx, cy + hs), (cx - hs, cy),
            ])

        for i, (collected, pos) in enumerate(zip(self.gold_collected, self.gold_pos)):
            if not collected and pos[0] >= 0:
                pygame.draw.circle(
                    self.window, (255, 165, 0),
                    (int(pos[0] * cs + cs // 2),
                     int(pos[1] * cs + cs // 2)),
                    max(2, cs // 5),
                )

        for e in self.enemies:
            pos, _ = self._enemy_pos_at(e)
            pygame.draw.circle(
                self.window, (180, 0, 180),
                (int(pos[0] * cs + cs // 2),
                 int(pos[1] * cs + cs // 2)),
                max(2, cs // 3),
            )

        agent_rect = (
            int(self.agent_pos[0] * cs) + pad3,
            int(self.agent_pos[1] * cs) + pad3,
            cs - 2 * pad3, cs - 2 * pad3,
        )
        pygame.draw.rect(self.window, (50, 120, 220), agent_rect)
        if self.has_key:
            pygame.draw.rect(self.window, (255, 215, 0), agent_rect, max(1, pad1))

        for ov in self.render_overlays:
            px = int(ov["pos"][0] * cs + cs // 2)
            py = int(ov["pos"][1] * cs + cs // 2)
            sz = ov.get("size", max(2, cs // 3))
            color = ov.get("color", (255, 255, 255))
            shape = ov.get("shape", "circle")
            line_w = max(1, cs // 10)

            if shape == "circle":
                s = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, 140), (sz, sz), sz)
                self.window.blit(s, (px - sz, py - sz))
            elif shape == "cross":
                pygame.draw.line(self.window, color, (px - sz, py - sz), (px + sz, py + sz), line_w)
                pygame.draw.line(self.window, color, (px - sz, py + sz), (px + sz, py - sz), line_w)
            elif shape == "rect":
                s = pygame.Surface((sz * 2, sz * 2), pygame.SRCALPHA)
                pygame.draw.rect(s, (*color, 140), (0, 0, sz * 2, sz * 2), line_w)
                self.window.blit(s, (px - sz, py - sz))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            frame = pygame.surfarray.array3d(self.window).transpose(1, 0, 2).copy()
            if self._frame_buffer is not None:
                self._frame_buffer.append(frame)
            return frame

class SimpleGoalExtractor(GoalExtractor):
    def __init__(self, obs_space: gym.spaces.Box):
        super().__init__(obs_space)
        self.coord_dim = 2
        self._dim = self.coord_dim
        self._low = obs_space.low[:self.coord_dim].astype(np.float32)
        self._high = (obs_space.high[:self.coord_dim] - 1).astype(np.float32)

    def extract_goal(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if obs.ndim == 1:
            return obs[:self.coord_dim].astype(np.float32)
        return obs[:, :self.coord_dim].astype(np.float32)

    def inject_goal(self, raw_obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        result = raw_obs.copy()
        result[:self.coord_dim] = goal
        return result

    def goal_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._low, self._high

def dungeon_context_fn(raw_obs):
    return int(raw_obs[14] > 0.5) if len(raw_obs) > 14 else 0
