import os
import json
import time
import argparse
import datetime
import numpy as np
import gymnasium as gym
import imageio
import gc
import collections
from tqdm.auto import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from sgghrl import SGGHRLAgent, setup_logging
from sgghrl.logging import logger
from sgghrl.training.callbacks import (
    CallbackList, RollingEpisodeStatsCallback, ProgressPrinterCallback,
    ManagerEpsilonGreedyExplorationCallback, SACTrainCallback,
    GraphExplorationBonusCallback,
    AdaptiveWorkerBudgetCallback, SetEnvAttrCallback,
    WorkerCurriculumCallback, WorkerBestCheckpointCallback,
    ManagerLRDecayOnPlateauCallback, SGGHRLCallback,
    ManagerEvalCallback, ManagerBestCheckpointOnEvalCallback,
    CheckpointCallback, DeltaDistanceShapingCallback
)
from sgghrl.core.inference import GraphPlannerStrategy, PolicyOnlyStrategy
from sgghrl.training.callbacks_her import HERCallback
from sgghrl.core.base import BaseManagerEnv
from .env_setup import (
    DungeonHeistHardEnv, DungeonWorkerEnv, DungeonManagerEnv,
    SimpleGoalExtractor, dungeon_context_fn
)


ENV_CONFIGS = {
    "hard-16": dict(
        size=16, max_steps=500, enemy_speed=0.2, traps_count=4, n_enemies=2,
        enemy_waypoints=50, enemy_hit_radius=1.35, enemy_speed_jitter=0.30,
        cell_size=16, dungeon_seed=42, maze=True,
    ),
    "hard-25": dict(
        size=25, max_steps=800, enemy_speed=0.2, traps_count=4, n_enemies=2,
        enemy_waypoints=50, enemy_hit_radius=1.35, enemy_speed_jitter=0.30,
        cell_size=16, dungeon_seed=42, maze=True,
    ),
    "easy-16": dict(
        size=16, max_steps=300, enemy_speed=0, traps_count=0, n_enemies=0,
        enemy_waypoints=0, enemy_hit_radius=1.35, enemy_speed_jitter=0,
        cell_size=16, dungeon_seed=42, maze=False,
    ),
    "easy-25": dict(
        size=25, max_steps=500, enemy_speed=0, traps_count=0, n_enemies=0,
        enemy_waypoints=0, enemy_hit_radius=1.35, enemy_speed_jitter=0,
        cell_size=16, dungeon_seed=42, maze=False,
    ),
}

WORKER_TIMESTEPS = 250000  # 250000
MANAGER_TIMESTEPS_BY_SIZE = {
    16: 80000,  # 80000
    25: 120000,  # 120000
}
PPO_TIMESTEPS_BY_SIZE = {
    16: 500000,  # 500000
    25: 1200000,  # 1200000
}

SHARED_NET_ARCH = [96, 96]

PPO_LOG_INTERVAL = 1000

WORKER_KWARGS = dict(
    learning_rate=8e-5,
    n_steps=512,
    batch_size=128,
    n_epochs=8,
    gamma=0.95,
    clip_range=0.2,
    ent_coef=0.03,
    max_grad_norm=0.5,
    policy_kwargs={"net_arch": [96, 48]}
)
MANAGER_KWARGS = dict(
    learning_rate=3e-4,
    buffer_size=80000,
    policy_kwargs={"net_arch": SHARED_NET_ARCH},
    gamma=0.98,
    tau=0.005
)

MAX_WORKER_STEPS = 16
SUCCESS_THRESHOLD = 0.5
MAX_GRAPH_NODES = 5000

EVAL_EPISODES = 20
EVAL_INTERVAL_PPO = 20000
EVAL_INTERVAL_MANAGER = 5000
EVAL_LEARNING_STARTS = 3000

CURRICULUM_STAGES = [
    (1, 0.9),
    (2, 0.9),
    (3, 0.9),
    (4, 0.9),
    (5, 0.9),
    (6, 0.9)
]

GIF_EPISODES = 3
GIF_FPS = 20


def _get_resume_info(env_name, algo, seed):
    """Прочитать прогресс обучения из сохранённого eval JSON."""
    fair_path = f"results/fair_{env_name}_{algo}_seed{seed}.json"
    if not os.path.exists(fair_path):
        return 0, 0, []
    try:
        with open(fair_path, 'r') as f:
            data = json.load(f)
        history = data.get("history", [])
        if not history:
            return 0, 0, []
        last = history[-1]
        trained_steps = last.get("step", 0)
        env_steps = last.get("env_steps_total", WORKER_TIMESTEPS) - WORKER_TIMESTEPS
        return trained_steps, max(env_steps, 0), history
    except (json.JSONDecodeError, KeyError):
        return 0, 0, []


def _ppo_timesteps(env_name: str) -> int:
    size = ENV_CONFIGS[env_name]["size"]
    return PPO_TIMESTEPS_BY_SIZE.get(size, 1200000)


def _manager_timesteps(env_name: str) -> int:
    env_config = ENV_CONFIGS[env_name]
    size = env_config["size"]
    return MANAGER_TIMESTEPS_BY_SIZE.get(size, 120000)


class PPODungeonWrapper(gym.ObservationWrapper):
    """Расширенные наблюдения для PPO — аналогичные manager, но без HRL-фичей."""

    def __init__(self, env: DungeonHeistHardEnv, view_radius: int = 5):
        super().__init__(env)
        self.view_radius = view_radius
        self.grid_size = float(max(env.grid_size))
        grid_dim = (2 * view_radius + 1) ** 2
        # 6 позиций × 2 + enemies + task_phase + local_grid
        obs_dim = 12 + env.n_enemies * 3 + 4 + grid_dim
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def _norm_pos(self, pos):
        if pos[0] < 0:
            return np.array([-1.0, -1.0], dtype=np.float32)
        return (np.asarray(pos, dtype=np.float32) / self.grid_size) * 2 - 1

    def _get_local_grid(self, agent_pos):
        ax, ay = int(round(agent_pos[0])), int(round(agent_pos[1]))
        r = self.view_radius
        size = int(self.grid_size)
        grid_w = 2 * r + 1
        grid = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = ax + dx, ay + dy
                if not (0 <= nx < size and 0 <= ny < size):
                    grid.append(1.0)
                elif (nx, ny) in self.env.obstacles:
                    grid.append(1.0)
                elif (nx, ny) in self.env.trap_positions:
                    grid.append(0.5)
                else:
                    grid.append(-1.0)
        for e in self.env.enemies:
            pos, _ = self.env._enemy_pos_at(e)
            ex = int(round(pos[0])) - ax + r
            ey = int(round(pos[1])) - ay + r
            if 0 <= ex < grid_w and 0 <= ey < grid_w:
                grid[ey * grid_w + ex] = 0.75
        return np.array(grid, dtype=np.float32)

    def observation(self, obs):
        agent_pos = obs[:2]
        enemy_features = []
        for e in self.env.enemies:
            pos, phase = self.env._enemy_pos_at(e)
            delta = pos - agent_pos
            rel = np.clip(delta / (self.grid_size * 0.5), -1, 1)
            enemy_features.extend([rel[0], rel[1], phase * 2 - 1])
        while len(enemy_features) < self.env.n_enemies * 3:
            enemy_features.extend([0.0, 0.0, 0.0])
        enemy_features = enemy_features[:self.env.n_enemies * 3]

        task_phase = [
            1.0 if obs[14] > 0.5 else -1.0,
            -1.0 if obs[6] < 0 else 1.0,
            -1.0 if obs[8] < 0 else 1.0,
            -1.0 if obs[10] < 0 else 1.0,
        ]
        return np.concatenate([
            self._norm_pos(obs[0:2]),  self._norm_pos(obs[2:4]),
            self._norm_pos(obs[4:6]),  self._norm_pos(obs[6:8]),
            self._norm_pos(obs[8:10]), self._norm_pos(obs[10:12]),
            enemy_features, task_phase,
            self._get_local_grid(agent_pos),
        ]).astype(np.float32)


class FairEvaluationCallback(SGGHRLCallback):
    """Записывает в JSON только результаты eval-прогонов. Поддерживает resume."""

    def __init__(self, save_path: str, worker_timesteps: int = 0,
                 step_offset: int = 0, env_steps_offset: int = 0,
                 existing_history: list = None):
        super().__init__()
        self.save_path = save_path
        self._worker_timesteps = worker_timesteps
        self._step_offset = step_offset
        self._env_steps = 0
        self._env_steps_offset = env_steps_offset
        self._ep_original = 0.0
        self.history = list(existing_history) if existing_history else []

    def on_training_start(self, ctx) -> bool:
        self._env_steps = self._env_steps_offset
        self._ep_original = 0.0
        return True

    def after_step(self, ctx) -> bool:
        info = getattr(ctx, 'info', getattr(ctx, 'step_info', {}))
        orig_r = info.get('raw_env_reward', 0.0)
        worker_steps = info.get('steps_taken', 1)
        self._ep_original += orig_r
        self._env_steps += worker_steps
        ctx.original_reward = orig_r
        ctx.episode_original_reward = self._ep_original
        return True

    def on_episode_end(self, ctx) -> bool:
        ctx.episode_original_reward_finished = self._ep_original
        self._ep_original = 0.0
        return True

    def on_eval_end(self, ctx, success_rate: float, avg_reward: float) -> bool:
        entry = {
            "step": ctx.step + self._step_offset,
            "env_steps_total": self._env_steps + self._worker_timesteps,
            "eval_success_rate": success_rate,
            "eval_avg_reward": avg_reward,
            "eval_avg_raw_reward": getattr(ctx, 'last_eval_raw_reward', avg_reward),
        }
        self.history.append(entry)
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump({"history": self.history}, f, indent=2)
        return True

    def on_training_end(self, ctx) -> bool:
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump({"history": self.history}, f, indent=2)
        logger.info("[FairEval] Saved %d eval entries to %s",
                    len(self.history), self.save_path)
        return True


def _cleanup():
    """Очистка RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class StreamingGifWriter:
    def __init__(self, path: str, fps: int = 20):
        self._writer = imageio.get_writer(path, fps=fps, loop=0)
        self.count = 0

    def append(self, frame: np.ndarray):
        self._writer.append_data(frame)
        self.count += 1

    def close(self):
        self._writer.close()

    def __len__(self):
        return self.count


def _get_base_env(env) -> DungeonHeistHardEnv:
    """Найти DungeonHeistHardEnv в цепочке обёрток."""
    cur = env
    for _ in range(10):
        if isinstance(cur, DungeonHeistHardEnv):
            return cur
        if hasattr(cur, 'env'):
            cur = cur.env
        else:
            break
    raise RuntimeError("DungeonHeistHardEnv not found in env chain")


def record_gif(env, policy_fn, path: str,
               n_episodes: int = GIF_EPISODES,
               max_frames: int = 1000):
    """Записать GIF с n эпизодами.

    Args:
        env: среда с render_mode=None (будем вызывать render вручную).
        policy_fn: функция (obs) -> action.
        path: путь к выходному .gif файлу.
        n_episodes: количество эпизодов.
        max_frames: максимальное количество кадров за все эпизоды.
    """
    base_env = _get_base_env(env)
    old_render_mode = base_env.render_mode

    base_env.render_mode = "rgb_array"
    base_env.window = None

    writer = StreamingGifWriter(path, fps=GIF_FPS)
    base_env._frame_buffer = writer

    needs_explicit_render = not isinstance(env, (DungeonManagerEnv, BaseManagerEnv))

    for ep in tqdm(range(n_episodes), desc='GIF recording'):
        if writer.count >= max_frames:
            break

        obs, _ = env.reset()
        base_env.render()
        done = False
        step = 0
        max_steps_per_episode = getattr(base_env, 'max_steps', 1000)  # 1500

        while not done and step < max_steps_per_episode:
            if writer.count >= max_frames:
                break

            action = policy_fn(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            step += 1

            if needs_explicit_render:
                base_env.render()

    n_frames = writer.count
    writer.close()

    base_env._frame_buffer = None
    base_env.render_mode = old_render_mode
    base_env.window = None

    if n_frames > 0:
        logger.info("GIF saved: %s (%d frames)", path, n_frames)
    else:
        logger.warning("GIF not saved: no frames captured")

    gc.collect()


class PPOEpisodeTracker(gym.Wrapper):
    """Обёртка для PPO: честный подсчёт Success Rate и Reward."""

    def __init__(self, env, log_every: int = 1):
        super().__init__(env)
        self.log_every = log_every
        self.history = []
        self.ep_rewards = []
        self.ep_successes = []
        self.current_reward = 0.0
        self.env_steps = 0

    def reset(self, **kwargs):
        self.current_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.current_reward += reward
        self.env_steps += 1

        if term or trunc:
            self.ep_rewards.append(self.current_reward)
            self.ep_successes.append(1.0 if term else 0.0)

            if len(self.ep_rewards) % self.log_every == 0:
                window = min(50, len(self.ep_rewards))
                self.history.append({
                    "env_steps_total": self.env_steps,
                    "avg_reward": float(np.mean(self.ep_rewards[-window:])),
                    "success_rate": float(np.mean(self.ep_successes[-window:])),
                })
        return obs, reward, term, trunc, info


class PPOLoggingCallback(BaseCallback):
    """SB3 callback для логирования прогресса PPO."""

    def __init__(self, env_wrapper: PPOEpisodeTracker,
                 total_timesteps: int, log_interval: int = 5000,
                 results_path: str = None, save_interval: int = 50000,
                 eval_env=None, eval_interval: int = 50000,
                 eval_episodes: int = 20):
        super().__init__(verbose=0)
        self.env_wrapper = env_wrapper
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.results_path = results_path
        self.save_interval = save_interval
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self._last_log_step = 0
        self._last_save_step = 0
        self._last_eval_step = 0
        self._start_time = None
        self.eval_history = []

    def _on_training_start(self):
        self._start_time = time.time()
        logger.info("PPO learning started (%d steps)", self.total_timesteps)

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # --- Periodic evaluation ---
        if (self.eval_env is not None
                and step - self._last_eval_step >= self.eval_interval):
            self._last_eval_step = step
            sr, avg_r = _evaluate_ppo_episodes(
                self.model, self.eval_env, self.eval_episodes)
            self.eval_history.append({
                "env_steps_total": step,
                "eval_success_rate": sr,
                "eval_avg_reward": avg_r,
                "eval_avg_raw_reward": avg_r,
            })
            logger.info("  [Eval] Step %d | SR=%.1f%% | R=%.2f",
                        step, sr * 100, avg_r)
            if self.results_path:
                with open(self.results_path, 'w') as f:
                    json.dump({"history": self.eval_history}, f, indent=2)

        # --- Training logging (без изменений) ---
        if step - self._last_log_step < self.log_interval:
            return True
        self._last_log_step = step

        elapsed = time.time() - self._start_time
        fps = step / max(elapsed, 1e-8)
        remaining = (self.total_timesteps - step) / max(fps, 1e-8)
        eta = datetime.timedelta(seconds=int(remaining))

        n_ep = len(self.env_wrapper.ep_rewards)
        if n_ep > 0:
            window = min(50, n_ep)
            avg_r = np.mean(self.env_wrapper.ep_rewards[-window:])
            avg_sr = np.mean(self.env_wrapper.ep_successes[-window:])
        else:
            avg_r = 0.0
            avg_sr = 0.0

        logger.info(
            "Step: %6d/%d | Ep: %4d | R: %8.2f | Succ: %5.1f%% | "
            "FPS: %.0f | ETA: %s",
            step, self.total_timesteps, n_ep,
            avg_r, avg_sr * 100, fps, eta,
        )
        return True

    def _on_training_end(self):
        # Final evaluation
        if self.eval_env is not None:
            sr, avg_r = _evaluate_ppo_episodes(
                self.model, self.eval_env, self.eval_episodes)
            self.eval_history.append({
                "env_steps_total": self.num_timesteps,
                "eval_success_rate": sr,
                "eval_avg_reward": avg_r,
                "eval_avg_raw_reward": avg_r,
            })
            logger.info("  [Final Eval] SR=%.1f%% | R=%.2f", sr * 100, avg_r)

        elapsed = datetime.timedelta(seconds=int(time.time() - self._start_time))
        n_ep = len(self.env_wrapper.ep_rewards)
        if n_ep > 0:
            window = min(50, n_ep)
            final_r = np.mean(self.env_wrapper.ep_rewards[-window:])
            final_sr = np.mean(self.env_wrapper.ep_successes[-window:])
        else:
            final_r = 0.0
            final_sr = 0.0

        logger.info(
            "PPO learning completed in %s | Episodes: %d | "
            "Final train: R=%.2f, SR=%.1f%%",
            elapsed, n_ep, final_r, final_sr * 100,
        )


def _ppo_path(env_name, seed):
    return f"results/{env_name}_ppo_model_seed{seed}"


def _manager_path(env_name, algo, seed):
    return f"results/{env_name}_{algo}_manager_seed{seed}"


def _worker_path(env_name, seed):
    return f"results/{env_name}_shared_worker_seed{seed}"


def _evaluate_ppo(env: PPOEpisodeTracker, model, n_episodes: int = 50):
    """Прогнать n_episodes для сбора статистики."""
    for _ in tqdm(range(n_episodes), desc="PPO evaluation"):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc


def _evaluate_ppo_episodes(model, eval_env, n_episodes: int):
    """Прогнать n_episodes с deterministic policy, вернуть (success_rate, avg_reward)."""
    rewards = []
    successes = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        terminated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = eval_env.step(action)
            ep_reward += reward
            done = term or trunc
            if term:
                terminated = True
        rewards.append(ep_reward)
        successes.append(1.0 if terminated else 0.0)
    return float(np.mean(successes)), float(np.mean(rewards))


def run_ppo(env_name, seed, results_path):
    ppo_timesteps = _ppo_timesteps(env_name)
    logger.info("Vanilla PPO (seed %d, env %s)", seed, env_name)
    env_config = ENV_CONFIGS[env_name]

    if os.path.exists(results_path):
        logger.info("PPO results found at %s, skipping training", results_path)
        env = PPODungeonWrapper(DungeonHeistHardEnv(**env_config))
        model_path = _ppo_path(env_name, seed)
        if os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path, env=env)
            gif_path = f"results/{env_name}_ppo_seed{seed}.gif"
            if not os.path.exists(gif_path):
                record_gif(env, lambda obs: model.predict(obs, deterministic=True)[0], gif_path)
            del model
        del env
        _cleanup()
        return

    base_env = DungeonHeistHardEnv(**env_config)
    enhanced_env = PPODungeonWrapper(base_env)
    env = PPOEpisodeTracker(enhanced_env)
    eval_env = PPODungeonWrapper(DungeonHeistHardEnv(**env_config))
    model_path = _ppo_path(env_name, seed)

    model = PPO(
        "MlpPolicy", env, seed=seed,
        learning_rate=3e-4, n_steps=4096, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": dict(pi=SHARED_NET_ARCH, vf=SHARED_NET_ARCH)},
        verbose=0,
    )

    callback = PPOLoggingCallback(
        env, total_timesteps=ppo_timesteps,
        log_interval=PPO_LOG_INTERVAL,
        results_path=results_path,
        eval_env=eval_env,
        eval_interval=EVAL_INTERVAL_PPO,
        eval_episodes=EVAL_EPISODES,
    )
    model.learn(total_timesteps=ppo_timesteps, callback=callback)
    model.save(model_path)

    with open(results_path, "w") as f:
        json.dump({"history": callback.eval_history}, f, indent=2)

    gif_path = f"results/{env_name}_ppo_seed{seed}.gif"
    record_gif(env, lambda obs: model.predict(obs, deterministic=True)[0], gif_path)

    del model, env, eval_env
    _cleanup()


def _make_agent(env_name, seed):
    env_config = ENV_CONFIGS[env_name]
    env = DungeonHeistHardEnv(**env_config)
    goal_extractor = SimpleGoalExtractor(env.observation_space)

    return SGGHRLAgent(
        env=env,
        goal_extractor=goal_extractor,
        worker_env_class=DungeonWorkerEnv,
        manager_env_class=DungeonManagerEnv,
        max_worker_steps=MAX_WORKER_STEPS,
        success_threshold=SUCCESS_THRESHOLD,
        discretization=1.0,
        max_graph_nodes=MAX_GRAPH_NODES,
        directed_graph=False,
        valid_state_fn=lambda coord: (int(coord[0]), int(coord[1])) not in env.obstacles,
        seed=seed,
        worker_kwargs=WORKER_KWARGS,
        manager_kwargs=MANAGER_KWARGS,
    )

def train_worker(env_name, agent, seed):
    path = _worker_path(env_name, seed)
    logger.info("Worker learning (%d steps, seed %d, env %s)", WORKER_TIMESTEPS, seed, env_name)

    worker_cbs = CallbackList([
        RollingEpisodeStatsCallback(window_size=20),
        SetEnvAttrCallback("continuous_goals", value_on_start=True, value_on_end=False),
        ProgressPrinterCallback(log_interval=2000),
        WorkerCurriculumCallback(
            curriculum_stages=CURRICULUM_STAGES,
            min_stage_steps=4096,
            rollback_threshold=0.7,
        ),
        WorkerBestCheckpointCallback(best_model_path=path, only_final_stage=True),
    ])
    result = agent.train_worker(total_timesteps=WORKER_TIMESTEPS, callbacks=worker_cbs)

    if not os.path.exists(path + ".zip"):
        agent.save_worker(path)

    logger.info(
        "Worker is ready: SR=%.1f%%, Best R=%.4f (step %d), время: %s",
        result.final_success_rate * 100,
        result.best_avg_reward,
        result.best_step,
        result.total_time,
    )
    return result


def load_or_train_worker(env_name, agent, seed):
    path = _worker_path(env_name, seed)
    if os.path.exists(path + ".zip"):
        logger.info("Worker is loading from %s.zip", path)
        agent.load_worker(path)
    else:
        train_worker(env_name, agent, seed)


def train_manager(env_name, agent, algo, results_path, seed):
    model_path = _manager_path(env_name, algo, seed)
    manager_timesteps = _manager_timesteps(env_name)

    trained_steps, env_steps_done, existing_history = _get_resume_info(env_name, algo, seed)

    if trained_steps >= manager_timesteps - 200:
        if os.path.exists(model_path + ".zip"):
            logger.info("Manager fully trained (%d/%d steps), loading...",
                        trained_steps, manager_timesteps)
            agent.load_manager(model_path)
            return

    remaining_steps = manager_timesteps - trained_steps

    if os.path.exists(model_path + ".zip") and trained_steps > 0:
        logger.info("Manager partially trained (%d/%d steps), resuming for %d steps...",
                    trained_steps, manager_timesteps, remaining_steps)
        agent.load_manager(model_path)
    else:
        trained_steps = 0
        env_steps_done = 0
        existing_history = []

    logger.info("Manager learning [%s] (%d steps, env %s)", algo.upper(), remaining_steps, env_name)

    if algo == "sgghrl":
        inference_strategy = GraphPlannerStrategy(
            n_policy_samples=8,
            include_graph_neighbors=True,
            include_path_nodes=8,
            include_frontier=False,
        )
    else:
        inference_strategy = PolicyOnlyStrategy()


    cbs = [
        RollingEpisodeStatsCallback(window_size=25),
        ProgressPrinterCallback(log_interval=1000),
        ManagerEpsilonGreedyExplorationCallback(
            epsilon_start=1,
            epsilon_min=0.05,
            epsilon_decay_steps=100000,  # надо хотя бы 100000, но не 150000, для корректного сравнения
            step_offset=trained_steps
        ),
        SACTrainCallback(
            learning_starts=3000,
            train_freq=2,
            gradient_steps=1,
            batch_size=256
        ),
        # HERCallback(k_future=1, strategy="future"),
        ManagerEvalCallback(
            eval_interval=EVAL_INTERVAL_MANAGER,
            eval_episodes=EVAL_EPISODES,
            learning_starts=EVAL_LEARNING_STARTS,
            inference_strategy=inference_strategy,
        ),
        ManagerBestCheckpointOnEvalCallback(
            best_model_path=f"checkpoints/{env_name}_{algo}_manager_best_seed{seed}",
        ),
        FairEvaluationCallback(
            save_path=f"results/fair_{env_name}_{algo}_seed{seed}.json",
            worker_timesteps=WORKER_TIMESTEPS,
            step_offset=trained_steps,
            env_steps_offset=env_steps_done,
            existing_history=existing_history,
        )
    ]

    if algo == "sgghrl":
        cbs.extend([
            # GraphExplorationBonusCallback(
            #     alpha=0.1,
            #     decay_steps=0,
            #     min_bonus=0.005,
            # ),
            DeltaDistanceShapingCallback(
                shaping_ratio=2.5,
                retreat_multiplier=1.5,
                max_delta_estimate=5
            ),
            AdaptiveWorkerBudgetCallback(
                base_steps=5,
                steps_per_hop=5,
                max_steps=15,
            )
        ])

    result = agent.train_manager(
        total_timesteps=remaining_steps, callbacks=CallbackList(cbs),
    )
    result.save(results_path)

    agent.save_manager(model_path)
    logger.info("Manager saved to %s", model_path)

    logger.info(
        "Manager is ready: Best SR=%.1f%%, Best R=%.4f (step %d), "
        "episodes: %d, env steps: %d, time: %s",
        result.best_success_rate * 100,
        result.best_avg_reward,
        result.best_step,
        result.total_episodes,
        result.env_steps_total,
        result.total_time,
    )


def run_hierarchical(env_name, algo, seed, results_path, phase):
    logger.info("%s (seed %d, env %s, phase=%s)", algo.upper(), seed, env_name, phase)

    agent = _make_agent(env_name, seed)

    if phase in ("worker", "all"):
        load_or_train_worker(env_name, agent, seed)

    if phase in ("manager", "all"):
        if phase != "all":
            load_or_train_worker(env_name, agent, seed)
        train_manager(env_name, agent, algo, results_path, seed)

    if phase in ("manager", "all"):
        gif_path = f"results/{env_name}_{algo}_seed{seed}.gif"

        if algo == "sgghrl":
            strategy = GraphPlannerStrategy(
                n_policy_samples=3,
                include_graph_neighbors=True,
                include_path_nodes=3,
                include_frontier=False,
            )
            def sgghrl_policy(obs):
                raw_obs = agent.manager_env.worker_env.last_obs
                return strategy.select_action(
                    obs, raw_obs, agent.manager, agent.graph,
                    agent.goal_extractor, agent.manager_env,
                )
            record_gif(agent.manager_env, sgghrl_policy, gif_path)
        else:
            def hrl_policy(obs):
                action, _ = agent.manager.predict(obs, deterministic=True)
                return action
            record_gif(agent.manager_env, hrl_policy, gif_path)

    del agent
    _cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGGHRL Experiment Runner")
    parser.add_argument("--algo", type=str,
                        choices=["ppo", "hrl", "sgghrl"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--env", type=str,
                        choices=list(ENV_CONFIGS.keys()), default="hard-16",
                        help="Тип среды: hard-16, hard-25, easy-16, easy-25")
    parser.add_argument("--phase", type=str,
                        choices=["worker", "manager", "all"], default="all")
    args = parser.parse_args()

    setup_logging()
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{args.env}_{args.algo}_seed{args.seed}.json"

    if args.algo == "ppo":
        run_ppo(args.env, args.seed, results_path)
    else:
        run_hierarchical(args.env, args.algo, args.seed, results_path, args.phase)