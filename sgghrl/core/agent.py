from __future__ import annotations

import os
import pickle
from typing import Optional, Type, Callable
import json

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC

from ..core.goals import GoalExtractor
from ..core.graph import StateGraph
from ..training.callbacks import (
    CallbackList,
)
from ..training.trainers import WorkerTrainer, ManagerTrainer
from ..utils.buffer_io import save_replay_buffer, load_replay_buffer
from ..logging import logger
from .._version import __version__ as _lib_version
from ..core.base import BaseGoalExtractor, BaseWorkerEnv, BaseManagerEnv
from ..core.results import WorkerTrainResult, ManagerTrainResult, DiagnoseResult
from ..seed import set_global_seed
from .inference import InferenceStrategy, PolicyOnlyStrategy, GraphPlannerStrategy

class SGGHRLAgent:
    """Subgoal Graph-Guided Hierarchical Reinforcement Learning агент.

    Двухуровневая иерархия: manager выбирает подцел
    worker достигает их. Граф состояний отслеживает
    посещённые состояния для exploration bonus.

    Args:
        env: базовая gymnasium-среда.
        goal_extractor: экстрактор целей (BaseGoalExtractor).
        worker_env_class: класс worker-среды (подкласс BaseWorkerEnv).
        manager_env_class: класс manager-среды (подкласс BaseManagerEnv).
        worker_kwargs: дополнительные аргументы для PPO.
        manager_kwargs: дополнительные аргументы для SAC.
        max_worker_steps: бюджет шагов worker'а на одно действие manager'а.
        success_threshold: порог расстояния для достижения подцели.
        discretization: размер ячейки сетки для графа состояний.
        max_graph_nodes: максимальное число узлов графа.
        context_fn: контекстная функция для ключей графа.
        worker_policy: "auto", "MlpPolicy", "CnnPolicy" и т.д.
        manager_policy: "auto", "MlpPolicy", "MultiInputPolicy" и т.д.
        seed: Глобальный seed для воспроизводимости (random, numpy, torch). None — без фиксации.
        directed_graph: True — направленный граф (A→B ≠ B→A). Для сред с односторонними переходами.
        valid_state_fn: Фильтр валидных координат графа для определения frontier.
            Пример: lambda c: c not in walls. None — все в пределах bounds валидны.

    Raises:
        TypeError: если goal_extractor, worker_env_class или
            manager_env_class не соответствуют контрактам.
    """
    def __init__(
        self,
        env: gym.Env,
        goal_extractor: GoalExtractor,
        worker_env_class: Type[gym.Wrapper],
        manager_env_class: Type[gym.Wrapper],
        worker_kwargs: Optional[dict] = None,
        manager_kwargs: Optional[dict] = None,
        max_worker_steps: int = 15,
        success_threshold: float = 0.5,
        discretization: float = 1.0,
        max_graph_nodes: int = 10000,
        context_fn=None,
        worker_policy: str = "auto",
        manager_policy: str = "auto",
        seed: Optional[int] = None,
        directed_graph: bool = False,
        valid_state_fn: Optional[Callable[[tuple], bool]] = None
    ):
        if not isinstance(goal_extractor, BaseGoalExtractor):
            raise TypeError(
                f"goal_extractor должен наследовать BaseGoalExtractor, "
                f"получен {type(goal_extractor).__name__}"
            )
        if not (isinstance(worker_env_class, type) and issubclass(worker_env_class, BaseWorkerEnv)):
            raise TypeError(
                f"worker_env_class должен быть подклассом BaseWorkerEnv, "
                f"получен {worker_env_class!r}"
            )
        if not (isinstance(manager_env_class, type) and issubclass(manager_env_class, BaseManagerEnv)):
            raise TypeError(
                f"manager_env_class должен быть подклассом BaseManagerEnv, "
                f"получен {manager_env_class!r}"
            )

        self.seed = set_global_seed(seed)

        self.base_env = env
        self.goal_extractor = goal_extractor

        self.graph = StateGraph(
            goal_extractor=goal_extractor,
            discretization=discretization,
            max_nodes=max_graph_nodes,
            context_fn=context_fn,
            directed=directed_graph,
            valid_state_fn=valid_state_fn
        )

        self.worker_env = worker_env_class(
            env=env,
            goal_extractor=goal_extractor,
            success_threshold=success_threshold,
        )

        worker_cfg = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "verbose": 0,
        }
        if worker_kwargs:
            worker_cfg.update(worker_kwargs)

        w_policy, worker_cfg = _auto_configure_policy(
            self.worker_env.observation_space, worker_policy, worker_cfg
        )
        self.worker = PPO(w_policy, self.worker_env, **worker_cfg)

        self.manager_env = manager_env_class(
            env=env,
            worker_env=self.worker_env,
            worker_model=self.worker,
            graph=self.graph,
            goal_extractor=goal_extractor,
            max_worker_steps=max_worker_steps,
            success_threshold=success_threshold,
        )

        manager_cfg = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "learning_starts": 100,
            "verbose": 0,
        }
        if manager_kwargs:
            manager_cfg.update(manager_kwargs)

        m_policy, manager_cfg = _auto_configure_policy(
            self.manager_env.observation_space, manager_policy, manager_cfg
        )
        self.manager = SAC(m_policy, self.manager_env, **manager_cfg)

    def save_buffer(self, path: str = "checkpoints/buffer.pkl"):
        """Сохранить replay buffer manager'а на диск.

        Args:
            path: путь к файлу.

        Returns:
            Количество сохранённых переходов.
        """
        return save_replay_buffer(self.manager.replay_buffer, path)

    def load_buffer(self, path: str = "checkpoints/buffer.pkl", append: bool = False):
        """Загрузить replay buffer manager'а с диска.

        Args:
            path: путь к файлу.
            append: если True — добавить к существующему буферу,
                иначе — заменить.

        Returns:
            Количество загруженных переходов.
        """
        return load_replay_buffer(self.manager.replay_buffer, path, append=append)

    def train_worker(self, total_timesteps: int, callbacks=None) -> WorkerTrainResult:
        """Обучить worker-политику (PPO).

        Args:
            total_timesteps: общее число шагов среды.
            callbacks: SGGHRLCallback, список колбэков или CallbackList.

        Returns:
            Словарь с best_avg_reward, best_success_rate, best_step,
            final_success_rate, total_episodes, total_time.
        """
        trainer = WorkerTrainer(self)
        cb = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)
        return trainer.train(total_timesteps=total_timesteps, callbacks=cb)

    def train_manager(self, total_timesteps: int, callbacks=None) -> ManagerTrainResult:
        """Обучить manager-политику (SAC).

        Args:
            total_timesteps: общее число шагов manager-уровня.
            callbacks: SGGHRLCallback, список колбэков или CallbackList.

        Returns:
            Словарь с best_success_rate, best_avg_reward, best_step,
            final_success_rate, total_episodes, env_steps_total, total_time.
        """
        trainer = ManagerTrainer(self)
        cb = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)
        return trainer.train(total_timesteps=total_timesteps, callbacks=cb)

    def set_render_mode(self, render_mode):
        """Установить режим рендеринга базовой среды.

        Args:
            render_mode: "human", None и т.д.
        """
        self.base_env.render_mode = render_mode

    def _evaluate_manager(self, n_episodes: int,
                          strategy: Optional[InferenceStrategy] = None):
        """Оценить manager с заданной стратегией инференса."""
        if strategy is None:
            strategy = PolicyOnlyStrategy()

        successes = 0
        total_reward = 0.0
        term = False
        for _ in range(n_episodes):
            obs, _ = self.manager_env.reset()
            done = False
            ep_r = 0.0
            while not done:
                raw_obs = self.manager_env.worker_env.last_obs
                action = strategy.select_action(
                    obs, raw_obs, self.manager, self.graph,
                    self.goal_extractor, self.manager_env,
                )
                obs, r, term, trunc, _ = self.manager_env.step(action)
                ep_r += float(r)
                done = bool(term or trunc)
            if term:
                successes += 1
            total_reward += ep_r
        return successes / max(n_episodes, 1), total_reward / max(n_episodes, 1)

    def predict(
        self,
        obs: np.ndarray,
        raw_obs: np.ndarray,
        strategy: Optional[InferenceStrategy] = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Выбрать действие manager'а с заданной стратегией.

        Args:
            obs: обёрнутое наблюдение manager'а.
            raw_obs: сырое наблюдение из базовой среды.
            strategy: стратегия инференса (None — PolicyOnlyStrategy).
            deterministic: игнорируется при использовании
                GraphPlannerStrategy (всегда выбирает лучшего по Q).

        Returns:
            Действие manager'а (np.ndarray).
        """
        if strategy is None:
            strategy = PolicyOnlyStrategy()

        return strategy.select_action(
            obs=obs,
            raw_obs=raw_obs,
            manager_model=self.manager,
            graph=self.graph,
            goal_extractor=self.goal_extractor,
            manager_env=self.manager_env,
        )

    def diagnose_worker(self, n_episodes=300, max_steps=50, setup_fn=None) -> DiagnoseResult:
        """Диагностика worker-политики.

        Args:
            n_episodes: число диагностических эпизодов.
            max_steps: бюджет шагов на эпизод.
            setup_fn: функция setup_fn(worker_env) -> obs,
                которая сбрасывает среду, выбирает тестовую цель
                и возвращает обёрнутое наблюдение.

        Returns:
            Словарь с success_rate, avg_steps, n_episodes.

        Raises:
            ValueError: если setup_fn не передан.
        """
        if setup_fn is None:
            raise ValueError("Передайте setup_fn(env) -> obs")

        successes = 0
        total_steps = []
        for _ in range(n_episodes):
            obs = setup_fn(self.worker_env)
            for step in range(max_steps):
                action, _ = self.worker.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.worker_env.step(int(action))
                if info.get("is_success", False):
                    successes += 1
                    total_steps.append(step + 1)
                    break
                if done or trunc:
                    total_steps.append(step + 1)
                    break

        sr = successes / max(n_episodes, 1)
        avg = float(np.mean(total_steps)) if total_steps else max_steps
        return DiagnoseResult(
            success_rate=sr,
            avg_steps=avg,
            n_episodes=n_episodes,
        )

    def demo(self, n_episodes: int = 5,
             strategy: Optional[InferenceStrategy] = None):
        """Запустить n_episodes с manager'ом и вывести результаты.

        Args:
            n_episodes: число эпизодов для демонстрации.
            strategy: стратегия инференса (None — PolicyOnlyStrategy).
        """
        if strategy is None:
            strategy = PolicyOnlyStrategy()

        logger.info("Demo (strategy: %s):", type(strategy).__name__)
        for ep in range(n_episodes):
            obs, _ = self.manager_env.reset()
            done = False
            term = False
            reward_sum = 0.0
            step_count = 0

            while not done:
                step_count += 1
                raw_obs = self.manager_env.worker_env.last_obs
                action = strategy.select_action(
                    obs, raw_obs, self.manager, self.graph,
                    self.goal_extractor, self.manager_env,
                )
                obs, reward, term, trunc, _ = self.manager_env.step(action)
                reward_sum += float(reward)
                done = bool(term or trunc)

            status = "SUCCESS" if term else "TIMEOUT"
            logger.info("Episode %d: %s, R=%.1f, Steps=%d", ep + 1, status, reward_sum, step_count)

    def save_worker(self, path: str = "models/worker"):
        """Сохранить worker-модель.

        Args:
            path: путь без расширения (SB3 добавит .zip).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.worker.save(path)

    def save_manager(self, path: str = "models/manager"):
        """Сохранить manager-модель и граф состояний.

        Args:
            path: путь без расширения. Граф сохраняется как {path}_graph.pkl.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.manager.save(path)

        graph_path = f"{path}_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump({"nodes": dict(self.graph.nodes), "edges": dict(self.graph.edges)}, f)

    def load_worker(self, path: str = "models/worker", ignore_mismatched_sizes: bool = False):
        """Загрузить worker-модель.

        Args:
            path: путь к сохранённой модели.
            ignore_mismatched_sizes: если True — загрузить совпадающие
                параметры и инициализировать новые нулями.
        """
        if ignore_mismatched_sizes:
            from ..nn.surgery import load_sb3_policy_state_dict, load_state_dict_ignore_mismatched
            saved_sd = load_sb3_policy_state_dict(path)
            skipped = load_state_dict_ignore_mismatched(self.worker.policy, saved_sd)
            if skipped:
                logger.warning("Worker: skipped %d mismatched params:", len(skipped))
                for key, old, new in skipped:
                    print(f"  {key}: {old} → {new}")
            else:
                logger.info("Worker: all params matched")
        else:
            self.worker = PPO.load(path, env=self.worker_env)
        self.manager_env.set_worker_model(self.worker)

    def load_manager(self, path: str = "models/manager", ignore_mismatched_sizes: bool = False):
        """Загрузить manager-модель и граф состояний.

        Args:
            path: путь к сохранённой модели.
            ignore_mismatched_sizes: если True — загрузить совпадающие
                параметры и инициализировать новые нулями.
        """
        if ignore_mismatched_sizes:
            from ..nn.surgery import load_sb3_policy_state_dict, load_state_dict_ignore_mismatched
            saved_sd = load_sb3_policy_state_dict(path)
            skipped = load_state_dict_ignore_mismatched(self.manager.policy, saved_sd)
            if skipped:
                logger.warning("Manager: skipped %d mismatched params:", len(skipped))
                for key, old, new in skipped:
                    print(f"  {key}: {old} → {new}")
            else:
                logger.info("Manager: all params matched")
        else:
            self.manager = SAC.load(path, env=self.manager_env)

        graph_path = f"{path}_graph.pkl"
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                data = pickle.load(f)
                self.graph.nodes = data["nodes"]
                self.graph.edges = data["edges"]
            self.graph.rebuild_index()

    def save_models(self, folder: str = "models"):
        """Сохранить обе модели в папку.

        Args:
            folder: путь к папке.
        """
        os.makedirs(folder, exist_ok=True)
        self.save_worker(f"{folder}/worker")
        self.save_manager(f"{folder}/manager")

    def load_models(self, folder: str = "models"):
        """Загрузить обе модели из папки.

        Args:
            folder: путь к папке.
        """
        self.load_worker(f"{folder}/worker")
        self.load_manager(f"{folder}/manager")

    def save(self, folder: str = "checkpoints/full"):
        """Полная сериализация агента: модели, граф, буфер, метаданные.

        Args:
            folder: путь к папке для сохранения.
        """
        os.makedirs(folder, exist_ok=True)

        self.save_worker(f"{folder}/worker")
        self.manager.save(f"{folder}/manager")
        self.save_buffer(f"{folder}/buffer.pkl")

        graph_path = f"{folder}/graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump({
                "nodes": dict(self.graph.nodes),
                "edges": dict(self.graph.edges),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            "version": _lib_version,
            "seed": self.seed,
            "max_worker_steps": getattr(self.manager_env, "_max_worker_steps", None),
            "success_threshold": getattr(self.manager_env, "success_threshold", None),
            "discretization": self.graph.discretization,
            "max_graph_nodes": self.graph.max_nodes,
            "graph_nodes_count": len(self.graph.nodes),
            "graph_edges_count": len(self.graph.edges),
            "buffer_size": self.manager.replay_buffer.buffer_size,
            "buffer_pos": self.manager.replay_buffer.pos,
            "buffer_full": self.manager.replay_buffer.full,
        }
        with open(f"{folder}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Full checkpoint saved to '%s'", folder)

    def load(self, folder: str = "checkpoints/full",
             load_buffer: bool = True,
             ignore_mismatched_sizes: bool = False):
        """Полная десериализация агента.

        Args:
            folder: путь к папке с чекпоинтом.
            load_buffer: загружать ли replay buffer.
            ignore_mismatched_sizes: игнорировать несовпадения размеров весов.
        """
        meta = {}
        meta_path = f"{folder}/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        logger.info("Loading checkpoint (v%s, seed=%s, graph=%d nodes)",
                     meta.get("version", "?"), meta.get("seed"),
                     meta.get("graph_nodes_count", 0))

        self.load_worker(f"{folder}/worker", ignore_mismatched_sizes=ignore_mismatched_sizes)
        self.load_manager(f"{folder}/manager", ignore_mismatched_sizes=ignore_mismatched_sizes)

        graph_path = f"{folder}/graph.pkl"
        if os.path.exists(graph_path) and not self.graph.nodes:
            with open(graph_path, "rb") as f:
                data = pickle.load(f)
                self.graph.nodes = data["nodes"]
                self.graph.edges = data["edges"]
            self.graph.rebuild_index()

        if load_buffer:
            buf_path = f"{folder}/buffer.pkl"
            if os.path.exists(buf_path):
                self.load_buffer(buf_path)

        logger.info("Full checkpoint loaded from '%s'", folder)

def _auto_configure_policy(obs_space, policy_hint, algo_kwargs):
    if policy_hint in (None, "auto"):
        if isinstance(obs_space, gym.spaces.Dict):
            policy = "MultiInputPolicy"
        elif len(obs_space.shape) == 3:
            policy = "CnnPolicy"
        else:
            policy = "MlpPolicy"
    else:
        policy = policy_hint

    if isinstance(obs_space, gym.spaces.Dict):
        pk = algo_kwargs.get("policy_kwargs") or {}
        if "features_extractor_class" not in pk:
            has_small = any(
                len(s.shape) == 3 and min(s.shape[1], s.shape[2]) < 36
                for s in obs_space.spaces.values()
            )
            if has_small:
                from ..nn.extractors import ImageVectorExtractor
                pk = dict(pk)
                pk["features_extractor_class"] = ImageVectorExtractor
                algo_kwargs = dict(algo_kwargs)
                algo_kwargs["policy_kwargs"] = pk

    return policy, algo_kwargs