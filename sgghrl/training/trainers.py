from __future__ import annotations

import time
import datetime
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.logger import configure

from sgghrl.training.callbacks import CallbackList, SGGHRLCallback
from ..utils import obs_to_tensor, obs_add_batch_dim
from sgghrl.core.base import CurriculumCapable
from sgghrl.core.results import WorkerTrainResult, ManagerTrainResult, TrainHistoryEntry

class WorkerTrainer:
    """Тренер для worker-политики (PPO).

    Реализует цикл сбора rollout'ов и обучения PPO
    с поддержкой колбэков GGHRL.

    Args:
        agent: экземпляр SGGHRLAgent.
    """

    def __init__(self, agent: "SGGHRLAgent"):
        self.agent = agent

    def train(self, total_timesteps: int, callbacks: Optional[SGGHRLCallback] = None) -> WorkerTrainResult:
        """Запустить обучение worker'а.

        Args:
            total_timesteps: общее число шагов среды.
            callbacks: колбэк или CallbackList.

        Returns:
            Словарь результатов с best_avg_reward, best_success_rate,
            best_step, final_success_rate, total_episodes, total_time.
        """
        agent = self.agent
        env = agent.worker_env
        model = agent.worker
        history = []

        cb = callbacks if isinstance(callbacks, CallbackList) else CallbackList([] if callbacks is None else [callbacks])

        ctx = SimpleNamespace(
            mode="worker",
            agent=agent,
            env=env,
            model=model,
            total_timesteps=int(total_timesteps),
            step=0,
            episode_count=0,
            episode_reward=0.0,
            episode_steps=0,
            stop_training=False,
            start_time=time.time(),
        )

        model.set_logger(configure(None, []))

        if cb.on_training_start(ctx) is False:
            return WorkerTrainResult()

        obs, _ = env.reset()
        model._last_episode_starts = np.array([True])

        while ctx.step < ctx.total_timesteps and not ctx.stop_training:
            for _ in range(model.n_steps):
                with torch.no_grad():
                    obs_tensor = obs_to_tensor(obs, model.device)
                    action, value, log_prob = model.policy(obs_tensor)
                    action = action.cpu().numpy().squeeze()
                    value = value.squeeze()
                    log_prob = log_prob.squeeze()

                if isinstance(env.action_space, gym.spaces.Discrete):
                    action = int(action)

                next_obs, reward, done, trunc, info = env.step(action)

                ctx.episode_reward += float(reward)
                ctx.episode_steps += 1

                model.rollout_buffer.add(
                    obs,
                    np.array([action]),
                    np.array([reward]),
                    model._last_episode_starts,
                    value,
                    log_prob,
                )

                obs = next_obs
                ctx.step += 1

                if done or trunc:
                    ctx.episode_count += 1
                    ctx.episode_reward_mean = ctx.episode_reward / max(ctx.episode_steps, 1)

                    ctx.episode_success = float(info.get("is_success", False))
                    if cb.on_episode_end(ctx) is False:
                        ctx.stop_training = True
                        break

                    ctx.episode_reward = 0.0
                    ctx.episode_steps = 0

                    obs, _ = env.reset()
                    model._last_episode_starts = np.array([True])
                else:
                    model._last_episode_starts = np.array([False])

                if cb.after_step(ctx) is False:
                    ctx.stop_training = True
                    break

                if ctx.step >= ctx.total_timesteps:
                    break

            with torch.no_grad():
                obs_tensor = obs_to_tensor(obs, model.device)
                _, last_value, _ = model.policy(obs_tensor)
                last_value = last_value.squeeze()

            if model.rollout_buffer.full:
                model.rollout_buffer.compute_returns_and_advantage(
                    last_values=last_value,
                    dones=model._last_episode_starts,
                )
                model.train()
                model.rollout_buffer.reset()
            else:
                model.rollout_buffer.reset()

            if cb.on_rollout_end(ctx) is False:
                ctx.stop_training = True

            # подготовка полей для логов/лучшей модели
            if hasattr(env, "get_success_rate"):
                ctx.success_rate = float(env.get_success_rate())
            else:
                _recent_successes = getattr(ctx, "recent_successes", None)
                if _recent_successes and len(_recent_successes) > 0:
                    ctx.success_rate = float(np.mean(_recent_successes))
                else:
                    ctx.success_rate = 0.0

            _recent_rewards = getattr(ctx, "recent_rewards", None)
            if _recent_rewards and len(_recent_rewards) > 0:
                ctx.avg_reward = float(np.mean(_recent_rewards))
            else:
                ctx.avg_reward = -float("inf")

            cdist = env.curriculum_distance if isinstance(env, CurriculumCapable) else 0.0

            ctx.log_line = (
                f"Step: {ctx.step:>6}/{ctx.total_timesteps} | "
                f"Ep: {ctx.episode_count:>4} | "
                f"R: {round(ctx.avg_reward, 4):>8.4f} | "
                f"Succ: {round(ctx.success_rate * 100, 4):>7.4f}% | "
                f"Dist: {round(cdist, 4):>6.4f}"
            )

            history.append(TrainHistoryEntry(
                step=ctx.step,
                episode=ctx.episode_count,
                avg_reward=ctx.avg_reward,
                success_rate=ctx.success_rate,
                stage=int(getattr(ctx, "stage_index", 0)),
                curriculum_distance=float(cdist),
                env_steps_total=ctx.step
            ))

            if cb.on_log(ctx) is False:
                ctx.stop_training = True

        cb.on_training_end(ctx)

        total_time = datetime.timedelta(seconds=int(time.time() - ctx.start_time))
        final_success = float(env.get_success_rate()) if hasattr(env, "get_success_rate") else 0.0

        return WorkerTrainResult(
            best_avg_reward=float(getattr(ctx, "best_avg_reward", getattr(ctx, "avg_reward", -float("inf")))),
            best_success_rate=float(getattr(ctx, "best_success_rate", 0.0)),
            best_step=int(getattr(ctx, "best_step", ctx.step)),
            final_success_rate=final_success,
            total_episodes=int(ctx.episode_count),
            total_time=total_time,
            history=history
        )


class ManagerTrainer:
    """Тренер для manager-политики (SAC).

    Реализует цикл взаимодействия со средой, заполнения
    replay buffer и обучения SAC с поддержкой колбэков GGHRL.

    Args:
        agent: экземпляр SGGHRLAgent.
    """
    def __init__(self, agent: "SGGHRLAgent"):
        self.agent = agent

    def train(self, total_timesteps: int, callbacks: Optional[SGGHRLCallback] = None) -> ManagerTrainResult:
        """Запустить обучение manager'а.

        Args:
            total_timesteps: общее число шагов manager-уровня.
            callbacks: колбэк или CallbackList.

        Returns:
            Словарь результатов с best_success_rate, best_avg_reward,
            best_step, final_success_rate, total_episodes,
            env_steps_total, total_time.
        """
        agent = self.agent
        env = agent.manager_env
        model = agent.manager
        history = []

        cb = callbacks if isinstance(callbacks, CallbackList) else CallbackList([] if callbacks is None else [callbacks])

        ctx = SimpleNamespace(
            mode="manager",
            agent=agent,
            env=env,
            model=model,
            total_timesteps=int(total_timesteps),
            step=0,
            episode_count=0,
            episode_reward=0.0,
            episode_original_reward=0.0,
            env_steps_total=0,
            stop_training=False,
            start_time=time.time(),
        )

        model.set_logger(configure(None, []))

        obs, _ = env.reset()
        current_raw_obs = env.worker_env.last_obs.copy()

        if cb.on_training_start(ctx) is False:
            return ManagerTrainResult()

        while ctx.step < ctx.total_timesteps and not ctx.stop_training:
            model.num_timesteps += 1
            ctx.step += 1

            # 1) подготовка перед действием
            ctx.obs = obs
            ctx.action = None
            ctx.current_raw_obs = current_raw_obs

            if cb.before_action(ctx) is False:
                ctx.stop_training = True
                break

            if ctx.action is None:
                ctx.action, _ = model.predict(obs, deterministic=False)

            # 2) шаг среды
            next_obs, reward, term, trunc, info = env.step(ctx.action)
            done = bool(term or trunc)

            # 3) ВСЕ поля ctx заполняются ДО after_step
            ctx.next_obs = next_obs
            ctx.reward = float(reward)
            ctx.original_reward = float(reward)
            ctx.terminated = bool(term)
            ctx.truncated = bool(trunc)
            ctx.done = done
            ctx.info = info

            ctx.env_steps_total += int(info.get("steps_taken", 1))
            next_raw_obs = env.worker_env.last_obs.copy()
            ctx.next_raw_obs = next_raw_obs

            # 4) after_step — callback'и могут модифицировать ctx.reward
            if cb.after_step(ctx) is False:
                ctx.stop_training = True
                break

            # 5) replay buffer
            ctx.episode_reward += ctx.reward
            ctx.episode_original_reward += ctx.original_reward

            model.replay_buffer.add(
                obs_add_batch_dim(obs),
                obs_add_batch_dim(next_obs),
                np.expand_dims(ctx.action, 0),
                np.array([ctx.reward]),
                np.array([ctx.terminated]),
                [{"TimeLimit.truncated": ctx.truncated}]
            )

            # 6) обработка конца эпизода + лог
            if done:
                ctx.episode_count += 1
                ctx.episode_success = 1.0 if ctx.terminated else 0.0
                ctx.episode_reward_finished = float(ctx.episode_reward)
                ctx.episode_original_reward_finished = float(ctx.episode_original_reward)

                if cb.on_episode_end(ctx) is False:
                    ctx.stop_training = True
                    break

                ctx.episode_reward = 0.0
                ctx.episode_original_reward = 0.0

                _recent_rewards = getattr(ctx, "recent_rewards", None)
                _recent_successes = getattr(ctx, "recent_successes", None)

                if _recent_rewards and len(_recent_rewards) > 0:
                    avg_reward = float(np.mean(_recent_rewards))
                else:
                    avg_reward = 0.0

                if _recent_successes and len(_recent_successes) > 0:
                    success_rate = float(np.mean(_recent_successes))
                else:
                    success_rate = 0.0

                ctx.avg_reward = avg_reward
                ctx.success_rate = success_rate
                orig_rewards = getattr(ctx, "recent_original_rewards", [])
                avg_orig_reward = float(np.mean(orig_rewards)) if len(orig_rewards) > 0 else 0.0
                ctx.log_line = (
                    f"Step: {ctx.step:>6}/{ctx.total_timesteps} | "
                    f"Ep: {ctx.episode_count:>4} | "
                    f"R: {round(avg_reward, 4):>9.4f} | "
                    f"R_orig: {round(avg_orig_reward, 4):>9.4f} | "
                    f"Succ: {round(success_rate * 100, 4):>7.4f}% | "
                    f"Eps: {round(float(getattr(ctx, 'epsilon', 0.0)), 4):.4f} | "
                    f"Graph: {len(agent.graph.nodes):>4} | "
                    f"EnvSteps: {ctx.env_steps_total:>7} | "
                    f"Best: {round(float(getattr(ctx, 'best_avg_reward', -float('inf'))), 4):.4f}"
                )

                history.append(TrainHistoryEntry(
                    step=ctx.step,
                    episode=ctx.episode_count,
                    avg_reward=avg_reward,
                    success_rate=success_rate,
                    env_steps_total=ctx.env_steps_total
                ))

                if cb.on_log(ctx) is False:
                    ctx.stop_training = True
                    break

                obs, _ = env.reset()
                current_raw_obs = env.worker_env.last_obs.copy()
            else:
                obs = next_obs
                current_raw_obs = next_raw_obs.copy()

            if not done:
                # подготовка строки лога (callbacks решают когда печатать)
                if hasattr(ctx, "recent_rewards") and len(ctx.recent_rewards) > 0:
                    avg_reward = float(np.mean(ctx.recent_rewards))
                    success_rate = float(np.mean(ctx.recent_successes))
                else:
                    avg_reward = 0.0
                    success_rate = 0.0

                ctx.avg_reward = avg_reward
                ctx.success_rate = success_rate
                orig_rewards = getattr(ctx, "recent_original_rewards", [])
                avg_orig_reward = float(np.mean(orig_rewards)) if len(orig_rewards) > 0 else 0.0

                ctx.log_line = (
                    f"Step: {ctx.step:>6}/{ctx.total_timesteps} | "
                    f"Ep: {ctx.episode_count:>4} | "
                    f"R: {avg_reward:>7.2f} | "
                    f"R_orig: {round(avg_orig_reward, 4):>9.4f} | "
                    f"Succ: {success_rate * 100:>5.1f}% | "
                    f"Eps: {float(getattr(ctx, 'epsilon', 0.0)):.3f} | "
                    f"Graph: {len(agent.graph.nodes):>4} | "
                    f"EnvSteps: {ctx.env_steps_total:>7} | "
                    f"Best: {float(getattr(ctx, 'best_avg_reward', -float('inf'))):.2f}"
                )

        cb.on_training_end(ctx)

        total_time = datetime.timedelta(seconds=int(time.time() - ctx.start_time))
        return ManagerTrainResult(
            best_success_rate=float(getattr(ctx, "best_success_rate", 0.0)),
            best_avg_reward=float(getattr(ctx, "best_avg_reward", -float("inf"))),
            best_step=int(getattr(ctx, "best_step", 0)),
            final_success_rate=float(getattr(ctx, "last_eval_success", 0.0)),
            final_avg_reward=float(getattr(ctx, "last_eval_reward", 0.0)),
            total_episodes=int(ctx.episode_count),
            env_steps_total=int(ctx.env_steps_total),
            total_time=total_time,
            history=history
        )