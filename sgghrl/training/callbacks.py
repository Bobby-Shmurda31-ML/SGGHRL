from __future__ import annotations

import random
import time
import datetime
import collections
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Sequence, List, Any, Tuple
import numpy as np

from ..core.base import CurriculumCapable
from ..logging import logger

class SGGHRLCallback:
    """Базовый класс для колбэков тренировочного цикла GGHRL.

    Все методы возвращают bool: False останавливает обучение.
    Переопределяйте нужные методы в подклассах.
    """

    def __init__(self):
        self.parent: Optional["CallbackList"] = None

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        """Вызывается один раз в начале обучения.

        Args:
            ctx: контекст с полями agent, env, model, total_timesteps и др.
        """
        return True

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        """Вызывается один раз в конце обучения.

        Args:
            ctx: контекст обучения.
        """
        return True

    def before_action(self, ctx: SimpleNamespace) -> bool:
        """Вызывается перед выбором действия (только manager).

        Можно подменить ctx.action для epsilon-greedy и т.д.

        Args:
            ctx: контекст с ctx.obs, ctx.action (None до подмены).
        """
        return True

    def after_step(self, ctx: SimpleNamespace) -> bool:
        """Вызывается после каждого шага среды.

        Args:
            ctx: контекст с ctx.obs, ctx.next_obs, ctx.reward, ctx.done и др.
        """
        return True

    def on_episode_end(self, ctx: SimpleNamespace) -> bool:
        """Вызывается в конце каждого эпизода.

        Args:
            ctx: контекст с ctx.episode_reward, ctx.episode_success и др.
        """
        return True

    def on_rollout_end(self, ctx: SimpleNamespace) -> bool:
        """Вызывается после завершения rollout'а (только worker/PPO).

        Args:
            ctx: контекст обучения.
        """
        return True

    def on_log(self, ctx: SimpleNamespace) -> bool:
        """Вызывается при готовности метрик для логирования.

        Args:
            ctx: контекст с ctx.avg_reward, ctx.success_rate, ctx.log_line и др.
        """
        return True

    def on_eval_end(self, ctx: SimpleNamespace, success_rate: float, avg_reward: float) -> bool:
        """Вызывается после оценки (evaluation) manager'а.

        Args:
            ctx: контекст обучения.
            success_rate: доля успешных эпизодов [0, 1].
            avg_reward: средняя награда за эпизод.
        """
        return True

    def set_parent(self, parent: "CallbackList"):
        self.parent = parent


class CallbackList(SGGHRLCallback):
    """Контейнер для нескольких колбэков. Вызывает их последовательно.

    Args:
        callbacks: список GGHRLCallback или None.
    """
    def __init__(self, callbacks: Optional[Sequence[SGGHRLCallback]] = None):
        self.callbacks: List[SGGHRLCallback] = list(callbacks) if callbacks is not None else []
        for cb in self.callbacks:
            cb.set_parent(self)

    def add(self, cb: SGGHRLCallback):
        """Добавить колбэк в список.

        Args:
            cb: колбэк для добавления.
        """
        cb.set_parent(self)
        self.callbacks.append(cb)

    def _call(self, name: str, *args, **kwargs) -> bool:
        for cb in self.callbacks:
            fn = getattr(cb, name)
            if fn(*args, **kwargs) is False:
                return False
        return True

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        return self._call("on_training_start", ctx)

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        return self._call("on_training_end", ctx)

    def before_action(self, ctx: SimpleNamespace) -> bool:
        return self._call("before_action", ctx)

    def after_step(self, ctx: SimpleNamespace) -> bool:
        return self._call("after_step", ctx)

    def on_episode_end(self, ctx: SimpleNamespace) -> bool:
        return self._call("on_episode_end", ctx)

    def on_rollout_end(self, ctx: SimpleNamespace) -> bool:
        return self._call("on_rollout_end", ctx)

    def on_log(self, ctx: SimpleNamespace) -> bool:
        return self._call("on_log", ctx)

    def on_eval_end(self, ctx: SimpleNamespace, success_rate: float, avg_reward: float) -> bool:
        for cb in self.callbacks:
            if cb.on_eval_end(ctx, success_rate, avg_reward) is False:
                return False
        return True


def _clone_state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


@dataclass
class LinearSchedule:
    start: float
    end: float
    duration_steps: int

    def value(self, step: int) -> float:
        if self.duration_steps <= 0:
            return float(self.end)
        t = min(max(step / self.duration_steps, 0.0), 1.0)
        return float(self.start + t * (self.end - self.start))


class SetEnvAttrCallback(SGGHRLCallback):
    """Устанавливает атрибут среды в начале и конце обучения.

    Args:
        attr: имя атрибута.
        value_on_start: значение при старте обучения.
        value_on_end: значение при завершении обучения (None — не менять).
    """
    def __init__(self, attr: str, value_on_start: Any, value_on_end: Any = None):
        super().__init__()
        self.attr = attr
        self.value_on_start = value_on_start
        self.value_on_end = value_on_end

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        if hasattr(ctx.env, self.attr):
            setattr(ctx.env, self.attr, self.value_on_start)
        return True

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        if self.value_on_end is not None and hasattr(ctx.env, self.attr):
            setattr(ctx.env, self.attr, self.value_on_end)
        return True


class ProgressPrinterCallback(SGGHRLCallback):
    """Печатает прогресс обучения с заданным интервалом.

    Выводит ctx.log_line с добавлением ETA.

    Args:
        log_interval: интервал в шагах между выводами.
    """
    def __init__(self, log_interval: int = 1000):
        super().__init__()
        self.log_interval = int(log_interval)
        self._last_log_step = 0
        self._last_log_time = None

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        self._last_log_step = 0
        self._last_log_time = time.time()
        return True

    def on_log(self, ctx):
        if (ctx.step - self._last_log_step) < self.log_interval:
            return True
        now = time.time()
        dt = max(now - (self._last_log_time or now), 1e-8)
        fps = (ctx.step - self._last_log_step) / dt
        eta_sec = int((ctx.total_timesteps - ctx.step) / fps) if fps > 0 else 0
        eta = datetime.timedelta(seconds=eta_sec)
        self._last_log_step = ctx.step
        self._last_log_time = now
        ctx.fps = fps
        ctx.eta = eta
        if hasattr(ctx, "log_line"):
            logger.info("%s | ETA: %s", ctx.log_line, eta)
        return True


class CheckpointCallback(SGGHRLCallback):
    """Периодическое сохранение модели на диск.

    Args:
        save_path: базовый путь для сохранения.
        save_freq: интервал в шагах (0 — только в конце).
        verbose: уровень вывода (0 — тихо, 1 — с сообщениями).
    """
    def __init__(self, save_path: str, save_freq: int = 0, verbose: int = 1):
        super().__init__()
        self.save_path = save_path
        self.save_freq = int(save_freq)
        self.verbose = verbose
        self._last_save_step = 0

    def _save(self, ctx: SimpleNamespace, suffix: str = ""):
        path = f"{self.save_path}{suffix}"
        if ctx.mode == "worker":
            ctx.agent.save_worker(path)
        elif ctx.mode == "manager":
            ctx.agent.save_manager(path)
        if self.verbose > 0:
            logger.info("[Checkpoint] Saved %s to '%s'", ctx.mode, path)

    def on_log(self, ctx: SimpleNamespace) -> bool:
        if self.save_freq > 0 and (ctx.step - self._last_save_step) >= self.save_freq:
            self._last_save_step = ctx.step
            self._save(ctx, suffix=f"_{ctx.step}")
        return True

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        self._save(ctx, suffix="_final")
        return True


class RollingEpisodeStatsCallback(SGGHRLCallback):
    """Ведёт скользящую статистику наград и success rate по эпизодам.

    Добавляет в ctx поля recent_rewards и recent_successes (deque).
    Очищает статистику при смене стадии curriculum.

    Args:
        window_size: размер окна скользящего среднего.
    """
    def __init__(self, window_size: int = 50):
        super().__init__()
        self.window_size = int(window_size)
        self._last_stage = None

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        ctx.recent_rewards = collections.deque(maxlen=self.window_size)
        ctx.recent_successes = collections.deque(maxlen=self.window_size)
        ctx.recent_original_rewards = collections.deque(maxlen=self.window_size)
        self._last_stage = None
        return True

    def on_episode_end(self, ctx: SimpleNamespace) -> bool:
        current_stage = getattr(ctx, "stage_index", None)
        if current_stage is not None and current_stage != self._last_stage:
            ctx.recent_rewards.clear()
            ctx.recent_successes.clear()
            self._last_stage = current_stage

        reward_val = getattr(ctx, "episode_reward_mean", ctx.episode_reward)
        ctx.recent_rewards.append(float(reward_val))
        ctx.recent_successes.append(float(ctx.episode_success))

        original_reward = getattr(ctx, "episode_original_reward", reward_val)
        ctx.recent_original_rewards.append(float(original_reward))
        return True


class WorkerCurriculumCallback(SGGHRLCallback):
    """Автоматическое продвижение по стадиям curriculum для worker'а.

    Требует чтобы worker-среда реализовывала CurriculumCapable.
    Поддерживает откат на предыдущую стадию при падении success rate.

    Args:
        curriculum_stages: список кортежей (max_distance, target_success_rate).
            None — без curriculum.
        min_stage_steps: минимум шагов на стадии перед возможной сменой.
        rollback_threshold: порог для отката (множитель от предыдущей цели).
        distance_decay: параметр экспоненциального затухания весов по расстоянию.
    """
    def __init__(
        self,
        curriculum_stages: Optional[list] = None,
        min_stage_steps: int = 10240,
        rollback_threshold: float = 0.75,
        distance_decay: float = 0.5,
    ):
        super().__init__()
        self.curriculum_stages = curriculum_stages
        self.min_stage_steps = int(min_stage_steps)
        self.rollback_threshold = float(rollback_threshold)
        self.distance_decay = float(distance_decay)

        self.current_stage = 0
        self.stage_start_step = 0
        self.final_stage_index = 0

    def _build_distance_weights(self, max_dist: int) -> dict:
        distances = list(range(1, max_dist + 1))
        if not distances:
            return {1: 1.0}
        raw = []
        for d in distances:
            raw.append(self.distance_decay ** (max_dist - d))
        total = sum(raw)
        return {d: w / total for d, w in zip(distances, raw)}

    def _apply_stage(self, ctx, stage_index: int):
        max_dist, target = self.curriculum_stages[stage_index]
        weights = self._build_distance_weights(int(max_dist))

        if not isinstance(ctx.env, CurriculumCapable):
            raise TypeError(
                f"{type(ctx.env).__name__} должен реализовать CurriculumCapable "
                f"(set_curriculum_weights + curriculum_distance)"
            )
        ctx.env.set_curriculum_weights(weights)

        ctx.is_final_stage = (stage_index == self.final_stage_index)
        ctx.stage_index = stage_index

        target_str = f"{target * 100:.0f}%" if target else "None"
        weights_str = " ".join(f"d{d}={w:.0%}" for d, w in sorted(weights.items()))
        logger.info(
            "\n[Stage %d/%d] MaxDist: %s | Target: %s | Weights: %s",
            stage_index + 1, len(self.curriculum_stages),
            max_dist, target_str, weights_str,
        )

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        if self.curriculum_stages is None:
            ctx.is_final_stage = True
            ctx.stage_index = 0
            return True

        self.current_stage = 0
        self.stage_start_step = 0
        self.final_stage_index = len(self.curriculum_stages) - 1
        self._apply_stage(ctx, 0)
        return True

    def after_step(self, ctx: SimpleNamespace) -> bool:
        if self.curriculum_stages is None:
            return True

        steps_in_stage = ctx.step - self.stage_start_step
        if steps_in_stage < self.min_stage_steps:
            return True

        success_rate = float(ctx.env.get_success_rate())
        old_stage = self.current_stage

        if self.current_stage > 0:
            prev_threshold = self.curriculum_stages[self.current_stage - 1][1]
            if prev_threshold and success_rate < prev_threshold * self.rollback_threshold:
                self.current_stage -= 1

        if self.current_stage == old_stage:
            current_threshold = self.curriculum_stages[self.current_stage][1]
            if current_threshold and success_rate >= current_threshold:
                if self.current_stage < len(self.curriculum_stages) - 1:
                    self.current_stage += 1

        if self.current_stage != old_stage:
            self.stage_start_step = ctx.step
            self._apply_stage(ctx, self.current_stage)
            direction = "UP" if self.current_stage > old_stage else "DOWN"
            logger.info("  %s Succ: %.1f%%", direction, success_rate * 100)

        return True


class WorkerBestCheckpointCallback(SGGHRLCallback):
    """Сохраняет лучшие веса worker'а по avg_reward.

    При завершении обучения восстанавливает лучшие веса.

    Args:
        best_model_path: путь для сохранения лучшей модели (None — не сохранять).
        only_final_stage: если True — отслеживать только на финальной стадии curriculum.
    """
    def __init__(self, best_model_path: Optional[str] = None, only_final_stage: bool = True):
        super().__init__()
        self.best_model_path = best_model_path
        self.only_final_stage = bool(only_final_stage)
        self.best_avg_reward = -float("inf")
        self.best_success_rate = 0.0
        self.best_step = 0
        self._policy_state = None

    def on_log(self, ctx: SimpleNamespace) -> bool:
        if self.only_final_stage and (not getattr(ctx, "is_final_stage", True)):
            return True

        avg_reward = float(ctx.avg_reward)
        success_rate = float(ctx.success_rate)

        is_better = (avg_reward > self.best_avg_reward) or (
            avg_reward == self.best_avg_reward and success_rate > self.best_success_rate
        )
        if not is_better:
            return True

        self.best_avg_reward = avg_reward
        self.best_success_rate = success_rate
        self.best_step = int(ctx.step)
        self._policy_state = _clone_state_dict_to_cpu(ctx.model.policy.state_dict())

        logger.info("New best: R=%.4f, %.1f%% success at step %d", avg_reward, success_rate * 100, ctx.step)
        if self.best_model_path:
            ctx.agent.save_worker(self.best_model_path)
        return True

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        if self._policy_state is not None:
            ctx.model.policy.load_state_dict(self._policy_state)
            logger.info("Restored best worker weights from step %d (R=%.4f)",
                        self.best_step, self.best_avg_reward)
        ctx.best_avg_reward = self.best_avg_reward
        ctx.best_success_rate = self.best_success_rate
        ctx.best_step = self.best_step
        return True


class StopOnThresholdCallback(SGGHRLCallback):
    """Останавливает обучение при достижении пороговых значений.

    Args:
        reward_threshold: минимальная средняя награда (None — не проверять).
        success_threshold: минимальный success rate (None — не проверять).
    """
    def __init__(self, reward_threshold: Optional[float] = None, success_threshold: Optional[float] = None):
        super().__init__()
        self.reward_threshold = reward_threshold
        self.success_threshold = success_threshold

    def on_log(self, ctx: SimpleNamespace) -> bool:
        if self.reward_threshold is None and self.success_threshold is None:
            return True
        reward_ok = self.reward_threshold is None or float(ctx.avg_reward) >= float(self.reward_threshold)
        success_ok = self.success_threshold is None or float(ctx.success_rate) >= float(self.success_threshold)
        if reward_ok and success_ok:
            ctx.stop_training = True
            return False
        return True


class ManagerEpsilonGreedyExplorationCallback(SGGHRLCallback):
    """Epsilon-greedy exploration для manager'а.

    Линейно снижает epsilon от epsilon_start до epsilon_min.
    С вероятностью epsilon заменяет действие на случайное.

    Args:
        epsilon_start: начальное значение epsilon.
        epsilon_min: минимальное значение epsilon.
        epsilon_decay_steps: число шагов для линейного снижения
            (None — использовать total_timesteps).
    """
    def __init__(self, epsilon_start: float = 0.5, epsilon_min: float = 0.05,
                 epsilon_decay_steps=None, step_offset: int = 0):
        super().__init__()
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_offset = int(step_offset)
        self._sched = None

    def on_training_start(self, ctx):
        decay = int(self.epsilon_decay_steps) if self.epsilon_decay_steps else int(ctx.total_timesteps)
        self._sched = LinearSchedule(self.epsilon_start, self.epsilon_min, decay)
        return True

    def before_action(self, ctx):
        eps = self._sched.value(ctx.step + self.step_offset)
        ctx.epsilon = eps
        if random.random() < eps:
            ctx.action = ctx.agent.manager.action_space.sample()
        return True


class SACTrainCallback(SGGHRLCallback):
    """Запускает обучение SAC с заданной частотой.

    Args:
        train_freq: частота обучения в шагах среды.
        gradient_steps: число gradient steps за один вызов train().
        batch_size: размер батча.
        learning_starts: число шагов до начала обучения.
    """
    def __init__(self, train_freq: int = 1, gradient_steps: int = 1, batch_size: int = 256, learning_starts: int = 1000):
        super().__init__()
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)

    def after_step(self, ctx: SimpleNamespace) -> bool:
        if ctx.step < self.learning_starts:
            return True
        if self.train_freq <= 0:
            return True
        if ctx.step % self.train_freq != 0:
            return True
        ctx.model.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
        return True


class ManagerEvalCallback(SGGHRLCallback):
    """Периодическая оценка manager-политики.

    Args:
        eval_interval: интервал оценки в шагах.
        eval_episodes: число эпизодов для оценки.
        learning_starts: число шагов до первой оценки.
        inference_strategy: стратегия инференса для eval
            (None — PolicyOnlyStrategy).
    """
    def __init__(self, eval_interval: int = 5000, eval_episodes: int = 10,
                 learning_starts: int = 1000, inference_strategy=None):
        super().__init__()
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.learning_starts = int(learning_starts)
        self._last_eval_step = 0
        self.inference_strategy = inference_strategy

    def on_episode_end(self, ctx: SimpleNamespace) -> bool:
        if ctx.step < self.learning_starts:
            return True
        if (ctx.step - self._last_eval_step) < self.eval_interval:
            return True

        self._last_eval_step = ctx.step

        success, avg_r, avg_raw_r = self._evaluate(ctx.agent, self.eval_episodes)
        ctx.last_eval_success = success
        ctx.last_eval_reward = avg_r
        ctx.last_eval_raw_reward = avg_raw_r

        logger.info("[Eval] Step %d | SR=%.1f%% | R=%.2f | R_raw=%.2f", ctx.step, success * 100, avg_r, avg_raw_r)

        if self.parent is not None:
            return self.parent.on_eval_end(ctx, success, avg_r)
        return True

    def _evaluate(self, agent, eval_episodes: int) -> Tuple[float, float, float]:
        from ..core.inference import PolicyOnlyStrategy

        strategy = self.inference_strategy or PolicyOnlyStrategy()

        successes = 0
        total_reward = 0.0
        total_raw_reward = 0.0
        env = agent.manager_env
        model = agent.manager

        for _ in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_raw_reward = 0.0
            while not done:
                raw_obs = env.worker_env.last_obs
                action = strategy.select_action(
                    obs, raw_obs, model, agent.graph,
                    agent.goal_extractor, env,
                )
                obs, reward, term, trunc, info = env.step(action)
                ep_reward += float(reward)
                ep_raw_reward += info.get('raw_env_reward', 0.0)
                done = bool(term or trunc)
            if term:
                successes += 1
            total_reward += ep_reward
            total_raw_reward += ep_raw_reward

        n = max(eval_episodes, 1)
        return successes / n, total_reward / n, total_raw_reward / n


class ManagerBestCheckpointOnEvalCallback(SGGHRLCallback):
    """Сохраняет лучшие веса manager'а по результатам evaluation.

    При завершении обучения восстанавливает лучшие веса.

    Args:
        best_model_path: путь для сохранения лучшей модели (None — не сохранять).
    """
    def __init__(self, best_model_path: Optional[str] = None):
        super().__init__()
        self.best_model_path = best_model_path
        self.best_success_rate = 0.0
        self.best_avg_reward = -float("inf")
        self.best_step = 0
        self._actor = None
        self._critic = None
        self._critic_target = None

    def on_eval_end(self, ctx: SimpleNamespace, success_rate: float, avg_reward: float) -> bool:
        is_better = (avg_reward > self.best_avg_reward) or (
            avg_reward == self.best_avg_reward and success_rate > self.best_success_rate
        )
        if not is_better:
            ctx.best_avg_reward = self.best_avg_reward
            ctx.best_success_rate = self.best_success_rate
            ctx.best_step = self.best_step
            return True

        self.best_success_rate = float(success_rate)
        self.best_avg_reward = float(avg_reward)
        self.best_step = int(ctx.step)

        m = ctx.model
        self._actor = _clone_state_dict_to_cpu(m.actor.state_dict())
        self._critic = _clone_state_dict_to_cpu(m.critic.state_dict())
        self._critic_target = _clone_state_dict_to_cpu(m.critic_target.state_dict())

        logger.info("New best manager: R=%.4f, %.1f%% success at step %d",
                     avg_reward, success_rate * 100, ctx.step)
        if self.best_model_path:
            ctx.agent.save_manager(self.best_model_path)
        return True

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        if self._actor is not None:
            m = ctx.model
            m.actor.load_state_dict(self._actor)
            m.critic.load_state_dict(self._critic)
            m.critic_target.load_state_dict(self._critic_target)
            logger.info("Restored best manager weights from step %d", self.best_step)
        ctx.best_success_rate = self.best_success_rate
        ctx.best_avg_reward = self.best_avg_reward
        ctx.best_step = self.best_step
        return True


class BufferCheckpointCallback(SGGHRLCallback):
    """Периодическое сохранение replay buffer на диск.

    Args:
        save_path: путь к файлу буфера.
        save_freq: интервал сохранения в шагах.
        save_on_end: сохранять ли при завершении обучения.
    """

    def __init__(self, save_path: str = "checkpoints/buffer.pkl",
                 save_freq: int = 50000, save_on_end: bool = True):
        super().__init__()
        self.save_path = save_path
        self.save_freq = int(save_freq)
        self.save_on_end = bool(save_on_end)
        self._last_save_step = 0

    def on_log(self, ctx: SimpleNamespace) -> bool:
        if self.save_freq > 0 and (ctx.step - self._last_save_step) >= self.save_freq:
            self._last_save_step = ctx.step
            ctx.agent.save_buffer(self.save_path)
        return True

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        if self.save_on_end:
            ctx.agent.save_buffer(self.save_path)
        return True

class ManagerEpsilonWithBurstsCallback(SGGHRLCallback):
    """Epsilon-greedy с периодическими всплесками exploration.

    Когда success_rate падает — epsilon временно повышается.

    Args:
        epsilon_start: начальное значение.
        epsilon_min: минимальное значение.
        epsilon_decay_steps: шаги линейного затухания.
        burst_epsilon: значение epsilon при всплеске.
        burst_trigger_drop: падение SR, запускающее всплеск.
        burst_relative_drop: Относительный порог падения SR для всплеска:
            триггер = max(burst_trigger_drop, best_sr × burst_relative_drop).
        burst_duration: длительность всплеска в шагах.
        min_sr_for_burst: Мин. SR для активации всплесков. Ниже — всплески отключены (SR нестабилен).
    """
    def __init__(self, epsilon_start=0.8, epsilon_min=0.08,
                 epsilon_decay_steps=80000,
                 burst_epsilon=0.4, burst_trigger_drop=0.10,
                 burst_relative_drop=0.3,
                 burst_duration=5000, min_sr_for_burst=0.15):
        super().__init__()
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.burst_epsilon = burst_epsilon
        self.burst_trigger_drop = burst_trigger_drop
        self.burst_duration = burst_duration
        self.burst_relative_drop = burst_relative_drop
        self.min_sr_for_burst = min_sr_for_burst

        self._best_sr = 0.0
        self._burst_until = 0
        self._sched = None

    def on_training_start(self, ctx):
        decay = self.epsilon_decay_steps or ctx.total_timesteps
        self._sched = LinearSchedule(self.epsilon_start, self.epsilon_min, decay)
        return True

    def before_action(self, ctx):
        base_eps = self._sched.value(ctx.step)

        # Обновить лучший SR
        sr = getattr(ctx, 'success_rate', 0.0)
        if sr > self._best_sr:
            self._best_sr = 0.9 * self._best_sr + 0.1 * sr
        else:
            self._best_sr = 0.995 * self._best_sr + 0.005 * sr

        # Проверить нужен ли всплеск
        if self._best_sr > self.min_sr_for_burst:
            drop_threshold = max(
                self.burst_trigger_drop,
                self._best_sr * self.burst_relative_drop
            )
            if sr < self._best_sr - drop_threshold and ctx.step > self._burst_until:
                self._burst_until = ctx.step + self.burst_duration
                logger.info("[EpsBurst] SR dropped %.1f%% → %.1f%% (thr=%.1f%%), burst until step %d",
                            self._best_sr * 100, sr * 100, drop_threshold * 100, self._burst_until)

        # Выбрать epsilon
        if ctx.step < self._burst_until:
            eps = max(base_eps, self.burst_epsilon)
        else:
            eps = base_eps

        ctx.epsilon = eps
        if random.random() < eps:
            ctx.action = ctx.agent.manager.action_space.sample()
        return True

class GraphExplorationBonusCallback(SGGHRLCallback):
    """Добавляет exploration bonus на основе графа состояний.

    bonus = alpha / sqrt(visit_count(next_state))
    Для новых состояний visit_count = 0, бонус = alpha.

    Args:
        alpha: масштаб бонуса.
        decay_steps: число шагов для линейного затухания alpha до 0
            (None — без затухания).
        min_bonus: Бонус ниже этого порога обнуляется. Убирает шум от часто посещённых состояний.
    """
    def __init__(self, alpha: float = 0.5, decay_steps: Optional[int] = None, min_bonus: float = 0.001):
        super().__init__()
        self.alpha = float(alpha)
        self.decay_steps = decay_steps
        self.min_bonus = float(min_bonus)

    def _current_alpha(self, step: int) -> float:
        if self.decay_steps is None or self.decay_steps <= 0:
            return self.alpha
        t = min(step / self.decay_steps, 1.0)
        return self.alpha * (1.0 - t)

    def after_step(self, ctx: SimpleNamespace) -> bool:
        graph = ctx.agent.graph
        raw_obs = ctx.next_raw_obs
        visit_count = max(graph.visit_count(raw_obs), 1)
        alpha = self._current_alpha(ctx.step)

        bonus = alpha / (visit_count ** 0.5)

        if bonus < self.min_bonus:
            bonus = 0.0

        ctx.reward = float(ctx.reward) + bonus
        return True


class AdaptiveWorkerBudgetCallback(SGGHRLCallback):
    """Адаптивный бюджет шагов worker'а на основе расстояния в графе.

    budget = base_steps + graph_distance * steps_per_hop
    Ограничен диапазоном [min_steps, max_steps].

    Требует чтобы manager_env имел атрибут _max_worker_steps (writable).

    Args:
        base_steps: минимальный бюджет.
        steps_per_hop: дополнительные шаги за каждый hop в графе.
        max_steps: максимальный бюджет.
        min_steps: минимальный бюджет.
        default_steps: Fallback-бюджет когда граф не может оценить расстояние.
        cache_interval: Интервал (в шагах manager'а) между пересчётами бюджета.
    """
    def __init__(self, base_steps: int = 5, steps_per_hop: int = 5,
                 max_steps: int = 50, min_steps: int = 3,
                 default_steps: int = 15, cache_interval: int = 3):
        super().__init__()
        self.base_steps = int(base_steps)
        self.steps_per_hop = int(steps_per_hop)
        self.max_steps = int(max_steps)
        self.min_steps = int(min_steps)
        self.default_steps = int(default_steps)
        self._original_max_steps = None
        self.cache_interval = int(cache_interval)
        self._last_budget_step = -1
        self._cached_budget = None

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        self._original_max_steps = getattr(ctx.env, '_max_worker_steps',
                                           self.default_steps)
        return True

    def _estimate_budget(self, graph, current_raw, subgoal):
        dist = graph.shortest_path_to_goal(current_raw, subgoal)
        if dist is None:
            return self.default_steps
        if dist == 0:
            return self.min_steps
        budget = self.base_steps + dist * self.steps_per_hop
        return max(self.min_steps, min(self.max_steps, budget))

    def before_action(self, ctx: SimpleNamespace) -> bool:
        env = ctx.env
        graph = ctx.agent.graph
        if not hasattr(env, '_max_worker_steps'):
            return True

        if (self._cached_budget is not None
                and ctx.step - self._last_budget_step < self.cache_interval):
            env._max_worker_steps = self._cached_budget
            return True

        current_raw = env.worker_env.last_obs

        goal = env.final_goal
        if goal is not None:
            budget = self._estimate_budget(graph, current_raw, goal)
        else:
            budget = self.default_steps

        env._max_worker_steps = budget
        self._cached_budget = budget
        self._last_budget_step = ctx.step
        return True

    def on_training_end(self, ctx: SimpleNamespace) -> bool:
        if self._original_max_steps is not None and hasattr(ctx.env, '_max_worker_steps'):
            ctx.env._max_worker_steps = self._original_max_steps
        return True

class FrontierExplorationBonusCallback(SGGHRLCallback):
    """Бонус за движение к frontier-узлам графа состояний.

    Frontier — узлы на границе исследованного пространства.
    Бонус тем выше, чем ближе agent к frontier после шага.

    bonus = alpha * (1 / (1 + dist_to_frontier))

    Если agent попал на frontier-узел — максимальный бонус.
    Если agent далеко от frontier — бонус стремится к нулю.

    Args:
        alpha: масштаб бонуса.
        decay_steps: число шагов для линейного затухания
            (None — без затухания).
        min_graph_nodes: минимальный размер графа для активации
            (на маленьком графе все узлы — frontier, бонус бессмыслен).
        cache_interval: Интервал между пересчётами frontier-расстояний (multi-source BFS).
    """
    def __init__(self, alpha: float = 0.3, decay_steps: Optional[int] = None,
                 min_graph_nodes: int = 10, cache_interval: int = 5):
        super().__init__()
        self.alpha = float(alpha)
        self.decay_steps = decay_steps
        self.min_graph_nodes = int(min_graph_nodes)
        self.cache_interval = int(cache_interval)
        self._cached_frontier_dist = None
        self._last_frontier_step = -1

    def _current_alpha(self, step: int) -> float:
        if self.decay_steps is None or self.decay_steps <= 0:
            return self.alpha
        t = min(step / self.decay_steps, 1.0)
        return self.alpha * (1.0 - t)

    def _compute_frontier_distances(self, graph) -> dict:
        """Multi-source BFS от всех frontier-узлов."""
        frontier = graph.get_frontier_nodes()
        if not frontier:
            return {}

        adj = graph._get_adjacency()
        dist = {k: 0 for k in frontier}
        queue = collections.deque(frontier)

        while queue:
            current = queue.popleft()
            for neighbor in adj.get(current, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def after_step(self, ctx: SimpleNamespace) -> bool:
        graph = ctx.agent.graph
        if len(graph.nodes) < self.min_graph_nodes:
            return True

        if (self._cached_frontier_dist is None or
                ctx.step - self._last_frontier_step >= self.cache_interval):
            self._cached_frontier_dist = self._compute_frontier_distances(graph)
            self._last_frontier_step = ctx.step

        key = graph.to_key(ctx.next_raw_obs)
        dist = self._cached_frontier_dist.get(key)
        if dist is None:
            return True

        alpha = self._current_alpha(ctx.step)
        visit_count = max(graph.visit_count(ctx.next_raw_obs), 1)
        bonus = alpha / ((1.0 + dist) * (visit_count ** 0.5))
        ctx.reward = float(ctx.reward) + bonus
        return True

class ManagerLRDecayOnPlateauCallback(SGGHRLCallback):
    """Уменьшение learning rate manager'а при стагнации награды.

    Отслеживает среднюю награду через EMA. Если за patience шагов
    не было улучшения на min_delta — умножает lr на factor.

    Args:
        patience: число шагов без улучшения для срабатывания.
        factor: множитель lr при срабатывании (0 < factor < 1).
        min_lr: минимальный lr, ниже которого не опускаться.
        min_delta: минимальное улучшение для сброса счётчика.
        cooldown: число шагов после срабатывания, в течение
            которых повторное снижение заблокировано.
        ema_alpha: коэффициент EMA для сглаживания reward.
    """
    def __init__(self, patience: int = 10000, factor: float = 0.5,
                 min_lr: float = 1e-5, min_delta: float = 0.01,
                 cooldown: int = 5000, ema_alpha: float = 0.01):
        super().__init__()
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.min_delta = float(min_delta)
        self.cooldown = int(cooldown)
        self.ema_alpha = float(ema_alpha)

        self._reward_ema = None
        self._best_ema = -float("inf")
        self._best_step = 0
        self._cooldown_until = 0
        self._current_lr = None
        self._decay_count = 0

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        lr = ctx.model.learning_rate
        self._current_lr = lr if isinstance(lr, float) else lr(1.0)
        self._reward_ema = None
        self._best_ema = -float("inf")
        self._best_step = 0
        self._cooldown_until = 0
        self._decay_count = 0
        return True

    def _apply_lr(self, model, new_lr: float):
        """Применить новый lr ко всем оптимизаторам SAC."""
        model.learning_rate = new_lr
        model.lr_schedule = lambda _: new_lr

        for optimizer in [model.actor.optimizer, model.critic.optimizer]:
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

        if model.ent_coef_optimizer is not None:
            for pg in model.ent_coef_optimizer.param_groups:
                pg["lr"] = new_lr

    def on_episode_end(self, ctx: SimpleNamespace) -> bool:
        reward = ctx.episode_original_reward_finished

        if self._reward_ema is None:
            self._reward_ema = reward
        else:
            self._reward_ema = (1 - self.ema_alpha) * self._reward_ema + self.ema_alpha * reward

        if self._reward_ema > self._best_ema + self.min_delta:
            self._best_ema = self._reward_ema
            self._best_step = ctx.step

        if ctx.step < self._cooldown_until:
            return True

        steps_without_improvement = ctx.step - self._best_step
        if steps_without_improvement < self.patience:
            return True

        new_lr = max(self._current_lr * self.factor, self.min_lr)

        if new_lr >= self._current_lr:
            return True

        self._decay_count += 1
        old_lr = self._current_lr
        self._current_lr = new_lr
        self._apply_lr(ctx.model, new_lr)

        self._best_step = ctx.step
        self._cooldown_until = ctx.step + self.cooldown

        logger.info(
            "[LR Decay #%d] %.2e → %.2e | EMA reward: %.4f | "
            "No improvement for %d steps",
            self._decay_count, old_lr, new_lr,
            self._reward_ema, steps_without_improvement,
        )
        self._best_ema = self._reward_ema

        return True

class DeltaDistanceShapingCallback(SGGHRLCallback):
    """Шейпинг по изменению графового расстояния до цели.

    Каждый шаг менеджера:
        delta = dist_before - dist_after  (>0 = приблизился)

        if delta > 0:  shaping = +delta * scale
        if delta < 0:  shaping = +delta * scale * retreat_multiplier
        if delta = 0:  shaping = 0

    Свойства:
    - Осцилляция наказывается (retreat > approach → net < 0)
    - Стояние на месте = 0 (нет бесплатных наград)
    - Нет γ-утечки
    - Сумма за эпизод телескопируется:
        Σ shaping ≈ scale × (dist_start - dist_end)
    - auto_scale калибрует по |avg_reward|

    Args:
        scale: множитель шейпинга (если auto_scale=False).
        retreat_multiplier: во сколько раз штраф за удаление
            сильнее награды за приближение (>1 → анти-осцилляция).
        min_graph_nodes: минимальный размер графа для активации.
        auto_scale: автоматический подбор scale.
        shaping_ratio: целевое отношение |max_shaping| / |avg_reward|.
        max_delta_estimate: ожидаемое макс. изменение расстояния за шаг
            (обычно = worker_view_radius). Используется для auto_scale.
    """

    def __init__(self,
                 scale: float = 0.1,
                 retreat_multiplier: float = 1.5,
                 min_graph_nodes: int = 20,
                 auto_scale: bool = True,
                 shaping_ratio: float = 0.3,
                 max_delta_estimate: float = 6.0):
        super().__init__()
        self.scale = float(scale)
        self.retreat_multiplier = float(retreat_multiplier)
        self.min_graph_nodes = int(min_graph_nodes)
        self.auto_scale = bool(auto_scale)
        self.shaping_ratio = float(shaping_ratio)
        self.max_delta_estimate = float(max_delta_estimate)

        self._effective_scale = float(scale)
        self._reward_ema = 0.0
        self._reward_count = 0

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        self._effective_scale = self.scale
        self._reward_ema = 0.0
        self._reward_count = 0
        return True

    def _update_scale(self, original_reward: float):
        if not self.auto_scale:
            return

        self._reward_count += 1
        alpha = 0.01
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * abs(original_reward)

        if self._reward_count < 50 or self._reward_ema < 1e-8:
            return

        self._effective_scale = (
            self._reward_ema * self.shaping_ratio / self.max_delta_estimate
        )

    def _get_distance(self, graph, raw_obs: np.ndarray,
                      goal: np.ndarray) -> Optional[float]:
        dist = graph.shortest_path_to_goal(raw_obs, goal)
        return float(dist) if dist is not None else None

    def after_step(self, ctx: SimpleNamespace) -> bool:
        graph = ctx.agent.graph
        if len(graph.nodes) < self.min_graph_nodes:
            return True

        goal = ctx.env.final_goal
        if goal is None:
            return True
        goal = np.asarray(goal, dtype=np.float32)

        raw_reward = float(getattr(ctx, 'original_reward', ctx.reward))
        self._update_scale(raw_reward)

        d_before = self._get_distance(graph, ctx.current_raw_obs, goal)
        d_after = self._get_distance(graph, ctx.next_raw_obs, goal)

        if d_before is None or d_after is None:
            return True

        delta = d_before - d_after

        if delta > 0:
            shaping = delta * self._effective_scale
        elif delta < 0:
            shaping = delta * self._effective_scale * self.retreat_multiplier
        else:
            shaping = 0.0

        ctx.reward = float(ctx.reward) + shaping
        return True