from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from ..training.callbacks import SGGHRLCallback
from ..training.her import HERBuffer
from ..utils import obs_add_batch_dim


class HERCallback(SGGHRLCallback):
    """Колбэк Hindsight Experience Replay для manager-обучения.

    Собирает переходы в HERBuffer и в конце каждого эпизода
    генерирует дополнительные переходы с подменёнными целями,
    добавляя их в replay buffer модели.

    Args:
        k_future: число будущих целей на каждый переход.
        strategy: стратегия HER — "future", "final" или "episode".
        reward_scale: масштаб HER-награды.
    """
    def __init__(self, k_future: int = 4, strategy: str = "future",
                 reward_scale: float = 0.1):
        super().__init__()
        self.k_future = int(k_future)
        self.strategy = str(strategy)
        self.reward_scale = float(reward_scale)
        self.her_buffer = None
        self.total_added = 0

    def on_training_start(self, ctx: SimpleNamespace) -> bool:
        self.her_buffer = HERBuffer(
            ctx.env, ctx.agent.goal_extractor,
            self.k_future, self.strategy,
            reward_scale=self.reward_scale,
        )
        self.total_added = 0
        return True

    def after_step(self, ctx: SimpleNamespace) -> bool:
        self.her_buffer.add(
            ctx.obs,
            np.array(ctx.action),
            ctx.next_obs,
            ctx.current_raw_obs,
            ctx.next_raw_obs,
            ctx.reward,
            ctx.done,
        )
        return True

    def on_episode_end(self, ctx: SimpleNamespace) -> bool:
        her_transitions = self.her_buffer.get_her_transitions()
        for trans in her_transitions:
            ctx.model.replay_buffer.add(
                obs_add_batch_dim(trans["obs"]),
                obs_add_batch_dim(trans["next_obs"]),
                np.expand_dims(trans["action"], 0),
                np.array([trans["reward"]]),
                np.array([trans["done"]]),
                [{}],
            )
        self.total_added += len(her_transitions)
        self.her_buffer.clear()
        ctx.her_transitions_added = self.total_added
        return True