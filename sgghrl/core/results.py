from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class TrainHistoryEntry:
    """Одна запись истории обучения."""
    step: int = 0
    episode: int = 0
    avg_reward: float = 0.0
    success_rate: float = 0.0
    stage: int = 0
    curriculum_distance: float = 0.0
    env_steps_total: int = 0


def _serialize_timedelta(td: datetime.timedelta) -> float:
    return td.total_seconds()


def _deserialize_timedelta(seconds: float) -> datetime.timedelta:
    return datetime.timedelta(seconds=seconds)


@dataclass(frozen=True)
class WorkerTrainResult:
    """Результат обучения worker-политики."""
    best_avg_reward: float = -float("inf")
    best_success_rate: float = 0.0
    best_step: int = 0
    final_success_rate: float = 0.0
    total_episodes: int = 0
    total_time: datetime.timedelta = field(default_factory=lambda: datetime.timedelta())
    history: List[TrainHistoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_avg_reward": self.best_avg_reward,
            "best_success_rate": self.best_success_rate,
            "best_step": self.best_step,
            "final_success_rate": self.final_success_rate,
            "total_episodes": self.total_episodes,
            "total_time": _serialize_timedelta(self.total_time),
            "history": [asdict(e) for e in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerTrainResult":
        history = [TrainHistoryEntry(**e) for e in d.get("history", [])]
        return cls(
            best_avg_reward=d.get("best_avg_reward", -float("inf")),
            best_success_rate=d.get("best_success_rate", 0.0),
            best_step=d.get("best_step", 0),
            final_success_rate=d.get("final_success_rate", 0.0),
            total_episodes=d.get("total_episodes", 0),
            total_time=_deserialize_timedelta(d.get("total_time", 0.0)),
            history=history,
        )

    def save(self, path: str):
        """Сохранить результат в JSON.

        Args:
            path: путь к файлу (.json).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "WorkerTrainResult":
        """Загрузить результат из JSON.

        Args:
            path: путь к файлу (.json).

        Returns:
            WorkerTrainResult.
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

@dataclass(frozen=True)
class ManagerTrainResult:
    """Результат обучения manager-политики."""
    best_success_rate: float = 0.0
    best_avg_reward: float = -float("inf")
    best_step: int = 0
    final_success_rate: float = 0.0
    final_avg_reward: float = 0.0
    total_episodes: int = 0
    env_steps_total: int = 0
    total_time: datetime.timedelta = field(default_factory=lambda: datetime.timedelta())
    history: List[TrainHistoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_success_rate": self.best_success_rate,
            "best_avg_reward": self.best_avg_reward,
            "best_step": self.best_step,
            "final_success_rate": self.final_success_rate,
            "final_avg_reward": self.final_avg_reward,
            "total_episodes": self.total_episodes,
            "env_steps_total": self.env_steps_total,
            "total_time": _serialize_timedelta(self.total_time),
            "history": [asdict(e) for e in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ManagerTrainResult":
        history = [TrainHistoryEntry(**e) for e in d.get("history", [])]
        return cls(
            best_success_rate=d.get("best_success_rate", 0.0),
            best_avg_reward=d.get("best_avg_reward", -float("inf")),
            best_step=d.get("best_step", 0),
            final_success_rate=d.get("final_success_rate", 0.0),
            final_avg_reward=d.get("final_avg_reward", 0.0),
            total_episodes=d.get("total_episodes", 0),
            env_steps_total=d.get("env_steps_total", 0),
            total_time=_deserialize_timedelta(d.get("total_time", 0.0)),
            history=history,
        )

    def save(self, path: str):
        """Сохранить результат в JSON.

        Args:
            path: путь к файлу (.json).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ManagerTrainResult":
        """Загрузить результат из JSON.

        Args:
            path: путь к файлу (.json).

        Returns:
            ManagerTrainResult.
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

@dataclass(frozen=True)
class DiagnoseResult:
    """Результат диагностики worker-политики."""
    success_rate: float = 0.0
    avg_steps: float = 0.0
    n_episodes: int = 0

    def to_dict(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "n_episodes": self.n_episodes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnoseResult":
        return cls(
            success_rate=d.get("success_rate", 0.0),
            avg_steps=d.get("avg_steps", 0.0),
            n_episodes=d.get("n_episodes", 0),
        )

    def save(self, path: str):
        """Сохранить результат в JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "DiagnoseResult":
        """Загрузить результат из JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))