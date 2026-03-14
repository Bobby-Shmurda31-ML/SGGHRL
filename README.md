# SGGHRL: Subgoal Graph-Guided Hierarchical Reinforcement Learning

SGGHRL — библиотека для иерархического обучения с подкреплением (HRL), решающая проблему редких наград и исследования в сложных средах (лабиринты, ключи-двери, ловушки, враги).

Метод использует **дискретизированный граф состояний (State Graph)**, который:
1. Формирует dense reward shaping через кратчайшие пути (potential-based, не искажает оптимальную политику).
2. Направляет исследование (exploration bonus к frontier-узлам и редко посещённым состояниям).
3. Выступает планировщиком на инференсе (граф генерирует кандидатов, SAC critic выбирает лучшего).
4. Адаптивно выделяет бюджет шагов Worker'у на основе топологии среды.

## Установка

Библиотеку можно установить напрямую из репозитория GitHub:

**Способ 1: Установка через pip (рекомендуется)**
```bash
pip install git+https://github.com/Bobby-Shmurda31-ML/SGGHRL.git
```

**Способ 2: Установка без Git (через архив)**
Если у вас не установлен Git, вы можете установить библиотеку напрямую из архива:
```bash
pip install https://github.com/Bobby-Shmurda31-ML/SGGHRL/archive/refs/heads/main.zip
```

**Для разработки:**
Если вы хотите редактировать исходный код библиотеки:
```bash
git clone https://github.com/Bobby-Shmurda31-ML/SGGHRL.git
cd SGGHRL
pip install -e ".[dev]"
```

## Архитектура

Двухуровневая иерархия:

- **Manager (SAC):** Анализирует глобальное состояние и выдаёт подцель (goal) для Worker'а.
- **Worker (PPO):** Goal-conditioned политика, обученная достигать локальных целей за ограниченное число шагов.
- **StateGraph:** Дискретизированный граф посещённых состояний. Строится динамически, поддерживает BFS-расстояния, frontier-детекцию, контекстные ключи.

Компоненты интегрированы через систему колбэков — reward shaping, exploration bonus, adaptive budget и HER подключаются как независимые модули.

Подробнее: [docs/architecture.md](docs/architecture.md)

## Быстрый старт

```python
from sgghrl import SGGHRLAgent, GoalExtractor, setup_logging
from sgghrl import (
    CallbackList, RollingEpisodeStatsCallback, ProgressPrinterCallback,
    WorkerCurriculumCallback, WorkerBestCheckpointCallback,
    ManagerEpsilonGreedyExplorationCallback, SACTrainCallback,
    GraphExplorationBonusCallback,
    AdaptiveWorkerBudgetCallback, HERCallback,
)
from your_env import YourEnv, YourWorkerEnv, YourManagerEnv

setup_logging()

env = YourEnv()
goal_extractor = GoalExtractor(env.observation_space)

agent = SGGHRLAgent(
    env=env,
    goal_extractor=goal_extractor,
    worker_env_class=YourWorkerEnv,
    manager_env_class=YourManagerEnv,
    max_worker_steps=15,
    success_threshold=0.5,
    seed=42,
)

# 1. Обучить Worker
agent.train_worker(total_timesteps=200_000, callbacks=CallbackList([
    RollingEpisodeStatsCallback(),
    ProgressPrinterCallback(log_interval=2000),
]))

# 2. Обучить Manager
agent.train_manager(total_timesteps=100_000, callbacks=CallbackList([
    RollingEpisodeStatsCallback(),
    ProgressPrinterCallback(log_interval=1000),
    SACTrainCallback(learning_starts=1000),
    ManagerEpsilonGreedyExplorationCallback(),
    HERCallback(k_future=2),
]))

# 3. Демо
agent.set_render_mode("human")
agent.demo(n_episodes=5)
```

## Воспроизведение экспериментов

Эксперименты сравнивают три подхода на среде DungeonHeist: **Vanilla PPO**, **Vanilla HRL** (без графовых компонентов) и **SGGHRL** (полный метод).

Среды: `easy-16`, `easy-25`, `hard-16`, `hard-25`. Каждый алгоритм запускается с 3 сидами (42, 123, 999).

### Запуск одного эксперимента

```bash
python -m experiments.run_experiment --algo sgghrl --env hard-16 --seed 42
```

По умолчанию `--phase all` обучает и Worker, и Manager. Можно разделить:

```bash
# Только Worker
python -m experiments.run_experiment --algo hrl --env hard-16 --seed 42 --phase worker

# Только Manager (Worker должен быть обучен)
python -m experiments.run_experiment --algo hrl --env hard-16 --seed 42 --phase manager
```

Результат — файл `experiment_results.png` с кривыми обучения и дcоверительными интервалами.

По умолчанию `--phase all` обучает и Worker, и Manager. Можно разделить:
```bash
# Только Worker
python -m experiments.run_experiment --algo hrl --seed 42 --phase worker

# Только Manager (Worker должен быть обучен)
python -m experiments.run_experiment --algo hrl --seed 42 --phase manager
```

### Построение графиков

```bash
python -m experiments.plot_results --env hard-16
```

Результат — файл `experiment_results_hard-16.png` с кривыми обучения и доверительными интервалами.

## Основные компоненты

### Колбэки обучения

| Колбэк | Описание |
|--------|----------|
| `RollingEpisodeStatsCallback` | Скользящая статистика наград и success rate |
| `ProgressPrinterCallback` | Вывод прогресса с ETA |
| `WorkerCurriculumCallback` | Автоматическое продвижение по стадиям curriculum |
| `WorkerBestCheckpointCallback` | Сохранение лучших весов Worker'а |
| `ManagerEpsilonGreedyExplorationCallback` | Epsilon-greedy с линейным затуханием |
| `ManagerEpsilonWithBurstsCallback` | Epsilon-greedy с всплесками при падении SR |
| `SACTrainCallback` | Запуск обучения SAC с заданной частотой |
| `HERCallback` | Hindsight Experience Replay |
| `GraphExplorationBonusCallback` | Exploration bonus на основе visit count |
| `DeltaDistanceShapingCallback` | Delta-distance reward shaping через граф |
| `AdaptiveWorkerBudgetCallback` | Адаптивный бюджет шагов Worker'а |
| `FrontierExplorationBonusCallback` | Бонус за движение к frontier-узлам |
| `ManagerLRDecayOnPlateauCallback` | Снижение LR при стагнации |
| `ManagerEvalCallback` | Периодическая оценка Manager'а |
| `ManagerBestCheckpointOnEvalCallback` | Сохранение лучших весов по eval |
| `CheckpointCallback` | Периодическое сохранение модели |
| `BufferCheckpointCallback` | Периодическое сохранение replay buffer |

### Стратегии инференса

| Стратегия | Описание |
|-----------|----------|
| `PolicyOnlyStrategy` | Чистая policy без графа |
| `GraphPlannerStrategy` | Граф генерирует кандидатов, critic выбирает лучшего |

### Утилиты нейросетей

`set_ppo_params`, `set_sac_params` — изменение гиперпараметров на лету.
`resize_layer`, `resize_network` — изменение архитектуры с сохранением весов.
`freeze_layers`, `unfreeze_all`, `freeze_all_except` — управление обучаемостью слоёв.

## Лицензия

MIT
