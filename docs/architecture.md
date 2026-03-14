# Архитектура SGGHRL

Метод Subgoal Graph-Guided Hierarchical RL (SGGHRL) состоит из нескольких ключевых компонентов, интегрированных через систему колбэков.

## 1. Двухуровневая иерархия

- **Manager (SAC)** — высокоуровневая политика. Наблюдает глобальное состояние среды (позиции объектов, локальную карту, историю подцелей, новизну) и выдаёт подцель (2D координату) для Worker'а.
- **Worker (PPO)** — низкоуровневая goal-conditioned политика. Получает подцель и навигирует к ней за ограниченное число шагов, используя локальную карту и направление к цели.

Manager выбирает подцель → Worker исполняет её → Manager получает награду и следующее наблюдение.

## 2. State Graph (Граф состояний)

Граф `G = (V, E)` строится динамически по мере исследования среды.

- **Узлы** `v ∈ V` — дискретизированные наблюдения (координаты квантуются с шагом `discretization`). Каждый узел хранит счётчик посещений.
- **Рёбра** `(u, v) ∈ E` — зарегистрированные переходы между состояниями. Хранят количество переходов и суммарную награду.
- **Контекст** — опциональная функция `context_fn(raw_obs) → hashable`, результат добавляется к ключу узла. Пример: `has_key` — граф различает состояния «до ключа» и «после ключа».
- **Frontier** — узлы, у которых хотя бы один сосед по сетке отсутствует в графе (и проходит фильтр `valid_state_fn`).

При превышении `max_nodes` удаляется наименее посещённый листовой узел.

Класс: `sgghrl.core.graph.StateGraph`

## 3. Delta-Distance Reward Shaping

Шейпинг на основе изменения графового расстояния до цели за один шаг manager'а:

- `delta > 0` (приблизился): `shaping = +delta × scale`
- `delta < 0` (удалился): `shaping = delta × scale × retreat_multiplier`
- `delta = 0` (на месте): `shaping = 0`

Свойства:
- **Анти-осцилляция:** при `retreat_multiplier > 1` штраф за удаление сильнее награды за приближение. Агент, бегающий туда-сюда, получает отрицательный суммарный шейпинг.
- **Нет бесплатных наград:** стояние на месте даёт 0.
- **Телескопирование:** сумма за эпизод ≈ `scale × (dist_start - dist_end)`.
- **Auto-scale:** отслеживается EMA абсолютной награды, `scale = EMA × shaping_ratio / max_delta_estimate`.

Колбэк: `DeltaDistanceShapingCallback`

## 4. Exploration Bonus

Два механизма:

### 4.1 Visit-count bonus
```
bonus = α / √(visit_count(s'))
```
Затухает линейно за `decay_steps`. Состояния с `bonus < min_bonus` не получают бонуса (убирает шум от часто посещённых).

Колбэк: `GraphExplorationBonusCallback`

### 4.2 Frontier bonus
```
bonus = α / ((1 + dist_to_frontier) × √visit_count)
```
Multi-source BFS от всех frontier-узлов, кэшируется с интервалом `cache_interval`.

Колбэк: `FrontierExplorationBonusCallback`

## 5. Graph Planner Strategy (Инференс)

На этапе инференса стандартный HRL использует только `a ~ π_θ(s)`. SGGHRL генерирует кандидатов из нескольких источников:

1. **Policy samples** — несколько стохастических + один детерминированный выход SAC.
2. **Соседи в графе** — непосредственные соседи текущего узла.
3. **Узлы на кратчайшем пути** к финальной цели (BFS + восстановление пути).
4. **Frontier-узлы** — ближайшие узлы на границе исследованного пространства.

Каждый кандидат конвертируется в action через `goal_to_action()`, SAC critic оценивает `Q(s, a_i)`, выбирается лучший. Резко снижает вероятность выбора цели за стеной.

Класс: `sgghrl.core.inference.GraphPlannerStrategy`

## 6. Adaptive Worker Budget

Вместо фиксированного бюджета шагов Worker'а, SGGHRL выделяет бюджет динамически:

```
budget = base_steps + steps_per_hop × d_graph(s, subgoal)
budget = clip(budget, min_steps, max_steps)
```

Если граф не может оценить расстояние — используется `default_steps`.

Колбэк: `AdaptiveWorkerBudgetCallback`

## 7. Hindsight Experience Replay (HER)

В конце каждого эпизода Manager'а генерируются дополнительные переходы с подменёнными целями:

- **future** — цели из будущих шагов эпизода (основная стратегия).
- **final** — цель = конечное состояние эпизода.
- **episode** — случайные цели из всего эпизода.

Если Manager-среда реализует протокол `HERCapable` — используются её методы `get_achieved_goal`, `compute_her_reward`, `relabel_obs_for_her`. Иначе — fallback на `GoalExtractor` со sparse-наградой.

Колбэк: `HERCallback`, буфер: `HERBuffer`

## 8. Curriculum Learning (Worker)

Worker обучается по стадиям — от близких целей к дальним:

1. На каждой стадии задаётся `max_distance` и `target_success_rate`.
2. Веса сэмплирования целей по расстоянию формируются экспоненциально (дальние цели — выше вес).
3. При достижении `target_success_rate` — переход на следующую стадию.
4. При падении ниже `rollback_threshold × prev_target` — откат назад.

Требует реализации протокола `CurriculumCapable` в Worker-среде.

Колбэк: `WorkerCurriculumCallback`

## 9. Система колбэков

Все компоненты подключаются как независимые колбэки (`SGGHRLCallback`), объединяемые в `CallbackList`. Хуки:

- `on_training_start` / `on_training_end` — инициализация / финализация.
- `before_action` — модификация действия (epsilon-greedy).
- `after_step` — модификация награды (shaping, bonus), обучение SAC.
- `on_episode_end` — статистика, HER, eval.
- `on_rollout_end` — после rollout'а PPO.
- `on_log` — логирование, чекпоинты.
- `on_eval_end` — после evaluation.