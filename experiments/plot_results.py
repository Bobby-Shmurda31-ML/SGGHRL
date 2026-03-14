import json
import os
import re
import collections
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import FuncFormatter

def _millions_formatter(x, pos):
    return f"{x / 1e6:.1f}M"

N_INTERP_POINTS = 200

ENV_CHOICES = ["hard-16", "hard-25", "easy-16", "easy-25"]
CONFIGS = {
    "ppo": {"label": "Vanilla PPO", "color": "#E91E63"},
    "hrl": {"label": "Vanilla HRL", "color": "#FF9800"},
    "sgghrl": {"label": "SGGHRL (My Method)", "color": "#4CAF50"},
}


def _discover_seeds(algo, env_name, results_dir="results"):
    seeds = {}
    if not os.path.isdir(results_dir):
        return []

    pattern_fair = re.compile(
        rf"^fair_{re.escape(env_name)}_{re.escape(algo)}_seed(\d+)\.json$"
    )
    pattern_norm = re.compile(
        rf"^{re.escape(env_name)}_{re.escape(algo)}_seed(\d+)\.json$"
    )

    for fname in os.listdir(results_dir):
        m = pattern_fair.match(fname)
        if m:
            seeds[int(m.group(1))] = "fair"
            continue
        m = pattern_norm.match(fname)
        if m:
            seed = int(m.group(1))
            if seed not in seeds:
                seeds[seed] = "normal"

    return sorted(seeds.items())


def _read_raw(algo, env_name, results_dir="results"):
    seed_info = _discover_seeds(algo, env_name, results_dir)
    curves = []

    for seed, ftype in seed_info:
        if ftype == "fair":
            path = os.path.join(results_dir, f"fair_{env_name}_{algo}_seed{seed}.json")
        else:
            path = os.path.join(results_dir, f"{env_name}_{algo}_seed{seed}.json")

        with open(path, "r") as f:
            data = json.load(f).get("history", [])
        if not data:
            continue

        steps, sr, rew = [], [], []
        recent_success = collections.deque(maxlen=25)

        for d in data:
            steps.append(d.get("env_steps_total", d.get("step", 0)))

            raw_r = d.get("eval_avg_raw_reward")
            if raw_r is None:
                raw_r = d.get("eval_avg_reward")
            if raw_r is None:
                raw_r = d.get("avg_reward_original")
            if raw_r is None:
                raw_r = d.get("avg_reward", 0.0)
            rew.append(raw_r)

            if "eval_success_rate" in d:
                sr.append(d["eval_success_rate"])
            elif "success_rate" in d:
                sr.append(d["success_rate"])
            elif "success" in d:
                recent_success.append(d["success"])
                sr.append(float(np.mean(recent_success)))
            else:
                sr.append(0.0)

        if not steps:
            continue

        curves.append({"seed": seed, "steps": steps, "sr": sr, "rew": rew})

    return curves


def _interpolate(curves, common_steps):
    """Интерполировать все кривые к единой оси X."""
    all_sr, all_rew = [], []
    for c in curves:
        all_sr.append(np.interp(common_steps, c["steps"], c["sr"]))
        all_rew.append(np.interp(common_steps, c["steps"], c["rew"]))
    return np.array(all_sr), np.array(all_rew)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=ENV_CHOICES, default="hard-16",
                        help="Тип среды для отображения")
    args = parser.parse_args()
    env_name = args.env

    raw = {}
    for algo in CONFIGS:
        raw[algo] = _read_raw(algo, env_name)

    if not any(raw[a] for a in CONFIGS):
        print(f"Нет данных для '{env_name}' в results/. Сначала запустите эксперименты.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for algo, cfg in CONFIGS.items():
        curves = raw[algo]
        if not curves:
            continue

        algo_min = min(min(c["steps"]) for c in curves if c["steps"])
        algo_max = max(max(c["steps"]) for c in curves if c["steps"])
        if algo_max <= algo_min:
            continue

        algo_steps = np.linspace(algo_min, algo_max, N_INTERP_POINTS)
        n_seeds = len(curves)
        all_sr, all_rew = _interpolate(curves, algo_steps)
        mean_sr, std_sr = all_sr.mean(0), all_sr.std(0)
        mean_r, std_r = all_rew.mean(0), all_rew.std(0)

        label = f"{cfg['label']} ({n_seeds} seeds)"

        ax1.plot(algo_steps, mean_sr * 100,
                 label=label, color=cfg["color"], linewidth=2.5)
        ax1.fill_between(algo_steps,
                         (mean_sr - std_sr) * 100, (mean_sr + std_sr) * 100,
                         color=cfg["color"], alpha=0.15)

        ax2.plot(algo_steps, mean_r,
                 label=label, color=cfg["color"], linewidth=2.5)
        ax2.fill_between(algo_steps,
                         mean_r - std_r, mean_r + std_r,
                         color=cfg["color"], alpha=0.15)

    ax1.xaxis.set_major_formatter(FuncFormatter(_millions_formatter))
    ax2.xaxis.set_major_formatter(FuncFormatter(_millions_formatter))

    ax1.set(title=f"Success Rate vs Env Steps [{env_name}]",
            xlabel="Environment Steps", ylabel="Success Rate (%)",
            ylim=(-5, 105))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set(title=f"True Reward vs Env Steps [{env_name}]",
            xlabel="Environment Steps", ylabel="Episodic Reward (Original)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    out = f"experiment_results_{env_name}.png"
    plt.savefig(out, dpi=300)
    print(f"График сохранён в '{out}'")
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()