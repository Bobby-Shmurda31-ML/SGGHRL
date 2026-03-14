import subprocess
import os
import sys
import time
import threading

_print_lock = threading.Lock()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

PYTHON = sys.executable
ENVS = ["easy-16", "easy-25", "hard-16", "hard-25"]
SEEDS = [42, 123, 999]

PARALLEL_PPO_N = 3
PARALLEL_WORKERS_N = 3
PARALLEL_MANAGERS_N = 2


def _stream_output(pipe, prefix, log_file):
    for line in iter(pipe.readline, ''):
        line = line.rstrip()
        if line:
            with _print_lock:
                print(f"  [{prefix}] {line}")
            log_file.write(line + "\n")
            log_file.flush()
    pipe.close()


def _stream_output(pipe, prefix, log_file):
    """Читать stdout процесса построчно, печатать с префиксом и писать в файл."""
    for line in iter(pipe.readline, ''):
        line = line.rstrip()
        if line:
            print(f"  [{prefix}] {line}")
            log_file.write(line + "\n")
            log_file.flush()
    pipe.close()


def run_batch(tasks, max_parallel):
    """Запустить задачи батчами по max_parallel с live-выводом логов."""
    for i in range(0, len(tasks), max_parallel):
        batch = tasks[i:i + max_parallel]
        procs = []

        for algo, env, seed, phase in batch:
            log_name = f"{env}_{algo}_seed{seed}"
            if phase != "all":
                log_name += f"_{phase}"
            log_path = os.path.join(LOG_DIR, f"{log_name}.log")

            cmd = [
                PYTHON, "-u", "-m", "experiments.run_experiment",
                "--algo", algo, "--env", env, "--seed", str(seed),
                "--phase", phase,
            ]

            print(f"  START: {' '.join(cmd[3:])} → {log_path}")
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            thread = threading.Thread(
                target=_stream_output,
                args=(proc.stdout, log_name, log_file),
                daemon=True,
            )
            thread.start()
            procs.append((proc, thread, log_file, log_name))

        for proc, thread, log_file, name in procs:
            proc.wait()
            thread.join(timeout=5)
            log_file.close()
            status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
            print(f"  DONE:  {name} [{status}]")
        print()


if __name__ == "__main__":
    start = time.time()

    # Все PPO
    print("PPO (parallel x%d)" % PARALLEL_PPO_N)
    ppo_tasks = [
        ("ppo", env, str(seed), "all")
        for env in ENVS for seed in SEEDS
    ]
    run_batch(ppo_tasks, PARALLEL_PPO_N)

    # Все worker'ы
    print("Workers (parallel x%d)" % PARALLEL_WORKERS_N)
    worker_tasks = [
        ("hrl", env, str(seed), "worker")
        for env in ENVS for seed in SEEDS
    ]
    run_batch(worker_tasks, PARALLEL_WORKERS_N)

    # Все HRL manager'ы
    print("HRL managers (parallel x%d)" % PARALLEL_MANAGERS_N)
    hrl_tasks = [
        ("hrl", env, str(seed), "manager")
        for env in ENVS for seed in SEEDS
    ]
    run_batch(hrl_tasks, PARALLEL_MANAGERS_N)

    # Все SGGHRL manager'ы
    print("SGGHRL managers (parallel x%d)" % PARALLEL_MANAGERS_N)
    sgghrl_tasks = [
        ("sgghrl", env, str(seed), "manager")
        for env in ENVS for seed in SEEDS
    ]
    run_batch(sgghrl_tasks, PARALLEL_MANAGERS_N)

    # Графики
    print("Plotting")
    for env in ENVS:
        subprocess.run([
            PYTHON, "-m", "experiments.plot_results",
            "--env", env
        ])

    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\nЭксперименты закончены ({hours}h {minutes}m)")
    print(f"Логи: {LOG_DIR}/")
    print("Графики: experiment_results_*.png")