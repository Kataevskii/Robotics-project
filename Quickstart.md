# Quickstart: DIAYN workspace

Run every command from `C:\dev\MVA\Robotics\Project`.

## 1. Environment

    conda activate D:\conda\envs\diayn

If `conda` isn't recognized, replace `python` below with `D:\conda\envs\diayn\python.exe`.
The env lives on D: because C: is full. Recreate it with the "Local setup" section of README.md.

## 2. One run

    python train.py --env point --seed 1 --log_dir results/point/default/seed_1

Envs: point, pendulum, mountain-car, inverted-pendulum, inverted-double-pendulum,
swimmer, hopper, half-cheetah, walker, ant, humanoid.
Any key of `diayn/config.py` can be overridden, e.g. `--num_skills 6 --n_epochs 500`.

## 3. Several seeds (parallel)

    python sweep.py --env pendulum --seeds 1 2 3 4 5 --jobs 3
    python sweep.py --env point --seeds 1 2 3 4 5 --jobs 3 --tag k6 --num_skills 6

- Output goes to `results/<env>/<tag>/seed_<s>/`, with logs in `stdout.log`.
- Rerunning the same command resumes unfinished runs; finished ones are skipped.
- Keep `--jobs` at or below 8 (CPU cores).

Follow a run live:

    Get-Content results\pendulum\default\seed_1\stdout.log -Wait -Tail 5

## 4. Reproduce the cheap experiments (5 seeds)

| Env                     | Command                                                                            | Time (local CPU) |
| ----------------------- | ---------------------------------------------------------------------------------- | ---------------- |
| Mountain car            | `python sweep.py --env mountain-car --seeds 1 2 3 4 5 --jobs 5`                    | ~4–5 h           |
| Pendulum                | `python sweep.py --env pendulum --seeds 1 2 3 4 5 --jobs 3`                        | ~3.5 h           |
| Point (original config) | `python sweep.py --env point --seeds 1 2 3 4 5 --jobs 3`                           | minutes          |
| Point, 6 skills         | `python sweep.py --env point --seeds 1 2 3 4 5 --jobs 3 --tag k6 --num_skills 6`   | minutes          |
| Point, 10× longer       | `python sweep.py --env point --seeds 1 2 3 4 5 --jobs 3 --tag long --n_epochs 500` | ~1.5 h           |

Turn off Windows sleep during long runs.

## 5. Statistics and exported data

    python aggregate.py results/mountain-car/default --out figures/mountain-car
    python aggregate.py results/pendulum/default --out figures/pendulum
    python aggregate.py results/point/default results/point/long results/point/k6 --out figures/point

Each `figures/<env>/` folder contains:

- `summary.csv`: final values, mean, std and 95% CI across seeds of
  - `diversity`: 1 + E[log q(z|s)]/log K; 0 = chance, 1 = perfectly distinguishable
  - `frac_distinct`: share of skills above chance
  - `epochs_to_90`: epochs to reach 90% of final diversity
  - the returns of the best skill and the mean over skills
- `summary_per_seed.csv`: the same values for each seed.
- `curves.csv`, `diversity.png`, `returns.png`: curves over training, with the mean and 95% CI.
- `eval_all_runs.csv`, `skills_all_runs.csv`: every raw evaluation row, overall and per skill.
- `skill_returns.png`: env return of every skill at the end, per seed.

## 6. Look at the skills

    python plot_traces.py results/point/k6/seed_1/snapshots/itr_50.pt --n_paths 3 --stochastic
    python plot_traces.py results/mountain-car/default/seed_1/snapshots/itr_1000.pt --x t --y obs:0
    python plot_traces.py results/pendulum/default/seed_1/snapshots/itr_1000.pt --x t --y obs:2 --n_paths 3
    python visualize_skills.py results/hopper/default/seed_1/snapshots/itr_1000.pt --skills 0 1 2

- `plot_traces.py` axes: `obs:<i>` (observation dimension), `info:<key>` (e.g. `info:x_position`), or `t` (time step).
- Videos only work for MuJoCo envs.

## 7. MuJoCo envs on Onyxia (GPU)

    bash onyxia/setup.sh
    bash onyxia/run.sh s3/<your-username>/diayn --env half-cheetah --seeds 1 2 3 4 5 --jobs 5

To resume in a new service:

    mc mirror s3/<your-username>/diayn/results results

then rerun the same `onyxia/run.sh` command.

## Layout

- `diayn/`: the package
  - `config.py`: hyperparameters
  - `agent.py`: SAC v1, GMM policy, discriminator
  - `envs.py`, `evaluation.py`, `plotting.py`
- `train.py`, `sweep.py`, `aggregate.py`, `plot_traces.py`, `visualize_skills.py`: entry points
- `results/`, `figures/`: outputs, git-ignored
- `README.md`: full details, compute budget, fidelity to the original code
