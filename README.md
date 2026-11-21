# DIAYN: multi-seed reproduction

A PyTorch reimplementation of *Diversity is All You Need* (Eysenbach et al., 2018), kept as close as possible to the authors' code ([ben-eysenbach/sac](https://github.com/ben-eysenbach/sac), [DIAYN.md](https://github.com/ben-eysenbach/sac/blob/master/DIAYN.md)), on current Gymnasium and MuJoCo.

The paper reports single runs. This project runs each experiment over several seeds and reports means with 95% intervals.

## Layout

| Path | Role |
|---|---|
| `diayn/config.py` | Hyperparameters of the original launcher (`SHARED_PARAMS`, `ENV_PARAMS`) |
| `diayn/agent.py` | SAC v1 (Q, V, target V, GMM policy) with the DIAYN discriminator; replay buffer |
| `diayn/envs.py` | Env construction, rllab-style action normalisation, 2D point env |
| `diayn/evaluation.py` | Rollout of all skills in lockstep; snapshot loading |
| `train.py` | One run (env, seed). Resumes from `checkpoint.pt` |
| `sweep.py` | Several seeds of one configuration in parallel processes |
| `aggregate.py` | Statistics and figures across seeds and configurations |
| `plot_traces.py`, `visualize_skills.py` | Per-skill trajectories and videos (the original's `scripts/`) |
| `onyxia/` | Setup and launch scripts for the SSP Cloud / Onyxia datalab |

## Local setup (Windows)

Drive C: is full, so the env lives on D:. Run this from an Anaconda Prompt, or from a PowerShell where `conda init` has been done:

```powershell
$env:CONDA_PKGS_DIRS = "D:\conda\pkgs"; $env:TEMP = "D:\conda\tmp"; $env:TMP = "D:\conda\tmp"
conda create -p D:\conda\envs\diayn python=3.11 pip -y
conda activate D:\conda\envs\diayn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The env already exists at `D:\conda\envs\diayn`. Run every command from this folder (`Robotics/Project`).

On this machine `results/` is a directory junction to `D:\diayn\results`, because C: is full. Recreate it on a fresh clone with `cmd /c mklink /J results D:\diayn\results`.

## Usage

```bash
# One run
python train.py --env point --seed 1 --log_dir results/point/default/seed_1

# Five seeds, three at a time. Rerunning the same command resumes unfinished runs.
python sweep.py --env hopper --seeds 1 2 3 4 5 --jobs 3

# A variant of the configuration: any key of diayn/config.py can be overridden
python sweep.py --env hopper --seeds 1 2 3 4 5 --jobs 3 --tag alpha0.01 --scale_entropy 0.01

# Statistics across seeds, comparing configurations
python aggregate.py results/hopper/default results/hopper/alpha0.01 --out figures/hopper

# Skills of one run
python plot_traces.py results/point/default/seed_1/snapshots/itr_50.pt
python plot_traces.py results/ant/default/seed_1/snapshots/itr_1000.pt --x info:x_position --y info:y_position
python visualize_skills.py results/hopper/default/seed_1/snapshots/itr_1000.pt --skills 0 1 2
```

Envs: `point`, `pendulum`, `mountain-car`, `inverted-pendulum`, `inverted-double-pendulum`, `swimmer`, `hopper`, `half-cheetah`, `walker`, `ant`, `humanoid`.

### What a run writes (`results/<env>/<tag>/seed_<s>/`)

- `config.json`: the full configuration of the run.
- `progress.csv`: one row per epoch (1,000 env steps).
  - Training diagnostics, averaged over the epoch: `qf`, `vf`, `bellman_error`, `discriminator_loss`, `discriminator_acc`, `discriminator_reward`, `log_pi`, `kl_surrogate_loss`, `vf_loss`.
  - Mean env return of the training episodes.
  - On evaluation epochs (every 10), the best skill, its mean env return, the mean return over skills, and the mean log q(z|s).
- `skill_eval.csv`: on every evaluation epoch, each skill's mean and std env return over 10 stochastic rollouts, and its mean log q(z|s). This is the raw data for the paper's skill-reward figures.
- `snapshots/itr_<n>.pt`: policy, Q function and discriminator, every 100 epochs.
- `checkpoint.pt`: everything needed to resume (nets, optimiser, replay buffer, RNG state), overwritten every 100 epochs.

The env reward is never used for training. It is only logged, to see whether some skills solve the task (one of the paper's claims).

## Compute: why the GPU

Measured on the local i7-10700 (one process): a gradient step costs about **10 ms**, and acting plus `env.step` about 0.5 ms. The 300-unit networks make the update compute-bound, and extra CPU threads barely help (95 → 124 steps/s with 4 threads).

| Run | Env steps | CPU time per seed |
|---|---|---|
| point (default) | 51k | ~5 min |
| pendulum, mountain-car, inverted pendulums (1,000 epochs) | 1M | ~1.5–3 h |
| hopper, half-cheetah, walker, ant at the paper's 10,000 epochs | 10M | ~30 h |
| same, cut to 3,000 epochs (the authors' demo snapshots are `itr_3000`) | 3M | ~9 h |

The small envs run fine locally. Use Onyxia's GPU for the MuJoCo locomotion envs. Several runs can share one GPU (`--jobs`), as each uses one CPU core and a few hundred MB of GPU memory. The GPU speed-up was not measured here: `onyxia/setup.sh` prints the steps/s on the pod.

To shorten runs, pass `--n_epochs 3000`. The statistics and figures do not depend on the run length.

## Onyxia (SSP Cloud)

1. Start a **vscode-pytorch-gpu** service (or jupyter-pytorch-gpu) with 1 GPU, at least as many CPUs as runs at once, and enough RAM. A full replay buffer takes about 1.7 GB for HalfCheetah (pool 1e7), 0.9 GB for Ant (pool 1e6) and less for the others. Each `checkpoint.pt` is about the same size, on disk and in S3. Plan for about 3 GB of RAM per HalfCheetah run.
2. Get this folder onto the pod: `git clone` your repo, or upload it.
3. `bash onyxia/setup.sh`: installs the requirements, checks CUDA and MuJoCo, and runs 2 training epochs.
4. Launch a sweep. It keeps running after the browser tab closes, and `results/` is mirrored to your S3 bucket every 15 minutes (`mc` is preconfigured on Onyxia; check with `mc ls s3/<your-username>`):
   ```bash
   bash onyxia/run.sh s3/<your-username>/diayn --env half-cheetah --seeds 1 2 3 4 5 --jobs 5
   ```
5. If the service is deleted or restarted, start a new one, restore with `mc mirror s3/<your-username>/diayn/results results`, and run the same command again. Each run resumes from its checkpoint.
6. Fetch the results locally (`mc` or the Onyxia file browser), then run `aggregate.py`, `plot_traces.py` and `visualize_skills.py` on your machine. Videos need an OpenGL display. On the pod, set `MUJOCO_GL=egl`.

## Fidelity to the original code

**Same as the original:**
- **Hyperparameters:** lr 3e-4 (all nets), discount 0.99, tau 0.01, batch 128, 300×300 ReLU MLPs (32 for point and pendulum), Xavier init, 50 skills, `scale_entropy` (α) 0.1, `epoch_length` 1000, replay pool 1e6 (1e7 for HalfCheetah), one gradient step per env step after `max_path_length` samples.
- **SAC v1:** a Q function, a V function, and a target V that is initialised independently and soft-updated every step.
- **GMM policy** with K = 4 components:
  - log σ is clipped to [-5, 2] and log w floored at -10, with L2 regularisation 1e-3 on the means and log σ.
  - Actions are squashed by tanh, the sample is stop-gradiented, and the actor uses the likelihood-ratio surrogate.
  - The deterministic action is the mean of the highest-Q component.
- **Discriminator** q(z|s): sees the state only (no actions). Pseudo-reward log q(z|s_t) − log p(z) on the current state, recomputed with the current discriminator for every batch. p(z) is uniform and fixed. `add_p_z` is off for mountain-car.
- **Updates:** all nets take their gradient step from the same parameters, as in the single TF `sess.run`.
- **Skill sampling:** a new skill once per epoch, not per episode, and episodes can span epochs. Time-limit ends are not terminal.
- **Actions:** the agent acts in [-1, 1], rescaled to the env bounds and clipped (rllab `normalize`). No observation or reward normalisation.
- **Best-skill evaluation:** every 10 epochs, 10 stochastic rollouts per skill.

**Deviations:**
- **Envs:** Gymnasium v5 and MuJoCo 4 instead of gym v1 and MuJoCo 1.31. The observations have the same layout, but Ant has 105 dims instead of 111 and Humanoid 348 instead of 376 (the world-body contact forces are dropped), and the physics differ slightly. Box2D envs are not included.
- **Point env:** `point-rllab` is not defined in the authors' repo. `PointEnv` reconstructs rllab's tutorial point env. Treat point results as qualitative.
- **First step after a reset:** the original kept the pre-reset observation for one step. Here the agent acts on the new observation.
- **Evaluation:** the original only logged the best skill's return. Every skill's return and log q(z|s) are now logged. The original's single deterministic rollout of the best skill is dropped.
- **Snapshots** every 100 epochs instead of every 10, to save storage.
- **Network init:** the original's first layer sums one affine map per input instead of using one layer on the concatenated inputs. The two are equivalent except for a slightly different Xavier scale.

## Extending: prior experiments

The prior p(z) is `agent.p_z` (a tensor). It is used both to sample the skill each epoch (`train.py`) and in the pseudo-reward (`DIAYN.update`). It is saved in snapshots and checkpoints. A learned or non-uniform prior only needs to change this tensor. The original's `learn_p_z` option, a softmax over each skill's mean log q(z|s), is not implemented yet.
