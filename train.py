"""Train DIAYN on one env and seed. Resumes automatically from <log_dir>/checkpoint.pt.

    python train.py --env half-cheetah --seed 1 --log_dir results/half-cheetah/default/seed_1
    python train.py --env point --seed 1 --log_dir results/point/default/seed_1 --num_skills 10

Any hyperparameter of diayn/config.py can be overridden with --<name> <value>.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diayn.agent import DIAYN, ReplayBuffer, STAT_NAMES
from diayn.config import ENV_PARAMS, SHARED_PARAMS, get_config
from diayn.envs import make_env
from diayn.evaluation import rollout_skills

CFG_KEYS = [*SHARED_PARAMS, 'max_path_length', 'n_epochs']
PROGRESS_COLS = ['epoch', 'total_steps', 'time', 'steps_per_s', 'episodes', 'train_return',
                 *STAT_NAMES, 'best_skill', 'best_skill_return', 'mean_skill_return', 'eval_logq']
SKILL_COLS = ['epoch', 'skill', 'return_mean', 'return_std', 'logq_mean']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', required=True, choices=list(ENV_PARAMS))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--log_dir', required=True)
    p.add_argument('--device', default='auto', help="'auto', 'cpu' or 'cuda'")
    p.add_argument('--threads', type=int, default=1, help='torch CPU threads per run')
    for k in CFG_KEYS:
        v = SHARED_PARAMS.get(k, 1)
        if isinstance(v, bool):
            typ = lambda s: s.lower() in ('1', 'true', 'yes')
        elif isinstance(v, int):
            typ = lambda s: int(float(s))  # accepts 1e7
        else:
            typ = float
        p.add_argument(f'--{k}', type=typ, default=None)
    return p.parse_args()


class CsvLog:
    def __init__(self, path, cols):
        new = not path.exists()
        self.f = open(path, 'a', newline='')
        self.w = csv.DictWriter(self.f, cols)
        if new:
            self.w.writeheader()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    (log_dir / 'snapshots').mkdir(parents=True, exist_ok=True)
    ckpt_path = log_dir / 'checkpoint.pt'
    resume = ckpt_path.exists()
    if resume:  # the run keeps its original config, CLI overrides are ignored
        cfg = json.loads((log_dir / 'config.json').read_text())
    else:
        cfg = get_config(args.env, **{k: getattr(args, k) for k in CFG_KEYS})
        cfg['seed'] = args.seed
        (log_dir / 'config.json').write_text(json.dumps(cfg, indent=2))
        for name in ('progress.csv', 'skill_eval.csv'):
            (log_dir / name).unlink(missing_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    seed, S, L = cfg['seed'], cfg['num_skills'], cfg['max_path_length']
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = make_env(cfg['env_id'], L)
    eval_envs = [make_env(cfg['env_id'], L) for _ in range(S)]
    for i, e in enumerate(eval_envs):
        e.reset(seed=10_000 * seed + i)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = DIAYN(obs_dim, act_dim, cfg, device)
    buffer = ReplayBuffer(obs_dim, act_dim, cfg['max_pool_size'])

    start_epoch, total_steps, elapsed = 0, 0, 0.0
    if resume:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ck['agent'])
        buffer.load_state_dict(ck['buffer'])
        rng.bit_generator.state = ck['np_rng']
        torch.set_rng_state(ck['torch_rng'])
        start_epoch, total_steps, elapsed = ck['epoch'] + 1, ck['total_steps'], ck['time']
        for name in ('progress.csv', 'skill_eval.csv'):
            path = log_dir / name
            if path.exists():
                df = pd.read_csv(path)
                df[df.epoch <= ck['epoch']].to_csv(path, index=False)
        print(f'Resumed from epoch {ck["epoch"]}')

    progress = CsvLog(log_dir / 'progress.csv', PROGRESS_COLS)
    skill_log = CsvLog(log_dir / 'skill_eval.csv', SKILL_COLS)
    obs, _ = env.reset(seed=seed + start_epoch)
    ep_return, finished = 0.0, []

    for epoch in range(start_epoch, cfg['n_epochs'] + 1):
        t0 = time.time()
        # As the original, the skill is resampled once per epoch, not per episode.
        z = int(rng.choice(S, p=agent.p_z.cpu().numpy()))
        stats, n_updates = torch.zeros(len(STAT_NAMES), device=device), 0
        for _ in range(cfg['epoch_length']):
            a = agent.act(obs[None], [z])[0]
            next_obs, r, term, trunc, _ = env.step(a)
            buffer.add(obs=obs, next_obs=next_obs, act=a, done=float(term), z=z)
            ep_return += r
            total_steps += 1
            if term or trunc:
                finished.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
            if buffer.size >= L:  # min_pool_size = max_path_length
                for _ in range(cfg['n_train_repeat']):
                    stats += agent.update(buffer.sample(cfg['batch_size'], rng, device))
                    n_updates += 1
        t_train = time.time() - t0

        row = dict(epoch=epoch, total_steps=total_steps, episodes=len(finished),
                   steps_per_s=round(cfg['epoch_length'] / t_train, 1),
                   train_return=np.mean(finished) if finished else '')
        finished = []
        if n_updates:
            row.update(zip(STAT_NAMES, (stats / n_updates).tolist()))

        if epoch % cfg['find_best_skill_interval'] == 0:
            # Stochastic rollouts of every skill, as the original's best-skill selection.
            rets, logqs = zip(*[rollout_skills(agent, eval_envs, L)[:2]
                                for _ in range(cfg['best_skill_n_rollouts'])])
            rets, logqs = np.array(rets), np.array(logqs)  # (n_rollouts, S)
            mean_ret = rets.mean(0)
            best = int(np.argmax(mean_ret))
            row.update(best_skill=best, best_skill_return=mean_ret[best],
                       mean_skill_return=mean_ret.mean(), eval_logq=logqs.mean())
            for s in range(S):
                skill_log.write(dict(epoch=epoch, skill=s, return_mean=mean_ret[s],
                                     return_std=rets[:, s].std(), logq_mean=logqs[:, s].mean()))

        elapsed += time.time() - t0
        row['time'] = round(elapsed, 1)
        progress.write(row)

        last = epoch == cfg['n_epochs']
        if epoch % cfg['snapshot_gap'] == 0 or last:
            sd = agent.state_dict()
            torch.save({k: sd[k] for k in ('policy', 'qf', 'disc', 'p_z')},
                       log_dir / 'snapshots' / f'itr_{epoch}.pt')
        if epoch % cfg['checkpoint_interval'] == 0 or last:
            tmp = ckpt_path.with_suffix('.tmp')
            torch.save(dict(agent=agent.state_dict(), buffer=buffer.state_dict(), epoch=epoch,
                            total_steps=total_steps, time=elapsed, np_rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state()), tmp)
            tmp.replace(ckpt_path)

        msg = f'epoch {epoch:5d} | steps {total_steps:8d} | {row["steps_per_s"]:6.1f} steps/s'
        if n_updates:
            msg += f' | disc_acc {row["discriminator_acc"]:.3f}'
        if 'best_skill_return' in row:
            msg += f' | best skill {best} return {mean_ret[best]:.1f} | logq {row["eval_logq"]:.2f}'
        print(msg, flush=True)


if __name__ == '__main__':
    main()
