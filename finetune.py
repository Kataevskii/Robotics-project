"""Finetune a pretrained DIAYN skill on the task reward (paper section 4.2.1, Figure 5), or
train the same networks from scratch.

    python finetune.py --pretrain_dir results/pendulum/default/seed_1 --init diayn --seed 1 \
        --log_dir results/pendulum/finetune/diayn/pre1_seed1

--init diayn   policy, Q, V and target V from the pretrained checkpoint (the paper's method)
--init actor   policy only, critics freshly initialised (the paper's critic claim)
--init random  every network freshly initialised (the paper's "random initialization")

The skill is fixed for the whole run. It is the best skill of the pretraining evaluation at the
checkpoint epoch (skill_eval.csv: 10 stochastic rollouts per skill), unless --skill is given.
The random baseline keeps the same skill input, so the networks are identical and the arms
differ only in their initial weights. SAC trains on the env reward with the pretraining
hyperparameters (config.json of the pretrained run), and the discriminator is not used.
Optimiser and replay buffer start empty in every arm.

Epoch 0 is an evaluation before any update. Every epoch logs deterministic and stochastic
returns over --eval_episodes episodes, from initial states that are the same for every run
and every evaluation.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diayn.agent import DIAYN, ReplayBuffer, STAT_NAMES
from diayn.envs import make_env
from diayn.evaluation import rollout_skills
from train import CsvLog

EVAL_SEED = 123_456
SELECT_SEED = 7_000_000  # skill-selection rollouts, disjoint from the evaluation states
TRANSFERRED = {'diayn': ('policy', 'qf', 'vf', 'vf_target'), 'actor': ('policy',), 'random': ()}
TRAIN_STATS = ['qf', 'vf', 'bellman_error', 'log_pi', 'kl_surrogate_loss', 'vf_loss']
COLS = ['epoch', 'total_steps', 'time', 'steps_per_s', 'episodes', 'train_return', *TRAIN_STATS,
        'eval_det_mean', 'eval_det_std', 'eval_sto_mean', 'eval_sto_std']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain_dir', required=True,
                   help='pretrained run (config.json, checkpoint.pt, skill_eval.csv). '
                        'The random arm only takes its config from it')
    p.add_argument('--init', required=True, choices=list(TRANSFERRED))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--log_dir', required=True)
    p.add_argument('--skill', type=int, default=None, help='default: best pretrained skill')
    p.add_argument('--select_rollouts', type=int, default=None,
                   help='choose the skill from this many new stochastic rollouts per skill '
                        "instead of the pretraining evaluation's 10, and count only their steps")
    p.add_argument('--n_epochs', type=int, default=50)
    p.add_argument('--eval_episodes', type=int, default=10)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    return p.parse_args()


def best_skill(pre_dir, epoch):
    """Best skill of the last pretraining evaluation at or before `epoch`."""
    df = pd.read_csv(pre_dir / 'skill_eval.csv')
    df = df[df.epoch == df.epoch[df.epoch <= epoch].max()]
    row = df.loc[df.return_mean.idxmax()]
    return int(row.skill), float(row.return_mean)


def select_skill(ck, pcfg, obs_dim, act_dim, device, n_rollouts, seed):
    """Best skill over n_rollouts new stochastic rollouts per skill of the pretrained policy.

    Initial states come from seeds disjoint from the evaluation's. Returns the skill, its mean
    return and the env steps used (reward-labelled steps that the cost-adjusted AUC counts).
    """
    pre_agent = DIAYN(obs_dim, act_dim, pcfg, device)
    pre_agent.load_state_dict({k: ck['agent'][k] for k in ('policy', 'qf')})
    L, S = pcfg['max_path_length'], pcfg['num_skills']
    envs = [make_env(pcfg['env_id'], L) for _ in range(S)]
    for i, e in enumerate(envs):
        e.reset(seed=SELECT_SEED + 1000 * seed + i)
    rets, steps = [], 0
    for _ in range(n_rollouts):
        ret, _, traces = rollout_skills(pre_agent, envs, L, record=True)
        rets.append(ret)
        steps += sum(len(t['info']) for t in traces)
    mean = np.mean(rets, 0)
    best = int(np.argmax(mean))
    return best, float(mean[best]), steps


@torch.no_grad()
def evaluate(agent, envs, z, max_path_length, deterministic):
    """One episode per env with skill z, all envs stepped in lockstep. Fixed initial states."""
    obs = np.stack([e.reset(seed=EVAL_SEED + i)[0] for i, e in enumerate(envs)])
    alive, ret = np.ones(len(envs), bool), np.zeros(len(envs))
    for _ in range(max_path_length):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        for i, a in zip(idx, agent.act(obs[idx], np.full(idx.size, z), deterministic)):
            o, r, term, trunc, _ = envs[i].step(a)
            ret[i] += r
            obs[i] = o
            if term or trunc:
                alive[i] = False
    return ret


def main():
    args = parse_args()
    pre, log_dir = Path(args.pretrain_dir), Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    prog_path = log_dir / 'progress.csv'
    if prog_path.exists():
        done = pd.read_csv(prog_path)
        if len(done) and done.epoch.max() >= args.n_epochs:
            print('Already finished')
            return
        prog_path.unlink()  # runs are short: an unfinished run restarts from scratch

    pcfg = json.loads((pre / 'config.json').read_text())
    ck = torch.load(pre / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if args.init == 'random':
        skill, skill_ret, selection_steps = 0, None, 0
    else:
        skill, skill_ret = best_skill(pre, ck['epoch'])
        # Reward-labelled env steps spent choosing the skill (upper bound: episodes may end early).
        selection_steps = pcfg['num_skills'] * pcfg['best_skill_n_rollouts'] * pcfg['max_path_length']
    if args.skill is not None:
        skill, selection_steps = args.skill, 0

    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    L = pcfg['max_path_length']
    env = make_env(pcfg['env_id'], L)
    eval_envs = [make_env(pcfg['env_id'], L) for _ in range(args.eval_episodes)]
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    arm, skill_pretrain_eval = args.init, skill
    if args.select_rollouts and args.init != 'random' and args.skill is None:
        skill, skill_ret, selection_steps = select_skill(ck, pcfg, obs_dim, act_dim, device,
                                                         args.select_rollouts, args.seed)
        arm = f'{args.init}_sel{args.select_rollouts}'

    cfg = dict(pcfg)  # same networks and SAC hyperparameters as pretraining
    cfg.update(init=args.init, arm=arm, seed=args.seed, pretrain_dir=str(pre),
               pretrain_seed=pcfg['seed'], pretrain_epoch=ck['epoch'], skill=skill,
               skill_pretrain_return=skill_ret, skill_pretrain_eval=skill_pretrain_eval,
               select_rollouts=args.select_rollouts, selection_steps=selection_steps,
               transferred=list(TRANSFERRED[args.init]), n_epochs=args.n_epochs,
               eval_episodes=args.eval_episodes, eval_seed=EVAL_SEED)
    (log_dir / 'config.json').write_text(json.dumps(cfg, indent=2))

    agent = DIAYN(obs_dim, act_dim, cfg, device)
    for k in TRANSFERRED[args.init]:
        agent.nets()[k].load_state_dict(ck['agent'][k])
    pool = min(cfg['max_pool_size'], args.n_epochs * cfg['epoch_length'])
    buffer = ReplayBuffer(obs_dim, act_dim, pool, reward=True)

    progress = CsvLog(prog_path, COLS)
    obs, _ = env.reset(seed=args.seed)
    zs = np.array([skill])
    ep_return, finished, total_steps, elapsed = 0.0, [], 0, 0.0
    print(f'init {args.init} | pretrained {pre} | skill {skill} | transferred {TRANSFERRED[args.init]}')

    for epoch in range(args.n_epochs + 1):
        t0 = time.time()
        row = dict(epoch=epoch)
        if epoch > 0:
            stats, n_updates = torch.zeros(len(STAT_NAMES), device=device), 0
            for _ in range(cfg['epoch_length']):
                a = agent.act(obs[None], zs)[0]
                next_obs, r, term, trunc, _ = env.step(a)
                buffer.add(obs=obs, next_obs=next_obs, act=a, done=float(term), z=skill, reward=r)
                ep_return += r
                total_steps += 1
                if term or trunc:  # time-limit ends are not terminal
                    finished.append(ep_return)
                    ep_return = 0.0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                if buffer.size >= L:  # min_pool_size = max_path_length, as pretraining
                    for _ in range(cfg['n_train_repeat']):
                        stats += agent.update(buffer.sample(cfg['batch_size'], rng, device))
                        n_updates += 1
            row.update(steps_per_s=round(cfg['epoch_length'] / (time.time() - t0), 1),
                       episodes=len(finished), train_return=np.mean(finished) if finished else '')
            finished = []
            if n_updates:
                row.update({k: v for k, v in zip(STAT_NAMES, (stats / n_updates).tolist())
                            if k in TRAIN_STATS})

        for name, det in (('det', True), ('sto', False)):
            ret = evaluate(agent, eval_envs, skill, L, det)
            row[f'eval_{name}_mean'], row[f'eval_{name}_std'] = ret.mean(), ret.std()
        elapsed += time.time() - t0
        row.update(total_steps=total_steps, time=round(elapsed, 1))
        progress.write(row)
        print(f'epoch {epoch:4d} | steps {total_steps:7d} | eval det {row["eval_det_mean"]:8.1f} '
              f'| eval sto {row["eval_sto_mean"]:8.1f}', flush=True)

    sd = agent.state_dict()
    torch.save({k: sd[k] for k in ('policy', 'qf', 'vf', 'vf_target')}, log_dir / 'final.pt')


if __name__ == '__main__':
    main()
