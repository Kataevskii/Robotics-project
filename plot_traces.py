"""Plot each skill's trajectory in two chosen coordinates (as scripts/plot_traces.py).

    python plot_traces.py results/point/default/seed_1/snapshots/itr_50.pt
    python plot_traces.py <snapshot> --x info:x_position --y info:y_position      # Ant
    python plot_traces.py <snapshot> --x t --y info:x_position                     # HalfCheetah, Hopper

A coordinate is obs:<i> (observation dim), info:<key> (Gymnasium info, e.g. the x position
the MuJoCo envs drop from the observation) or t (time step).
Up to 8 skills share one panel; more are drawn as one small panel per skill.
"""
import argparse
from pathlib import Path

import numpy as np

from diayn.envs import make_env
from diayn.evaluation import load_snapshot, rollout_skills
from diayn.plotting import SERIES, plt


def series(trace, spec):
    if spec == 't':
        return np.arange(len(trace['info'])) + 1
    kind, key = spec.split(':', 1)
    if kind == 'obs':
        return np.array(trace['obs'][1:])[:, int(key)]
    return np.array([info[key] for info in trace['info']])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('snapshot')
    p.add_argument('--x', default='obs:0')
    p.add_argument('--y', default='obs:1')
    p.add_argument('--n_paths', type=int, default=1, help='rollouts per skill')
    p.add_argument('--max_path_length', type=int, default=None, help='default: training value')
    p.add_argument('--stochastic', action='store_true', help='sample actions (default: deterministic)')
    args = p.parse_args()

    agent, cfg = load_snapshot(args.snapshot)
    S, L = cfg['num_skills'], args.max_path_length or cfg['max_path_length']
    envs = [make_env(cfg['env_id'], L) for _ in range(S)]
    for i, e in enumerate(envs):
        e.reset(seed=i)
    paths = [rollout_skills(agent, envs, L, not args.stochastic, record=True)[2]
             for _ in range(args.n_paths)]

    if S <= len(SERIES):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        axes = [ax] * S
    else:
        cols = 10
        rows = -(-S // cols)
        fig, grid = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows), sharex=True, sharey=True)
        axes = grid.ravel()
        for ax in axes[S:]:
            ax.set_visible(False)
    for z in range(S):
        color = SERIES[z] if S <= len(SERIES) else SERIES[0]
        for traces in paths:
            axes[z].plot(series(traces[z], args.x), series(traces[z], args.y), color=color, lw=1.5,
                         label=f'skill {z}' if traces is paths[0] else None)
        if S > len(SERIES):
            axes[z].set_title(f'skill {z}', fontsize=8, loc='left')
    if S <= len(SERIES):
        axes[0].legend(loc='best')
        axes[0].set_xlabel(args.x)
        axes[0].set_ylabel(args.y)
    else:
        fig.supxlabel(args.x)
        fig.supylabel(args.y)
    snap = Path(args.snapshot)
    fig.suptitle(f'{cfg["env"]} seed {cfg["seed"]}, {snap.stem}', x=0.01, ha='left')
    fig.tight_layout()
    out = snap.with_name(f'{snap.stem}_{args.x}_{args.y}_trace.png'.replace(':', '-'))
    fig.savefig(out, dpi=150)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
