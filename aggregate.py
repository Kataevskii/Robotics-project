"""Statistics across seeds, for one or more configurations.

    python aggregate.py results/pendulum/default results/pendulum/alpha0.01 --out figures/pendulum

Each argument is a configuration folder containing seed_*/ runs.

Diversity metrics (K = number of skills, log q from the run's discriminator on fresh
stochastic rollouts of every skill, every 10 epochs):
  diversity      1 + E_z E_s[log q(z|s)] / log K. DIAYN's objective, a lower bound on
                 I(S;Z), divided by its maximum log K: 0 = chance, 1 = fully distinguishable
  frac_distinct  share of skills whose mean log q(z|s) is above chance, log(1/K)
  epochs_to_90   first evaluation epoch where diversity reaches 90% of its final value

Writes to --out:
  diversity.png, returns.png, skill_returns.png   figures
  curves.csv              per configuration and epoch: mean and 95% t-interval across seeds
  summary_per_seed.csv    final values of every run
  summary.csv             mean, std and 95% t-interval across seeds of the final values
  eval_all_runs.csv       every evaluation row of every run (config, seed, epoch, metrics)
  skills_all_runs.csv     every per-skill evaluation row of every run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from diayn.plotting import MUTED, SERIES, plt

DIVERSITY = [('diversity', 'Diversity: 1 + E[log q(z|s)] / log K'),
             ('frac_distinct', 'Fraction of skills above chance'),
             ('discriminator_acc', 'Discriminator accuracy (training batches)')]
RETURNS = [('best_skill_return', 'Env return of the best skill'),
           ('mean_skill_return', 'Mean env return over skills')]
FINAL = ['diversity', 'frac_distinct', 'epochs_to_90', 'discriminator_acc',
         'best_skill_return', 'mean_skill_return']


def t_interval(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan, np.nan
    m, h = x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def load_run(run):
    """Evaluation rows of one run with the derived diversity metrics, and its per-skill rows."""
    K = json.loads((run / 'config.json').read_text())['num_skills']
    prog = pd.read_csv(run / 'progress.csv')
    skills = pd.read_csv(run / 'skill_eval.csv')
    ev = prog.dropna(subset=['eval_logq'])[['epoch', 'total_steps', 'eval_logq', 'best_skill',
                                             'best_skill_return', 'mean_skill_return',
                                             'discriminator_acc']].copy()
    ev['diversity'] = 1 + ev.eval_logq / np.log(K)
    frac = skills.groupby('epoch').logq_mean.apply(lambda q: (q > -np.log(K)).mean())
    ev['frac_distinct'] = ev.epoch.map(frac)
    for df in (ev, skills):
        df.insert(0, 'seed', int(run.name.split('_')[1]))
        df.insert(1, 'num_skills', K)
    return ev, skills


def load_config(folder):
    runs = [load_run(r) for r in sorted(Path(folder).glob('seed_*')) if (r / 'skill_eval.csv').exists()]
    if not runs:
        raise SystemExit(f'No runs with skill_eval.csv in {folder}')
    ev, skills = (pd.concat(x, ignore_index=True) for x in zip(*runs))
    # Keep the evaluation epochs every seed has reached, so curves average the same seeds.
    common = ev.groupby('epoch').seed.nunique()
    ev = ev[ev.epoch.isin(common[common == ev.seed.nunique()].index)]
    return ev, skills[skills.epoch <= ev.epoch.max()]


def plot_curves(data, metrics, path, curve_rows):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2))
    for color, (tag, (ev, _)) in zip(SERIES, data.items()):
        n = ev.seed.nunique()
        for ax, (col, title) in zip(axes, metrics):
            g = ev.groupby('epoch')
            ci = g[col].apply(t_interval)
            steps = g.total_steps.first() / 1e6
            m, lo, hi = (np.array([c[i] for c in ci]) for i in range(3))
            ax.plot(steps, m, color=color, label=f'{tag} (n={n})')
            ax.fill_between(steps, lo, hi, color=color, alpha=0.18, lw=0)
            ax.set_title(title, loc='left')
            ax.set_xlabel('environment steps (millions)')
            if col in ('diversity', 'frac_distinct'):
                ax.axhline(0 if col == 'diversity' else 1, color=MUTED, lw=1, ls='--')
            curve_rows += [dict(config=tag, metric=col, epoch=e, total_steps=s, n_seeds=n,
                                mean=a, ci95_low=b, ci95_high=c)
                           for e, s, a, b, c in zip(ci.index, g.total_steps.first(), m, lo, hi)]
    if len(data) > 1:
        axes[0].legend(loc='best')
    fig.suptitle('Mean and 95% interval across seeds', x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('configs', nargs='+')
    p.add_argument('--out', default='figures')
    args = p.parse_args()
    if len(args.configs) > len(SERIES):
        raise SystemExit(f'At most {len(SERIES)} configurations per figure')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {Path(c).name: load_config(c) for c in args.configs}

    curve_rows = []
    plot_curves(data, DIVERSITY, out / 'diversity.png', curve_rows)
    plot_curves(data, RETURNS, out / 'returns.png', curve_rows)
    pd.DataFrame(curve_rows).to_csv(out / 'curves.csv', index=False)
    pd.concat([ev.assign(config=t) for t, (ev, _) in data.items()]).to_csv(out / 'eval_all_runs.csv', index=False)
    pd.concat([sk.assign(config=t) for t, (_, sk) in data.items()]).to_csv(out / 'skills_all_runs.csv', index=False)

    per_seed = []
    for tag, (ev, _) in data.items():
        for seed, g in ev.groupby('seed'):
            final = g.iloc[-1]
            reached = g[g.diversity >= 0.9 * final.diversity].epoch
            per_seed.append(dict(config=tag, seed=seed, epoch=final.epoch,
                                 epochs_to_90=reached.iloc[0] if final.diversity > 0 else np.nan,
                                 **{k: final[k] for k in FINAL if k != 'epochs_to_90'}))
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(out / 'summary_per_seed.csv', index=False)
    rows = []
    for tag, g in per_seed.groupby('config', sort=False):
        for metric in FINAL:
            m, lo, hi = t_interval(g[metric])
            rows.append(dict(config=tag, metric=metric, n_seeds=len(g), mean=m,
                             std=g[metric].std(ddof=1), ci95_low=lo, ci95_high=hi))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'summary.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # Env return of every skill at the last common evaluation: one dot column per seed.
    panel_w = max(4.0, 1.5 + 0.6 * per_seed.groupby('config').size().max())
    fig, axes = plt.subplots(1, len(data), figsize=(panel_w * len(data), 3.2), squeeze=False,
                             sharey=True)
    for ax, color, (tag, (ev, skills)) in zip(axes[0], SERIES, data.items()):
        last = ev.epoch.max()
        seeds = sorted(skills.seed.unique())
        for i, seed in enumerate(seeds):
            r = skills[(skills.seed == seed) & (skills.epoch == last)].return_mean.to_numpy()
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(r))
            ax.scatter(i + jitter, r, s=14, color=color, alpha=0.7, lw=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(r)] * 2, color=MUTED, lw=1.5)
        ax.set_xticks(range(len(seeds)), [f's{s}' for s in seeds])
        ax.set_title(f'{tag} (epoch {last})', loc='left')
        ax.set_xlabel('seed')
    axes[0][0].set_ylabel('env return (not used for training)')
    fig.suptitle('Env return of every skill at the last evaluation\n(line: median per seed)',
                 x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'skill_returns.png', dpi=150)
    print(f'Figures and tables written to {out}')


if __name__ == '__main__':
    main()
