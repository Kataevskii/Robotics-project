"""Statistics and figure of the policy-initialization experiment (output of finetune_sweep.py).

    python finetune_stats.py results/pendulum/finetune --out figures/pendulum-finetune

Unit of analysis: in the diayn and actor arms, a pretrained DIAYN run. Its finetuning
repetitions share the pretrained weights, so they are averaged first and are not independent
samples. In the random arm, a run.

Fixed before the runs (not tuned on the results):
- Return: deterministic evaluation (eval_det_mean, 10 fixed initial states), as the original
  SAC code.
- Primary test: DIAYN (actor + critic) vs random init on `auc`, the mean evaluation return over
  finetuning epochs 0..N (area under the curve, normalised). One-sided Welch t-test, H1 DIAYN >
  random, alpha 0.05.
- Secondary tests, Holm-corrected together: every other metric x comparison.
- `auc_cost_adjusted` counts the reward-labelled steps spent choosing the skill: the DIAYN curves
  are shifted right by `selection_steps`, and every arm is averaged over the same window of
  reward-labelled steps.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from diayn.plotting import MUTED, SERIES, plt

ARMS = ['diayn', 'actor', 'random']
LABELS = {'diayn': 'DIAYN init, actor + critic', 'actor': 'DIAYN init, actor only',
          'random': 'Random init'}
COLORS = {'diayn': SERIES[0], 'random': SERIES[1], 'actor': SERIES[2]}
COMPARISONS = [('diayn', 'random'), ('actor', 'random'), ('diayn', 'actor')]
PRIMARY = ('diayn', 'random', 'auc')
LOWER_IS_BETTER = {'steps_to_threshold'}
N_BOOT = 10_000


def load(root):
    runs = []
    for d in sorted(Path(root).glob('*/*/')):
        if not (d / 'progress.csv').exists():
            continue
        cfg = json.loads((d / 'config.json').read_text())
        df = pd.read_csv(d / 'progress.csv')
        if df.epoch.max() < cfg['n_epochs']:
            print(f'skipping unfinished run {d}')
            continue
        unit = f'pre{cfg["pretrain_seed"]}' if cfg['init'] != 'random' else d.name
        runs.append(dict(init=cfg['init'], run=d.name, unit=unit, cfg=cfg,
                         curve=df.set_index('epoch')['eval_det_mean']))
    if not runs:
        raise SystemExit(f'no finished runs under {root}')
    return runs


def run_metrics(curve, cfg, threshold, window):
    E, N = cfg['epoch_length'], cfg['n_epochs']
    steps = curve.index.to_numpy() * E
    y = curve.to_numpy()
    reached = np.flatnonzero(y >= threshold)
    # Reward-labelled steps: finetuning steps plus those spent choosing the skill.
    labelled = steps + cfg['selection_steps']
    grid = np.arange(window[0], window[1] + 1, E)
    return dict(
        auc=y.mean(),
        auc_first_10k=y[steps <= 10_000].mean(),
        jumpstart=y[0],
        return_5k=np.interp(5_000, steps, y),
        return_10k=np.interp(10_000, steps, y),
        final=y[-5:].mean(),
        # Censored at N+1 epochs when never reached (rank tests handle it, means understate it).
        steps_to_threshold=steps[reached[0]] if reached.size else (N + 1) * E,
        reached_threshold=float(reached.size > 0),
        auc_cost_adjusted=np.interp(grid, labelled, y).mean(),
    )


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)) if sp > 0 else np.nan


def compare(a, b, lower_is_better, rng):
    alt = 'less' if lower_is_better else 'greater'
    boot = (rng.choice(a, (N_BOOT, len(a))).mean(1) - rng.choice(b, (N_BOOT, len(b))).mean(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    same = np.ptp(np.r_[a, b]) == 0
    return dict(mean_a=a.mean(), mean_b=b.mean(), diff=a.mean() - b.mean(), ci95_low=lo,
                ci95_high=hi, hedges_g=hedges_g(a, b),
                p_welch=np.nan if same else stats.ttest_ind(a, b, equal_var=False,
                                                             alternative=alt).pvalue,
                p_mannwhitney=np.nan if same else stats.mannwhitneyu(a, b,
                                                                     alternative=alt).pvalue)


def holm(p):
    p = np.asarray(p, float)
    out, ok = np.full_like(p, np.nan), ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.maximum.accumulate((len(q) - np.arange(len(q))) * q[order]).clip(max=1)
    res = np.empty_like(q)
    res[order] = adj
    out[ok] = res
    return out


def unit_curves(runs, init, x_shift=False):
    """Mean curve of each unit of an arm (index: env steps, or reward-labelled steps)."""
    cols = {}
    for r in (r for r in runs if r['init'] == init):
        c = r['curve'].copy()
        c.index = c.index * r['cfg']['epoch_length'] + (r['cfg']['selection_steps'] if x_shift else 0)
        cols.setdefault(r['unit'], []).append(c)
    return pd.DataFrame({u: pd.concat(cs, axis=1).mean(axis=1) for u, cs in cols.items()})


def boot_band(df, rng):
    m = df.to_numpy()
    idx = rng.integers(0, m.shape[1], (2000, m.shape[1]))
    means = m[:, idx].mean(2)
    return m.mean(1), *np.percentile(means, [2.5, 97.5], axis=1)


def plot(runs, arms, threshold, out, rng):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, shift, xlabel in ((axes[0], False, 'Finetuning env steps'),
                              (axes[1], True, 'Reward-labelled env steps (incl. skill selection)')):
        for init in arms:
            df = unit_curves(runs, init, shift)
            x = df.index.to_numpy() / 1000
            for u in df:
                ax.plot(x, df[u], color=COLORS[init], lw=0.8, alpha=0.25)
            mean, lo, hi = boot_band(df, rng)
            ax.fill_between(x, lo, hi, color=COLORS[init], alpha=0.18, lw=0)
            # No end-of-line labels: the arms converge to the same return, so they would collide.
            ax.plot(x, mean, color=COLORS[init], lw=2, label=f'{LABELS[init]} (n={df.shape[1]})')
        ax.axhline(threshold, color=MUTED, lw=1, ls='--')
        ax.set_xlabel(f'{xlabel}, thousands')
    axes[0].set_ylabel('Evaluation return (deterministic)')
    axes[0].set_title('Learning curves, mean and 95% bootstrap CI over units')
    axes[1].set_title('Same, counting the rollouts that choose the skill')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(arms), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'policy_init.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', help='output of finetune_sweep.py, e.g. results/pendulum/finetune')
    p.add_argument('--out', required=True)
    p.add_argument('--threshold', type=float, default=-200.0,
                   help='return counted as solved (Pendulum: -200)')
    p.add_argument('--seed', type=int, default=0, help='bootstrap seed')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    runs = load(args.root)
    arms = [a for a in ARMS if any(r['init'] == a for r in runs)]
    cfgs = [r['cfg'] for r in runs]
    window = (max(c['selection_steps'] for c in cfgs),
              min(c['n_epochs'] * c['epoch_length'] for c in cfgs))

    per_run = pd.DataFrame([dict(init=r['init'], unit=r['unit'], run=r['run'],
                                 skill=r['cfg']['skill'],
                                 skill_pretrain_return=r['cfg']['skill_pretrain_return'],
                                 **run_metrics(r['curve'], r['cfg'], args.threshold, window))
                            for r in runs])
    per_run.to_csv(out / 'per_run.csv', index=False)
    metrics = [c for c in per_run.columns
               if c not in ('init', 'unit', 'run', 'skill', 'skill_pretrain_return')]
    per_unit = per_run.groupby(['init', 'unit'], as_index=False)[
        ['skill_pretrain_return', *metrics]].mean()
    per_unit.to_csv(out / 'per_unit.csv', index=False)

    summary = per_unit.groupby('init')[metrics].agg(['mean', 'std', 'count']).T
    summary.to_csv(out / 'summary.csv')

    tests = []
    for a, b in COMPARISONS:
        if a not in arms or b not in arms:
            continue
        for m in metrics:
            if m == 'reached_threshold':
                continue
            xa = per_unit.loc[per_unit.init == a, m].to_numpy()
            xb = per_unit.loc[per_unit.init == b, m].to_numpy()
            tests.append(dict(a=a, b=b, metric=m, n_a=len(xa), n_b=len(xb),
                              primary=(a, b, m) == PRIMARY,
                              **compare(xa, xb, m in LOWER_IS_BETTER, rng)))
    tests = pd.DataFrame(tests)
    sec = ~tests.primary
    tests['p_welch_holm'] = np.nan
    tests.loc[sec, 'p_welch_holm'] = holm(tests.loc[sec, 'p_welch'])
    tests.loc[~sec, 'p_welch_holm'] = tests.loc[~sec, 'p_welch']
    tests.to_csv(out / 'tests.csv', index=False)

    rows = []
    for init in arms:
        df = unit_curves(runs, init)
        mean, lo, hi = boot_band(df, rng)
        rows.append(pd.DataFrame(dict(init=init, steps=df.index, mean=mean, ci95_low=lo,
                                      ci95_high=hi, n_units=df.shape[1])))
    pd.concat(rows).to_csv(out / 'curves.csv', index=False)
    plot(runs, arms, args.threshold, out, rng)

    pd.set_option('display.width', 160, 'display.precision', 3)
    print('Per unit:\n', per_unit[['init', 'unit', 'skill_pretrain_return', 'jumpstart', 'auc',
                                   'final', 'steps_to_threshold']].to_string(index=False))
    print('\nTests (one-sided, A better than B):')
    print(tests[['a', 'b', 'metric', 'n_a', 'n_b', 'diff', 'ci95_low', 'ci95_high', 'hedges_g',
                 'p_welch', 'p_mannwhitney', 'p_welch_holm']].to_string(index=False))
    prim = tests[tests.primary]
    if len(prim):
        r = prim.iloc[0]
        print(f'\nPRIMARY (DIAYN actor+critic vs random, AUC): diff {r["diff"]:.1f} '
              f'[{r.ci95_low:.1f}, {r.ci95_high:.1f}], g = {r.hedges_g:.2f}, '
              f'Welch p = {r.p_welch:.4f}')
    print(f'\nWrote {out}/policy_init.png, tests.csv, summary.csv, per_unit.csv, per_run.csv, curves.csv')


if __name__ == '__main__':
    main()
