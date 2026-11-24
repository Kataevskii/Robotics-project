"""Poster figures for the policy-initialization experiment (docs/poster-policy-init.md).

    python poster_figures.py --out docs/img

Reads the finetuning runs (results/pendulum/finetune, results/pendulum/finetune_sel1), their
pretrained runs (results/pendulum/default) and the test table of finetune_stats.py
(figures/pendulum-finetune-sel1/tests.csv). The critic-calibration and skill figures roll out
the pretrained checkpoints, which takes seconds and trains nothing.

Figures (one per poster insight):
  time_to_solve.png          fraction of runs that reached -200, against finetuning steps
  effects.png                difference to random init, with 95% CI, for every metric
  start_vs_solve.png         return before finetuning against steps to solve, one dot per run
  critic_values.png          evaluation return and critic estimate V during finetuning
  critic_calibration.png     pretrained critic V(s) against Monte-Carlo soft returns
  selection_cost.png         advantage over random init against the cost of choosing the skill
  pretrain_seed_variance.png AUC per pretrained seed, and what pseudo-replication would claim
  pretrained_skills.png      what the pretrained Pendulum skills do, and which one is chosen
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from diayn.agent import DIAYN, GMMPolicy
from diayn.envs import make_env
from diayn.plotting import INK_2, MUTED, SERIES, plt
from finetune import best_skill
from finetune_stats import COLORS, LABELS, boot_band, load, run_metrics, unit_curves

ROOTS = ['results/pendulum/finetune', 'results/pendulum/finetune_sel1']
PRETRAIN = Path('results/pendulum/default')
TESTS = Path('figures/pendulum-finetune-sel1/tests.csv')
THRESHOLD = -200.0
END = 50_000
ARMS = ['diayn', 'diayn_sel1', 'actor', 'random']
MARKERS = {'diayn': 'o', 'diayn_sel1': 'D', 'actor': '^', 'random': 's'}
AUTHORS_SELECTION = 2_000  # mujoco_all_diayn_finetune.py: 2 rollouts x 5 skills x 200 steps
CHOSEN, OTHER = SERIES[0], MUTED


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out / name}')


# 1. Faster learning --------------------------------------------------------------------------
def fig_time_to_solve(per_run, out):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for arm in ARMS:
        s = np.sort(per_run.loc[per_run.init == arm, 'steps_to_threshold'].to_numpy()) / 1000
        n = len(s)
        ax.step(np.r_[0, s, END / 1000], np.r_[0, np.arange(1, n + 1) / n, 1], where='post',
                color=COLORS[arm], lw=2, label=f'{LABELS[arm]}: median {np.median(s):g}k')
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Finetuning env steps, thousands')
    ax.set_ylabel('Fraction of runs solved (return ≥ −200)')
    ax.set_title('Time to solve Pendulum, one step per run (10 runs per arm)')
    ax.legend(loc='lower right', fontsize=8)
    save(fig, out, 'time_to_solve.png')


# 2. Effect sizes -----------------------------------------------------------------------------
EFFECT_METRICS = [('jumpstart', 'Return before finetuning'),
                  ('return_5k', 'Return after 5k steps'),
                  ('return_10k', 'Return after 10k steps'),
                  ('auc', 'Mean return over 50k steps (AUC)'),
                  ('final', 'Final return (last 5k steps)'),
                  ('auc_cost_adjusted', 'AUC counting skill-selection steps')]
EFFECT_ARMS = ['diayn', 'diayn_sel1', 'actor']


def fig_effects(tests, out):
    t = tests[tests.b == 'random']
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    y, yticks, ylabels = 0, [], []
    for metric, label in EFFECT_METRICS:
        yticks.append(y - 1)
        ylabels.append(label)
        for k, arm in enumerate(EFFECT_ARMS):
            r = t[(t.a == arm) & (t.metric == metric)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = y - k * 0.8
            ax.plot([r.ci95_low, r.ci95_high], [yy, yy], color=COLORS[arm], lw=2)
            ax.plot(r['diff'], yy, MARKERS[arm], color=COLORS[arm], ms=7,
                    mec='white', mew=1.5, label=LABELS[arm] if metric == 'jumpstart' else None)
            ax.annotate(f'{r["diff"]:+.0f}', (r.ci95_high, yy), xytext=(5, 0),
                        textcoords='offset points', va='center', fontsize=7, color=INK_2)
        y -= 3.4
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(yticks, ylabels)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Difference to random init in evaluation return, with 95% bootstrap CI')
    ax.set_title('Large early gains, no difference at the end')
    ax.legend(loc='lower right', fontsize=8)
    save(fig, out, 'effects.png')


# 3. Not just a good start --------------------------------------------------------------------
def fig_start_vs_solve(per_run, out):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    rnd = per_run.loc[per_run.init == 'random', 'steps_to_threshold'] / 1000
    ax.axhspan(rnd.quantile(0.25), rnd.quantile(0.75), color=COLORS['random'], alpha=0.12, lw=0)
    ax.axhline(rnd.median(), color=COLORS['random'], lw=1, ls='--')
    for arm in ARMS:
        d = per_run[per_run.init == arm]
        ax.scatter(d.jumpstart, d.steps_to_threshold / 1000, s=36, marker=MARKERS[arm],
                   color=COLORS[arm], edgecolor='white', linewidth=1, label=LABELS[arm], zorder=3)
    ax.annotate('random init: median and interquartile range', (-1000, rnd.quantile(0.75)),
                xytext=(0, 3), textcoords='offset points', fontsize=7, color=INK_2)
    censored = (END + 1000) / 1000
    if (per_run.steps_to_threshold / 1000 >= censored).any():
        ax.annotate('not reached within 50k steps', (-1180, censored), xytext=(0, 0),
                    textcoords='offset points', fontsize=7, color=INK_2, va='center')
    ax.set_xlabel('Evaluation return before finetuning')
    ax.set_ylabel('Steps to reach −200, thousands')
    ax.set_title('Bad starts still solve fast after DIAYN pretraining')
    ax.legend(loc='center right', fontsize=7)
    save(fig, out, 'start_vs_solve.png')


# 4. The transferred critic -------------------------------------------------------------------
def unit_frame(arm, column):
    cols = {}
    for f in sorted(Path(ROOTS[0], arm).glob('*/progress.csv')):
        cfg = json.loads((f.parent / 'config.json').read_text())
        unit = f'pre{cfg["pretrain_seed"]}' if arm != 'random' else f.parent.name
        c = pd.read_csv(f).set_index('total_steps')[column]
        cols.setdefault(unit, []).append(c)
    return pd.DataFrame({u: pd.concat(cs, axis=1).mean(axis=1) for u, cs in cols.items()}).dropna()


def fig_critic_values(out, rng):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True)
    for arm in ['diayn', 'actor', 'random']:
        for ax, column in zip(axes, ['eval_det_mean', 'vf']):
            df = unit_frame(arm, column)
            x = df.index.to_numpy() / 1000
            mean, lo, hi = boot_band(df, rng)
            ax.fill_between(x, lo, hi, color=COLORS[arm], alpha=0.18, lw=0)
            ax.plot(x, mean, color=COLORS[arm], lw=2, label=LABELS[arm])
    axes[0].axhline(THRESHOLD, color=MUTED, lw=1, ls='--')
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel('Evaluation return')
    axes[1].set_ylabel('Critic estimate V\n(mean over training batches)')
    axes[1].set_xlabel('Finetuning env steps, thousands')
    axes[0].set_title('The transferred critic starts at +130; fresh critics dive to −300')
    axes[0].legend(loc='lower right', fontsize=8)
    save(fig, out, 'critic_values.png')


@torch.no_grad()
def soft_returns(agent, cfg, z, n_rollouts, extra=500):
    """States of n stochastic rollouts of skill z, with the critic's V and the Monte-Carlo soft
    returns of the pseudo-reward and of the task reward (entropy term alpha * -log pi included,
    as the SAC v1 value target). Rollouts run `extra` steps past the time limit so the returns of
    the first max_path_length states are not truncated (gamma^500 < 0.01)."""
    L, gamma, alpha = cfg['max_path_length'], cfg['discount'], cfg['scale_entropy']
    log_pz = np.log(1.0 / cfg['num_skills']) if cfg['add_p_z'] else 0.0
    env = make_env(cfg['env_id'], L + extra)
    out = []
    for k in range(n_rollouts):
        obs, _ = env.reset(seed=9_000_000 + k)
        O, R, RP, H = [], [], [], []
        for _ in range(L + extra):
            o = agent.aug(agent._tensor(obs[None]), agent._tensor([z], torch.long))
            log_w, mu, log_sig = agent.policy(o)
            x = GMMPolicy.sample(log_w, mu, log_sig)
            log_pi, _ = agent.policy.log_prob(log_w, mu, log_sig, x)
            corr = torch.log(1 - torch.tanh(x) ** 2 + 1e-6).sum(1)
            O.append(obs)
            H.append(float(log_pi - corr))
            RP.append(float(agent.log_q(obs[None], [z])[0]) - log_pz)
            obs, r, term, trunc, _ = env.step(torch.tanh(x)[0].numpy())
            R.append(r)
        g_task, g_pseudo, gt, gp = [], [], 0.0, 0.0
        for r, rp, h in zip(R[::-1], RP[::-1], H[::-1]):
            gt = r - alpha * h + gamma * gt
            gp = rp - alpha * h + gamma * gp
            g_task.append(gt)
            g_pseudo.append(gp)
        O = np.array(O[:L], np.float32)
        v = agent.vf(agent.aug(agent._tensor(O), agent._tensor([z] * L, torch.long))).squeeze(1)
        out.append(pd.DataFrame(dict(v=v.numpy(), pseudo=g_pseudo[::-1][:L], task=g_task[::-1][:L])))
    return pd.concat(out)


def pretrained(seed):
    run = PRETRAIN / f'seed_{seed}'
    cfg = json.loads((run / 'config.json').read_text())
    ck = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
    env = make_env(cfg['env_id'], cfg['max_path_length'])
    agent = DIAYN(env.observation_space.shape[0], env.action_space.shape[0], cfg, 'cpu')
    agent.load_state_dict({k: ck['agent'][k] for k in ('policy', 'qf', 'vf', 'vf_target', 'disc')})
    torch.manual_seed(seed)
    return agent, cfg, ck['epoch'], run


def fig_critic_calibration(out):
    seeds = range(1, 6)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2))
    rows = []
    for j, seed in enumerate(seeds):
        agent, cfg, epoch, run = pretrained(seed)
        z, _ = best_skill(run, epoch)
        d = soft_returns(agent, cfg, z, n_rollouts=10)
        for i, target in enumerate(['pseudo', 'task']):
            ax = axes[i, j]
            ax.scatter(d.v, d[target], s=3, color=CHOSEN, alpha=0.15, lw=0)
            rho = stats.spearmanr(d.v, d[target]).statistic
            rows.append(dict(seed=seed, skill=z, target=target, spearman=rho,
                             v_mean=d.v.mean(), return_mean=d[target].mean()))
            if target == 'pseudo':
                lim = [min(d.v.min(), d.pseudo.min()), max(d.v.max(), d.pseudo.max())]
                ax.plot(lim, lim, color=MUTED, lw=1, ls='--')
            ax.text(0.04, 0.94, f'Spearman ρ = {rho:.2f}', transform=ax.transAxes,
                    fontsize=8, color=INK_2, va='top')
            if i == 0:
                ax.set_title(f'Pretrained seed {seed} (skill {z})')
            if i == 1:
                ax.set_xlabel('Pretrained critic V(s)')
    axes[0, 0].set_ylabel('Soft return of the\nDIAYN pseudo-reward')
    axes[1, 0].set_ylabel('Soft return of the\ntask reward')
    fig.suptitle('The pretrained critic is accurate for the pseudo-reward (top, dashed: y = x) '
                 'but not for the task (bottom)', fontsize=10)
    pd.DataFrame(rows).to_csv(out / 'critic_calibration.csv', index=False)
    save(fig, out, 'critic_calibration.png')
    return pd.DataFrame(rows)


# 5. Selection cost ---------------------------------------------------------------------------
def adjusted_auc(df, shift, start):
    grid = np.arange(start, END + 1, 1000)
    return np.array([np.interp(grid, df.index.to_numpy() + shift, df[u].to_numpy()).mean()
                     for u in df])


def fig_selection_cost(runs, tests, out, rng):
    diayn, rnd = unit_curves(runs, 'diayn'), unit_curves(runs, 'random')
    costs = np.arange(0, 15_001, 250)
    mean, lo, hi = [], [], []
    for c in costs:
        a, b = adjusted_auc(diayn, c, c), adjusted_auc(rnd, 0, c)
        boot = (rng.choice(a, (4000, len(a))).mean(1) - rng.choice(b, (4000, len(b))).mean(1))
        mean.append(a.mean() - b.mean())
        lo_, hi_ = np.percentile(boot, [2.5, 97.5])
        lo.append(lo_)
        hi.append(hi_)
    mean = np.array(mean)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = costs / 1000
    ax.fill_between(x, lo, hi, color=COLORS['diayn'], alpha=0.18, lw=0)
    ax.plot(x, mean, color=COLORS['diayn'], lw=2,
            label='DIAYN learning curves, shifted by the cost (same skill assumed)')
    ax.axhline(0, color=MUTED, lw=1)
    for arm, cost in (('diayn_sel1', 1000), ('diayn', 10_000)):
        r = tests[(tests.a == arm) & (tests.b == 'random') & (tests.metric == 'auc_cost_adjusted')].iloc[0]
        ax.errorbar(cost / 1000, r['diff'], yerr=[[r['diff'] - r.ci95_low], [r.ci95_high - r['diff']]],
                    fmt=MARKERS[arm], color=COLORS[arm], ms=7, mec='white', mew=1.5, capsize=0,
                    lw=2, label=f'measured: {LABELS[arm].lower()}', zorder=3)
    cross = np.flatnonzero(np.diff(np.sign(mean)) < 0)
    if cross.size:
        i = cross[0]
        c0 = np.interp(0, [mean[i + 1], mean[i]], [x[i + 1], x[i]])
        ax.axvline(c0, color=INK_2, lw=1, ls=':')
        ax.annotate(f'break-even ≈ {c0:.1f}k steps', (c0, min(lo)), xytext=(4, 0),
                    textcoords='offset points', fontsize=8, color=INK_2, va='bottom')
    ax.axvline(AUTHORS_SELECTION / 1000, color=MUTED, lw=1, ls='--')
    ax.annotate("authors' finetuning script\n(2 rollouts per skill)", (AUTHORS_SELECTION / 1000, max(hi)),
                xytext=(4, 0), textcoords='offset points', fontsize=7, color=INK_2, va='top')
    ax.set_xlabel('Reward-labelled steps spent choosing the skill, thousands\n'
                  '(Pendulum: 1k steps = 1 rollout of each of the 5 skills)')
    ax.set_ylabel('AUC advantage over random init\n(counting selection steps)')
    ax.set_title('How much skill selection the speed-up can pay for')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=7)  # below: the plot is full
    save(fig, out, 'selection_cost.png')


# 6. Unit of replication ----------------------------------------------------------------------
def fig_pretrain_seed_variance(per_run, out):
    d = per_run[per_run.init == 'diayn'].copy()
    d['seed'] = d.unit.str[3:].astype(int)
    rnd = per_run.loc[per_run.init == 'random', 'auc'].to_numpy()
    # One-way random-effects ICC: share of the DIAYN arm's variance due to the pretrained seed.
    groups = [g.auc.to_numpy() for _, g in d.groupby('seed')]
    k, n = len(groups[0]), len(groups)
    grand = d.auc.mean()
    msb = k * sum((g.mean() - grand) ** 2 for g in groups) / (n - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    unit_means = d.groupby('seed').auc.mean().to_numpy()
    p_units = stats.ttest_ind(unit_means, rnd, equal_var=False, alternative='greater').pvalue
    p_naive = stats.ttest_ind(d.auc.to_numpy(), rnd, equal_var=False, alternative='greater').pvalue

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for s, g in d.groupby('seed'):
        ax.scatter([s] * len(g), g.auc, s=36, marker='o', color=COLORS['diayn'],
                   edgecolor='white', linewidth=1, zorder=3)
        ax.plot([s - 0.25, s + 0.25], [g.auc.mean()] * 2, color=COLORS['diayn'], lw=2)
    jitter = np.linspace(-0.25, 0.25, len(rnd))
    ax.scatter(7 + jitter, rnd, s=36, marker='s', color=COLORS['random'], edgecolor='white',
               linewidth=1, zorder=3)
    ax.plot([6.6, 7.4], [rnd.mean()] * 2, color=COLORS['random'], lw=2)
    ax.set_xticks([1, 2, 3, 4, 5, 7], ['seed 1', 'seed 2', 'seed 3', 'seed 4', 'seed 5', 'random\ninit'])
    ax.set_xlabel('DIAYN init (actor + critic), 2 finetuning runs per pretrained seed')
    ax.set_ylabel('Mean return over 50k steps (AUC)')
    ax.set_title(f'{icc:.0%} of the DIAYN arm\'s variance comes from the pretrained seed')
    ax.text(0.02, 0.04, f'Welch p, 5 pretrained seeds as units: {p_units:.4f}\n'
                        f'Welch p, 10 runs treated as independent: {p_naive:.6f}',
            transform=ax.transAxes, fontsize=7, color=INK_2)
    save(fig, out, 'pretrain_seed_variance.png')
    return icc, p_units, p_naive


# Context: the pretrained skills --------------------------------------------------------------
def fig_pretrained_skills(out, seed=2):
    agent, cfg, epoch, run = pretrained(seed)
    z_best, _ = best_skill(run, epoch)
    env = make_env(cfg['env_id'], cfg['max_path_length'])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    for z in range(cfg['num_skills']):
        obs, _ = env.reset(seed=1)  # same start for every skill
        th, thd = [], []
        for _ in range(cfg['max_path_length']):
            th.append(np.arctan2(obs[1], obs[0]))
            thd.append(obs[2])
            obs, *_ = env.step(agent.act(obs[None], [z])[0])
        th, thd = np.array(th), np.array(thd)
        th[1:][np.abs(np.diff(th)) > np.pi] = np.nan  # break the line where the angle wraps
        best = z == z_best
        ax.plot(th, thd, color=CHOSEN if best else OTHER, lw=2 if best else 1,
                alpha=1 if best else 0.7, zorder=3 if best else 2)
    ax.axvline(0, color=MUTED, lw=1, ls='--')
    ax.set_xlabel('Pole angle, rad (0 = upright)')
    ax.set_ylabel('Angular velocity, rad/s')
    ax.set_title(f'Seed {seed}: its {cfg["num_skills"]} skills from one start state\n'
                 '(blue: chosen skill, highest task return; grey: the others)')

    for s in range(1, 6):
        df = pd.read_csv(PRETRAIN / f'seed_{s}' / 'skill_eval.csv')
        df = df[df.epoch == df.epoch.max()]
        best = df.return_mean.idxmax()
        ax2.scatter([s] * len(df), df.return_mean, s=30, color=OTHER, zorder=2)
        ax2.scatter([s], [df.loc[best, 'return_mean']], s=48, color=CHOSEN, edgecolor='white',
                    linewidth=1, zorder=3)
    ax2.axhline(THRESHOLD, color=MUTED, lw=1, ls='--')
    ax2.set_xticks(range(1, 6), [f'seed {s}' for s in range(1, 6)])
    ax2.set_ylabel('Task return of each skill\n(end of pretraining)')
    ax2.set_title('Chosen skill per pretrained seed (blue)')
    save(fig, out, 'pretrained_skills.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='docs/img')
    p.add_argument('--seed', type=int, default=0, help='bootstrap seed')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    runs = load(ROOTS)
    per_run = pd.DataFrame([dict(init=r['init'], unit=r['unit'], run=r['run'],
                                 **run_metrics(r['curve'], r['cfg'], THRESHOLD, [0], END))
                            for r in runs])
    tests = pd.read_csv(TESTS)

    fig_time_to_solve(per_run, out)
    fig_effects(tests, out)
    fig_start_vs_solve(per_run, out)
    fig_critic_values(out, rng)
    calib = fig_critic_calibration(out)
    fig_selection_cost(runs, tests, out, rng)
    icc, p_units, p_naive = fig_pretrain_seed_variance(per_run, out)
    fig_pretrained_skills(out)

    print('\nCritic calibration (Spearman rho of V against Monte-Carlo soft return):')
    print(calib.round(2).to_string(index=False))
    print(f'\nICC {icc:.2f} | Welch p by pretrained seed {p_units:.4f} | as independent runs {p_naive:.2e}')


if __name__ == '__main__':
    main()
