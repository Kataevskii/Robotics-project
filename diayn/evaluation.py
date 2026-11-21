import json
from pathlib import Path

import numpy as np
import torch

from .agent import DIAYN
from .envs import make_env


def rollout_skills(agent, envs, max_path_length, deterministic=False, record=False, render=False):
    """One episode per skill, all skills stepped in lockstep (envs[i] runs skill i).

    Returns the env return per skill, the mean log q(z|s) over each episode, and,
    if record/render, per-skill traces with observations, infos and frames.
    """
    S = len(envs)
    obs = np.stack([e.reset()[0] for e in envs])
    alive = np.ones(S, bool)
    ret, logq, steps = np.zeros(S), np.zeros(S), np.zeros(S)
    traces = [dict(obs=[o.copy()], info=[], frames=[]) for o in obs]
    if render:
        for i, e in enumerate(envs):
            traces[i]['frames'].append(e.render())
    for _ in range(max_path_length):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        logq[idx] += agent.log_q(obs[idx], idx)
        steps[idx] += 1
        for i, a in zip(idx, agent.act(obs[idx], idx, deterministic)):
            o, r, term, trunc, info = envs[i].step(a)
            ret[i] += r
            obs[i] = o
            if record:
                traces[i]['obs'].append(o.copy())
                traces[i]['info'].append(info)
            if render:
                traces[i]['frames'].append(envs[i].render())
            if term or trunc:
                alive[i] = False
    return ret, logq / np.maximum(steps, 1), traces


def load_snapshot(path, device='cpu'):
    """Rebuild the agent of a run from <run_dir>/snapshots/itr_<n>.pt. Returns (agent, cfg)."""
    path = Path(path)
    cfg = json.loads((path.parent.parent / 'config.json').read_text())
    env = make_env(cfg['env_id'], cfg['max_path_length'])
    agent = DIAYN(env.observation_space.shape[0], env.action_space.shape[0], cfg, device)
    agent.load_state_dict(torch.load(path, map_location=device))
    return agent, cfg
