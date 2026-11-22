"""DIAYN on SAC v1, following ben-eysenbach/sac (sac/algos/diayn.py, sac/distributions/gmm.py).

SAC v1 means: one Q function, a state-value function V with a soft-updated target V,
and a GMM policy squashed by tanh whose sample carries no gradient, so the actor is
trained with the original likelihood-ratio surrogate.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
LOG_SIG_MIN, LOG_SIG_MAX, LOG_W_MIN = -5.0, 2.0, -10.0
STAT_NAMES = ['qf', 'vf', 'bellman_error', 'discriminator_loss', 'discriminator_acc',
              'discriminator_reward', 'log_pi', 'kl_surrogate_loss', 'vf_loss']


def mlp(in_dim, out_dim, hidden):
    """ReLU MLP, linear output, Xavier-uniform weights and zero biases (sac/misc/mlp.py)."""
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class GMMPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, K, hidden, reg):
        super().__init__()
        self.K, self.Da, self.reg = K, act_dim, reg
        self.net = mlp(obs_dim, K * (2 * act_dim + 1), hidden)

    def forward(self, obs):
        out = self.net(obs).view(-1, self.K, 2 * self.Da + 1)
        log_w = out[..., 0].clamp(min=LOG_W_MIN)
        mu = out[..., 1:1 + self.Da]
        log_sig = out[..., 1 + self.Da:].clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return log_w, mu, log_sig

    @staticmethod
    @torch.no_grad()
    def sample(log_w, mu, log_sig):
        """Pre-tanh sample x. The original stop-gradients x."""
        k = torch.distributions.Categorical(logits=log_w).sample()
        i = torch.arange(len(k), device=k.device)
        return mu[i, k] + log_sig[i, k].exp() * torch.randn_like(mu[i, k])

    def log_prob(self, log_w, mu, log_sig, x):
        """Mixture log-density of x (pre-tanh) and the L2 regulariser on mu and log_sig."""
        z = (x[:, None, :] - mu) * torch.exp(-log_sig)
        log_p_xz = -0.5 * (z ** 2).sum(-1) - log_sig.sum(-1) - 0.5 * self.Da * np.log(2 * np.pi)
        log_p = torch.logsumexp(log_p_xz + log_w, 1) - torch.logsumexp(log_w, 1)
        reg = self.reg * 0.5 * ((log_sig ** 2).mean() + (mu ** 2).mean())
        return log_p, reg


class DIAYN:
    def __init__(self, obs_dim, act_dim, cfg, device):
        S, hidden = cfg['num_skills'], [cfg['layer_size']] * 2
        self.S, self.device = S, device
        self.discount, self.tau = cfg['discount'], cfg['tau']
        self.scale_entropy, self.add_p_z = cfg['scale_entropy'], cfg['add_p_z']

        aug = obs_dim + S
        self.policy = GMMPolicy(aug, act_dim, cfg['K'], hidden, cfg['reg']).to(device)
        self.qf = mlp(aug + act_dim, 1, hidden).to(device)
        self.vf = mlp(aug, 1, hidden).to(device)
        self.vf_target = mlp(aug, 1, hidden).to(device)
        self.disc = mlp(obs_dim, S, hidden).to(device)  # sees the state only, not the skill
        # Adam is element-wise, so one optimiser over all nets is the original's four
        # optimisers with the same learning rate.
        self.opt = torch.optim.Adam([*self.policy.parameters(), *self.qf.parameters(),
                                     *self.vf.parameters(), *self.disc.parameters()], lr=cfg['lr'])
        # Uniform, fixed prior. Kept as a tensor so it can be changed for prior experiments.
        self.p_z = torch.full((S,), 1.0 / S, device=device)
        # The original's target V is initialised independently and soft-updated once at start.
        self._update_target()

    def aug(self, obs, z):
        return torch.cat([obs, F.one_hot(z, self.S).float()], -1)

    def _tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    @torch.no_grad()
    def act(self, obs, z, deterministic=False):
        """Batched actions in [-1, 1] for observations obs (B, Do) and skills z (B,)."""
        o = self.aug(self._tensor(obs), self._tensor(z, torch.long))
        log_w, mu, log_sig = self.policy(o)
        if deterministic:
            # As GMMPolicy: mean of the highest-Q component, not squashed (the env wrapper clips).
            B, K, Da = mu.shape
            q = self.qf(torch.cat([o.repeat_interleave(K, 0), mu.reshape(B * K, Da)], -1))
            a = mu[torch.arange(B, device=mu.device), q.view(B, K).argmax(1)]
        else:
            a = torch.tanh(GMMPolicy.sample(log_w, mu, log_sig))
        return a.cpu().numpy()

    @torch.no_grad()
    def log_q(self, obs, z):
        """log q(z|s) of the discriminator."""
        z = self._tensor(z, torch.long)
        return F.log_softmax(self.disc(self._tensor(obs)), -1).gather(1, z[:, None]).squeeze(1).cpu().numpy()

    def update(self, b):
        """One gradient step on all nets from the same parameters, then the target update.

        If the batch carries a 'reward' (finetuning on the task, finetune.py), it replaces the
        pseudo-reward and the discriminator gets no gradient, so Adam leaves it unchanged.
        """
        o, o2 = self.aug(b['obs'], b['z']), self.aug(b['next_obs'], b['z'])

        # Discriminator, and pseudo-reward log q(z|s_t) - log p(z) on the current state.
        logits = self.disc(b['obs'])
        log_q = F.log_softmax(logits, -1).gather(1, b['z'][:, None]).squeeze(1)
        if 'reward' in b:
            reward = b['reward']
            disc_loss = torch.zeros((), device=reward.device)
        else:
            disc_loss = -log_q.mean()
            reward = log_q.detach()
            if self.add_p_z:
                reward = reward - torch.log(self.p_z[b['z']] + EPS)

        # Q function.
        with torch.no_grad():
            y = reward + (1 - b['done']) * self.discount * self.vf_target(o2).squeeze(1)
        q = self.qf(torch.cat([o, b['act']], -1)).squeeze(1)
        qf_loss = 0.5 * ((y - q) ** 2).mean()

        # Policy (likelihood-ratio surrogate) and V function.
        log_w, mu, log_sig = self.policy(o)
        x = GMMPolicy.sample(log_w, mu, log_sig)
        log_pi, reg = self.policy.log_prob(log_w, mu, log_sig, x)
        with torch.no_grad():
            corr = torch.log(1 - torch.tanh(x) ** 2 + EPS).sum(1)
            q_pi = self.qf(torch.cat([o, torch.tanh(x)], -1)).squeeze(1)
            scaled_log_pi = self.scale_entropy * (log_pi - corr)
        v = self.vf(o).squeeze(1)
        kl_loss = (log_pi * (scaled_log_pi - q_pi + v.detach())).mean()
        vf_loss = 0.5 * ((v - (q_pi - scaled_log_pi)) ** 2).mean()

        self.opt.zero_grad()
        (disc_loss + qf_loss + kl_loss + reg + vf_loss).backward()
        self.opt.step()
        self._update_target()

        acc = (logits.argmax(1) == b['z']).float().mean()
        return torch.stack([q.mean(), v.mean(), qf_loss, disc_loss, acc, reward.mean(),
                            log_pi.mean(), kl_loss, vf_loss]).detach()

    @torch.no_grad()
    def _update_target(self):
        for pt, p in zip(self.vf_target.parameters(), self.vf.parameters()):
            pt.mul_(1 - self.tau).add_(p, alpha=self.tau)

    def nets(self):
        return dict(policy=self.policy, qf=self.qf, vf=self.vf, vf_target=self.vf_target,
                    disc=self.disc)

    def state_dict(self):
        sd = {k: n.state_dict() for k, n in self.nets().items()}
        sd['opt'], sd['p_z'] = self.opt.state_dict(), self.p_z
        return sd

    def load_state_dict(self, sd):
        for k, n in self.nets().items():
            if k in sd:
                n.load_state_dict(sd[k])
        if 'opt' in sd:
            self.opt.load_state_dict(sd['opt'])
        if 'p_z' in sd:
            self.p_z = sd['p_z'].to(self.device)


class ReplayBuffer:
    """Stores env observations and the skill index. The one-hot is appended when sampling."""

    def __init__(self, obs_dim, act_dim, size, reward=False):
        self.max, self.size, self.ptr = size, 0, 0
        self.data = dict(obs=np.zeros((size, obs_dim), np.float32),
                         next_obs=np.zeros((size, obs_dim), np.float32),
                         act=np.zeros((size, act_dim), np.float32),
                         done=np.zeros(size, np.float32),
                         z=np.zeros(size, np.int64))
        if reward:  # env reward, only stored when finetuning on the task
            self.data['reward'] = np.zeros(size, np.float32)

    def add(self, **tr):
        for k, v in tr.items():
            self.data[k][self.ptr] = v
        self.ptr = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, n, rng, device):
        idx = rng.integers(0, self.size, n)  # with replacement, as SimpleReplayBuffer
        return {k: torch.as_tensor(v[idx], device=device) for k, v in self.data.items()}

    def state_dict(self):
        return dict(size=self.size, ptr=self.ptr,
                    data={k: v[:self.size].copy() for k, v in self.data.items()})

    def load_state_dict(self, sd):
        self.size, self.ptr = sd['size'], sd['ptr']
        for k, v in sd['data'].items():
            self.data[k][:self.size] = v
