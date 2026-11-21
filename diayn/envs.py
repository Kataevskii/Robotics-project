import gymnasium as gym
import numpy as np


class PointEnv(gym.Env):
    """2D point mass: the state is the position and the action a bounded displacement.

    The authors' 'point-rllab' id is not defined in their repo. This follows rllab's
    tutorial PointEnv (start uniform in [-1, 1]^2, step <= 0.1 per axis, reward
    -||s||, done within 0.01 of the origin).
    """

    observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float64)
    action_space = gym.spaces.Box(-0.1, 0.1, (2,), np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.np_random.uniform(-1, 1, size=2)
        return self._state.copy(), {}

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        done = bool(abs(x) < 0.01 and abs(y) < 0.01)
        return self._state.copy(), -float(np.hypot(x, y)), done, False, {}


class NormalizedActions(gym.ActionWrapper):
    """rllab's normalize(): the agent acts in [-1, 1], rescaled to the env bounds and clipped."""

    def __init__(self, env):
        super().__init__(env)
        self.lb, self.ub = env.action_space.low, env.action_space.high
        self.action_space = gym.spaces.Box(-1.0, 1.0, env.action_space.shape, np.float32)

    def action(self, a):
        return np.clip(self.lb + (a + 1.0) * 0.5 * (self.ub - self.lb), self.lb, self.ub)


def make_env(env_id, max_path_length, render_mode=None):
    if env_id == 'point':
        env = gym.wrappers.TimeLimit(PointEnv(), max_path_length)
    else:
        env = gym.make(env_id, max_episode_steps=max_path_length, render_mode=render_mode)
    return NormalizedActions(env)
