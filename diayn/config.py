"""Hyperparameters of the original DIAYN launcher (ben-eysenbach/sac, examples/mujoco_all_diayn.py).

Gym v1/v0 ids are replaced by their current Gymnasium equivalents. Box2D envs
(lunar-lander, bipedal-walker) are left out to avoid the box2d dependency.
"""

SHARED_PARAMS = dict(
    lr=3e-4,
    discount=0.99,
    tau=0.01,
    K=4,                      # GMM components of the policy
    reg=1e-3,                 # L2 regulariser on GMM means / log-stds
    layer_size=300,
    batch_size=128,
    max_pool_size=int(1e6),
    n_train_repeat=1,
    epoch_length=1000,
    num_skills=50,
    scale_entropy=0.1,        # alpha
    add_p_z=True,             # pseudo-reward is log q(z|s) - log p(z)
    find_best_skill_interval=10,
    best_skill_n_rollouts=10,
    # Not in the original (storage / pre-emption on Onyxia):
    snapshot_gap=100,         # original: 10
    checkpoint_interval=100,  # full resumable state, overwritten in place
)

ENV_PARAMS = {
    'swimmer': dict(env_id='Swimmer-v5', max_path_length=1000, n_epochs=10000),
    'hopper': dict(env_id='Hopper-v5', max_path_length=1000, n_epochs=10000),
    'half-cheetah': dict(env_id='HalfCheetah-v5', max_path_length=1000, n_epochs=10000,
                         max_pool_size=int(1e7)),
    'walker': dict(env_id='Walker2d-v5', max_path_length=1000, n_epochs=10000),
    'ant': dict(env_id='Ant-v5', max_path_length=1000, n_epochs=10000),
    'humanoid': dict(env_id='Humanoid-v5', max_path_length=1000, n_epochs=20000),
    'point': dict(env_id='point', max_path_length=100, n_epochs=50, layer_size=32),
    'inverted-pendulum': dict(env_id='InvertedPendulum-v5', max_path_length=1000, n_epochs=1000),
    'inverted-double-pendulum': dict(env_id='InvertedDoublePendulum-v5', max_path_length=1000,
                                     n_epochs=1000),
    'pendulum': dict(env_id='Pendulum-v1', max_path_length=200, n_epochs=1000, layer_size=32,
                     num_skills=5),
    'mountain-car': dict(env_id='MountainCarContinuous-v0', max_path_length=1000, n_epochs=1000,
                         add_p_z=False),
}


def get_config(env, **overrides):
    cfg = dict(SHARED_PARAMS)
    cfg.update(ENV_PARAMS[env])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg['env'] = env
    return cfg
