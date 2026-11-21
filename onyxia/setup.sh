#!/usr/bin/env bash
# One-time setup inside an Onyxia service (e.g. vscode-pytorch-gpu).
#   bash onyxia/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# The pytorch images already ship torch with CUDA; requirements.txt only asks for torch>=2.2,
# so pip keeps the preinstalled build.
pip install -r requirements.txt

python - <<'EOF'
import gymnasium as gym
import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
env = gym.make('HalfCheetah-v5')
env.reset(seed=0)
env.step(env.action_space.sample())
print('MuJoCo OK')
EOF

# Two short epochs on the GPU (if any) to check the training path and measure throughput.
python train.py --env hopper --seed 0 --log_dir /tmp/diayn_smoke --n_epochs 2 \
  --find_best_skill_interval 2 --best_skill_n_rollouts 1
rm -rf /tmp/diayn_smoke
