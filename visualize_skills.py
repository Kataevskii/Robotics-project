"""Record one video per skill (as scripts/visualize_skills.py --separate_videos=True).

    python visualize_skills.py results/half-cheetah/default/seed_1/snapshots/itr_3000.pt
    python visualize_skills.py <snapshot> --skills 0 7 12 --max_path_length 300

Videos go to <run_dir>/videos/<snapshot>/skill_<z>.mp4. MuJoCo envs only. For the
point env use plot_traces.py. On a headless Linux machine set MUJOCO_GL=egl.
"""
import argparse
from pathlib import Path

import imageio

from diayn.envs import make_env
from diayn.evaluation import load_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('snapshot')
    p.add_argument('--skills', type=int, nargs='*', default=None, help='default: all')
    p.add_argument('--max_path_length', type=int, default=100)
    p.add_argument('--stochastic', action='store_true', help='sample actions (default: deterministic)')
    args = p.parse_args()

    agent, cfg = load_snapshot(args.snapshot)
    snap = Path(args.snapshot)
    out = snap.parent.parent / 'videos' / snap.stem
    out.mkdir(parents=True, exist_ok=True)
    for z in args.skills if args.skills is not None else range(cfg['num_skills']):
        env = make_env(cfg['env_id'], args.max_path_length, render_mode='rgb_array')
        obs, _ = env.reset(seed=z)
        frames = [env.render()]
        for _ in range(args.max_path_length):
            obs, _, term, trunc, _ = env.step(agent.act(obs[None], [z], not args.stochastic)[0])
            frames.append(env.render())
            if term or trunc:
                break
        imageio.mimsave(out / f'skill_{z:02d}.mp4', frames, fps=env.metadata.get('render_fps', 30))
        env.close()
        print(f'skill {z}: {len(frames)} frames')
    print(f'Videos in {out}')


if __name__ == '__main__':
    main()
