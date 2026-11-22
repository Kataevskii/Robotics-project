"""Launch the policy-initialization experiment: every arm of finetune.py, in parallel processes.

    python finetune_sweep.py --pretrain_root results/pendulum/default --out results/pendulum/finetune --jobs 8

- diayn and actor arms: --reps finetuning runs from each pretrained seed, in
  <out>/<init>/pre<p>_seed<s>.
- random arm: --random_seeds runs, in <out>/random/seed_<s>.
- Finetuning seeds are 1, 2, ... in every arm, so run k of each arm shares its env and
  sampling seeds.
- Extra arguments (e.g. --n_epochs 50) are passed to finetune.py. Rerunning the command skips
  finished runs and restarts unfinished ones.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain_root', required=True, help='e.g. results/pendulum/default')
    p.add_argument('--pretrain_seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--reps', type=int, default=2, help='finetuning runs per pretrained seed')
    p.add_argument('--random_seeds', type=int, default=10, help='runs of the random arm')
    p.add_argument('--inits', nargs='+', default=['diayn', 'actor', 'random'])
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=1)
    args, extra = p.parse_known_args()
    root, out = Path(args.pretrain_root), Path(args.out)

    runs = []  # (finetune seed, init, pretrained run, log dir)
    for init in args.inits:
        if init == 'random':
            for s in range(1, args.random_seeds + 1):
                runs.append((s, init, root / f'seed_{args.pretrain_seeds[0]}', out / init / f'seed_{s}'))
        else:
            s = 0
            for ps in args.pretrain_seeds:
                for _ in range(args.reps):
                    s += 1
                    runs.append((s, init, root / f'seed_{ps}', out / init / f'pre{ps}_seed{s}'))
    runs.sort(key=lambda r: r[0])  # interleave the arms, so a partial sweep stays balanced

    def run(spec):
        seed, init, pre, log_dir = spec
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / 'finetune.py'), '--pretrain_dir', str(pre), '--init', init,
               '--seed', str(seed), '--log_dir', str(log_dir), *extra]
        with open(log_dir / 'stdout.log', 'a') as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        status = 'done' if code == 0 else f'FAILED (exit {code}), see {log_dir / "stdout.log"}'
        print(f'{log_dir}: {status}', flush=True)
        return code

    print(f'{len(runs)} runs, {args.jobs} at a time')
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(run, runs))
    sys.exit(max(codes))


if __name__ == '__main__':
    main()
