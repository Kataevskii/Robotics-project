"""Run several seeds of one configuration in parallel processes.

    python sweep.py --env half-cheetah --seeds 1 2 3 4 5 --jobs 5
    python sweep.py --env half-cheetah --seeds 1 2 3 --tag alpha0.01 --scale_entropy 0.01

Each run goes to <log_root>/<env>/<tag>/seed_<s> with its output in stdout.log.
Extra arguments are passed to train.py. Rerunning the command resumes unfinished runs.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--jobs', type=int, default=1, help='runs at the same time')
    p.add_argument('--tag', default='default', help='name of this configuration')
    p.add_argument('--log_root', default='results')
    args, extra = p.parse_known_args()

    def run(seed):
        log_dir = Path(args.log_root) / args.env / args.tag / f'seed_{seed}'
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / 'train.py'), '--env', args.env, '--seed', str(seed),
               '--log_dir', str(log_dir), *extra]
        with open(log_dir / 'stdout.log', 'a') as f:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        status = 'done' if code == 0 else f'FAILED (exit {code}), see {log_dir / "stdout.log"}'
        print(f'seed {seed}: {status}', flush=True)
        return code

    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(run, args.seeds))
    sys.exit(max(codes))


if __name__ == '__main__':
    main()
