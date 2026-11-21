#!/usr/bin/env bash
# Start a sweep that survives closing the browser tab, and mirror results/ to S3 every 15 min.
#   bash onyxia/run.sh <s3-destination> <sweep.py arguments>
#   bash onyxia/run.sh s3/<your-username>/diayn --env half-cheetah --seeds 1 2 3 4 5 --jobs 5
#
# To resume in a new service: mc mirror <s3-destination>/results results, then rerun the
# same command. Runs continue from their checkpoint.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
dest=$1
shift
mkdir -p results

nohup python sweep.py "$@" > "results/sweep_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
pid=$!
nohup bash -c "
  while kill -0 $pid 2>/dev/null; do
    mc mirror --overwrite --quiet results '$dest/results'
    sleep 900
  done
  mc mirror --overwrite --quiet results '$dest/results'
" >> results/sync.log 2>&1 &

echo "Sweep running (PID $pid). Logs: results/<env>/<tag>/seed_*/stdout.log"
echo "Mirrored to $dest/results every 15 minutes and once at the end."
