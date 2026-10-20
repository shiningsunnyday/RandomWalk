#!/usr/bin/env bash
# Third queue: starts once the `log` arm (queue1's first arm) has drained, keeping total concurrency sane.
H=$(cd "$(dirname "$0")" && pwd)
while pgrep -f "run_arm.sh.*jobs/log.txt" > /dev/null; do sleep 60; done
echo "log arm drained $(date '+%F %T'); starting queue3 (tseed)"
bash "$H/queue.sh" third "tseed.txt:tseed:36"
