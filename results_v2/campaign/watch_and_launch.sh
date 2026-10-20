#!/usr/bin/env bash
# Wait until the first-wave control arms have drained, then start the second queue so two arms run at once.
H=$(cd "$(dirname "$0")" && pwd)
while pgrep -f "run_arm.sh $H/jobs/base_undir.txt" > /dev/null || pgrep -f "run_arm.sh.*base_undir" > /dev/null; do sleep 60; done
echo "control arms drained $(date '+%F %T'); starting queue2"
bash "$H/queue.sh" second "log_smooth.txt:log_smooth:56" "combo_y.txt:combo_y:56" "combo_scaf.txt:combo_scaf:56" "huber.txt:huber:56"
