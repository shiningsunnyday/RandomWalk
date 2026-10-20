#!/usr/bin/env bash
# Sequential arm queue. usage: queue.sh <name> "<jobfile>:<arm>:<parallel>" ...
H=$(cd "$(dirname "$0")" && pwd); NAME=$1; shift
for spec in "$@"; do
  IFS=: read -r J ARM P <<< "$spec"
  echo "=== $(date '+%F %T') START $ARM ($(wc -l < $H/jobs/$J) runs, P=$P)"
  bash "$H/run_arm.sh" "$H/jobs/$J" "$ARM" "$P" >> "$H/logs/${ARM}_driver.log" 2>&1
  echo "=== $(date '+%F %T') END $ARM"
done
echo "=== QUEUE $NAME COMPLETE $(date '+%F %T')"
