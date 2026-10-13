#!/bin/bash
# Minimal job queue: runs the commands of a file (one per line), keeping at most MAXJ study processes alive.
cd "$(dirname "$0")"
MAXJ=${MAXJ:-12}
count() { pgrep -fc "^python3 -u (sweep_pdrag|ql_study|reynolds|ql_diag|ql_conv|pdrag_extra)" ; }
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  while [ "$(count)" -ge "$MAXJ" ]; do sleep 30; done
  echo "$(date +%H:%M:%S) start: $cmd" >> data/queue.log
  OPENBLAS_NUM_THREADS=1 nohup bash -c "$cmd" > /dev/null 2>&1 &
  sleep 20
done < "$1"
echo "$(date +%H:%M:%S) queue empty" >> data/queue.log
