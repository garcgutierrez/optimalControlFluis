#!/bin/bash
# Keep the multi-threaded JAX jobs on cores 0-11, to leave room for other users of the machine.
while true; do
  for pid in $(pgrep -f "python3 -u (ql_study|pdrag_extra|ql_conv|ql_diag)"); do taskset -a -p -c 0-11 $pid > /dev/null 2>&1; done
  sleep 60
done
