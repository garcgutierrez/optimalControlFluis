#!/bin/bash
# Resolution check of the outer case at high Reynolds numbers (after run_re.sh)
cd "$(dirname "$0")"
while pgrep -f "reynolds.py" > /dev/null; do sleep 30; done
for a in wall distributed; do python3 -u reynolds.py 10000 outer $a 320; python3 -u reynolds.py 5000 outer $a 256; done > data/re_conv2.log 2>&1
