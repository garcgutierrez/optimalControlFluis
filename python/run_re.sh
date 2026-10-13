#!/bin/bash
# Reynolds-number study (runs sequentially per resolution group)
cd "$(dirname "$0")"
for Rt in 550 1000 2000; do for c in inner outer; do for a in wall distributed; do python3 -u reynolds.py $Rt $c $a 128; done; done; done > data/re_low.log 2>&1 &
for c in inner outer; do for a in wall distributed; do python3 -u reynolds.py 5000 $c $a 192; done; done > data/re_5000.log 2>&1 &
for c in inner outer; do for a in wall distributed; do python3 -u reynolds.py 10000 $c $a 256; done; done > data/re_10000.log 2>&1 &
for c in inner; do for a in wall distributed; do python3 -u reynolds.py 10000 $c $a 320; python3 -u reynolds.py 5000 $c $a 256; done; done > data/re_conv.log 2>&1 &
wait
