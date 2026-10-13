#!/bin/bash
# Resolution check of the inner case at the highest Reynolds numbers
cd "$(dirname "$0")"
(python3 -u reynolds.py 10000 inner wall 384; python3 -u reynolds.py 5000 inner wall 320; python3 -u reynolds.py 2000 inner wall 192) > data/re_conv3.log 2>&1
