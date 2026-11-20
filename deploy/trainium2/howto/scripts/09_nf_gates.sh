#!/bin/bash
source $HOME/howto/scripts/common_env.sh
python $HOME/howto/scripts/09_nf_gates.py 2>&1 | grep -E "^TP=|Error"
