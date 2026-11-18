#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=0
python $HOME/howto/scripts/01_matmul.py 2>&1 | grep -v -E "^\s*$" | tail -60
ls -R /tmp/howto_cache | head -30
