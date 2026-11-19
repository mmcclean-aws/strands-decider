#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61235
python $HOME/howto/scripts/02_nki_kernels.py 2>&1 | grep -v -E "aliasing_analysis|INFO|UserWarning|_manual_seed|guards.py|^\s*$|warnings.warn|PJRT_DEVICE" | tail -80
