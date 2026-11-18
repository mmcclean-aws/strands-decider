#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=3 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61237
F="aliasing_analysis|INFO|UserWarning|_manual_seed|guards.py|^\s*$|warnings.warn|PJRT_DEVICE"
python $HOME/howto/scripts/06d_overhead.py 2>&1 | grep -v -E "$F" | tail -3
NEURON_RT_IO_RING_CACHE_SIZE=32 python $HOME/howto/scripts/06d_overhead.py 2>&1 | grep -v -E "$F" | tail -3
NEURON_RT_IO_RING_CACHE_SIZE=32 NEURON_RT_DISABLE_EXECUTION_BARRIER=1 python $HOME/howto/scripts/06d_overhead.py 2>&1 | grep -v -E "$F" | tail -3
