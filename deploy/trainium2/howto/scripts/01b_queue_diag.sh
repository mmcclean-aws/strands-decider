#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=0
for q in unset 32; do
  if [ $q = unset ]; then unset NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS; else export NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=$q; fi
  python $HOME/howto/scripts/01b_queue_diag.py 2>&1 | grep -v -E "aliasing_analysis|INFO|WARNING|UserWarning|_manual_seed|guards.py" | tail -12
done
