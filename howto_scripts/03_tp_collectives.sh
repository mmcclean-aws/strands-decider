#!/bin/bash
source $HOME/howto/scripts/common_env.sh
unset NEURON_RT_ROOT_COMM_ID   # the script derives one per group
for W in 2 4 8; do
  echo "===== world=$W"
  timeout 900 torchrun --nproc_per_node=$W --master_port=$((29600+W)) $HOME/howto/scripts/03_tp_collectives.py 2>&1 \
    | grep -v -E "aliasing_analysis|INFO|WARNING|UserWarning|_manual_seed|guards.py|^\s*$|warnings.warn|PJRT_DEVICE" | tail -25
done
