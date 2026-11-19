#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=6 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61240
python $HOME/howto/scripts/07_eager_probe.py 2>&1 | grep -E "^OK|^FAIL|H2D"
echo "=== TP world=4 all_reduce with NEURON_RT_DISABLE_EXECUTION_BARRIER=1"
unset NEURON_RT_ROOT_COMM_ID
NEURON_RT_DISABLE_EXECUTION_BARRIER=1 ONLY_TESTS=all_reduce timeout 600 torchrun --nproc_per_node=4 --master_port=29704 $HOME/howto/scripts/03_tp_collectives.py 2>&1 | grep -E "^world|^all_reduce|^CPL"
echo "=== TP world=4 all_reduce default barrier (with IO ring cache 32)"
ONLY_TESTS=all_reduce timeout 600 torchrun --nproc_per_node=4 --master_port=29705 $HOME/howto/scripts/03_tp_collectives.py 2>&1 | grep -E "^world|^all_reduce|^CPL"
