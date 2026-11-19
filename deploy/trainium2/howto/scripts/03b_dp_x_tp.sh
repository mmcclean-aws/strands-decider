#!/bin/bash
source $HOME/howto/scripts/common_env.sh
unset NEURON_RT_ROOT_COMM_ID
F="^world|^all_reduce|TP=|FAILED|Error"
echo "=== (a) two independent TP=2 jobs at the same time: cores 0-1 and 2-3 (own master_port + root comm port each)"
CORE_BASE=0 ONLY_TESTS=all_reduce timeout 600 torchrun --nproc_per_node=2 --master_port=29801 $HOME/howto/scripts/03_tp_collectives.py > /tmp/tpA.log 2>&1 &
CORE_BASE=2 ONLY_TESTS=all_reduce timeout 600 torchrun --nproc_per_node=2 --master_port=29802 $HOME/howto/scripts/03_tp_collectives.py > /tmp/tpB.log 2>&1 &
wait
grep -E "$F" /tmp/tpA.log | head -5; grep -E "$F" /tmp/tpB.log | head -5
echo "=== (b) one world=4 job with two TP=2 sub-groups"
CORE_BASE=4 TP=2 timeout 600 torchrun --nproc_per_node=4 --master_port=29803 $HOME/howto/scripts/03b_subgroups.py 2>&1 | grep -E "^world|Error|error" | head -5
