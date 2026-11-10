#!/bin/bash
source $HOME/howto/scripts/common_env.sh
F="aliasing_analysis|INFO|UserWarning|_manual_seed|guards.py|^\s*$|warnings.warn|PJRT_DEVICE|WARNING"
echo "== solo (core 0)"
NEURON_RT_VISIBLE_CORES=0 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61234 python $HOME/howto/scripts/04_dp_replica.py 2>&1 | grep -v -E "$F" | tail -2
echo "== 8 concurrent replicas, cores 0-7, unique NEURON_RT_ROOT_COMM_ID per process"
for c in 0 1 2 3 4 5 6 7; do
  NEURON_RT_VISIBLE_CORES=$c NEURON_RT_ROOT_COMM_ID=127.0.0.1:$((61234+c)) python $HOME/howto/scripts/04_dp_replica.py > /tmp/dp_$c.log 2>&1 &
done
wait
for c in 0 1 2 3 4 5 6 7; do grep -v -E "$F" /tmp/dp_$c.log | tail -1; done
echo "== 8 concurrent replicas sharing ONE NEURON_RT_ROOT_COMM_ID"
for c in 0 1 2 3 4 5 6 7; do
  NEURON_RT_VISIBLE_CORES=$c NEURON_RT_ROOT_COMM_ID=127.0.0.1:61234 python $HOME/howto/scripts/04_dp_replica.py > /tmp/dps_$c.log 2>&1 &
done
wait
for c in 0 1 2 3 4 5 6 7; do grep -v -E "$F" /tmp/dps_$c.log | tail -1; done
echo "== 2 processes on the SAME core (expect failure: core already in use)"
for i in 0 1; do NEURON_RT_VISIBLE_CORES=0 NEURON_RT_ROOT_COMM_ID=127.0.0.1:$((61300+i)) python $HOME/howto/scripts/04_dp_replica.py > /tmp/dpx_$i.log 2>&1 & done
wait
for i in 0 1; do grep -v -E "$F" /tmp/dpx_$i.log | grep -E "core=|Error|error" | head -3; done
