#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=7 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61241
run() { # $1=cache tag $2=flags
  export NEURON_LIBTORCH_CACHE_ROOT=/tmp/howto_flags_$1 TORCHINDUCTOR_CACHE_DIR=/tmp/howto_flags_$1/inductor
  echo -n "cache=$1 "; python $HOME/howto/scripts/08_autocast.py $2 2>&1 | grep -E "^flags|Error" | head -3
}
rm -rf /tmp/howto_flags_*
run A default
run A vllm_neuron        # same cache, different flags -> stale reuse?
run B autocast_none
run C vllm_neuron
rm -rf /tmp/howto_flags_A/inductor
run A vllm_neuron        # only NEFF cache left
