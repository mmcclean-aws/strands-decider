#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61235
python $HOME/howto/scripts/02_nki_kernels.py 2>&1 | grep -E "maxrel|FAILED"
echo "--- lite route detection with VLLM_NEURON_BACKEND unset:"
env -u VLLM_NEURON_BACKEND python -c "import libtorch_neuronx_lite as l; print(l.COMPILE_CACHE_STATE)" 2>&1 | tail -1
