#!/bin/bash
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=4 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61238
F="aliasing_analysis|INFO|UserWarning|_manual_seed|guards.py|^\s*$|warnings.warn|PJRT_DEVICE|WARNING"
C=/tmp/howto_cold_cache; rm -rf $C; mkdir -p $C
export NEURON_LIBTORCH_CACHE_ROOT=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor
S=$HOME/howto/scripts/05_caching.py
( time python $S cold ) 2>&1 | grep -v -E "$F" | grep -E "^\[|real|Error|error" | head -20
echo "--- cache tree after cold run:"; du -sh $C/neuron/native_neff_cache $C/inductor 2>/dev/null; ls $C/inductor; ls $C/inductor/dynamo 2>/dev/null | head; find $C/neuron/native_neff_cache -name "*.neff" | wc -l
( time python $S warm ) 2>&1 | grep -v -E "$F" | grep -E "^\[|real|Error|error" | head -20
rm -rf $C/inductor
( time python $S neff_only ) 2>&1 | grep -v -E "$F" | grep -E "^\[|real|Error|error" | head -20
python $S recompile 2>&1 | grep -v -E "$F" | grep -E "^\[|recompile|shape|Error" | head -30
