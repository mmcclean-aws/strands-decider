# Source this on the trn2 host before running any howto python script.
export VLLM_PLUGINS=none                      # don't let vllm load the neuron platform plugin
export VLLM_NEURON_BACKEND=neuron_native      # tell libtorch_neuronx_lite this is the native route
export NEURON_LOGICAL_NC_CONFIG=2             # LNC=2 (default on trn2)
export NEURON_LIBTORCH_CACHE_ROOT=/tmp/howto_cache   # native NEFFs -> $ROOT/neuron/native_neff_cache
export NEURON_RT_LOG_LEVEL=ERROR
# NRT refuses to load ANY native NEFF (even single-core ones) without a CCOM root:
export NEURON_RT_ROOT_COMM_ID=${NEURON_RT_ROOT_COMM_ID:-127.0.0.1:61234}
# Async exec queue depth (default 8 -> "Execution Queue Full"); vllm-neuron worker uses 32.
export NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=${NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS:-32}
# Reuse the per-exec DMA I/O ring (vllm-neuron worker default). Without it a pipelined
# gdn_cte call costs 3.2 ms vs 1.2 ms with it (device time 1.06 ms).
export NEURON_RT_IO_RING_CACHE_SIZE=${NEURON_RT_IO_RING_CACHE_SIZE:-32}
mkdir -p $HOME/howto /tmp/howto_cache
cd $HOME/howto
