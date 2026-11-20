# Serving on AWS Trainium2 (EC2)

This guide hosts a strands decider model on a `trn2.48xlarge` EC2 instance, with
`strands-decider serve --device neuron` ([docs/inference.md, Serving on AWS Trainium](../../docs/inference.md#serving-on-aws-trainium)).
One server runs on one logical NeuronCore. The instance has 64 of them (16 Trainium2 chips at
LNC=2, four logical cores each), so it holds up to 64 independent servers. A request with a long
state can run instead on a tensor-parallel group of two or four cores of one chip.

It covers the environment, one server, a server per core, tensor parallel, and how to choose
between them ([cost_model/](cost_model/)). The Neuron SDK notes the port was built on are in
[howto/](howto/native_stack_howto.md).

## Environment

Use the Neuron Deep Learning Container for PyTorch inference, SDK 2.33 or later, for example
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`.
It has the Neuron SDK's PyTorch-native backend (`libtorch_neuronx_lite`), the compiler, NKI and
nkilib. Install the package in it with the `neuron` extra, whose packages are on the Neuron
repository rather than PyPI:

```bash
pip install "strands-decider[neuron] @ git+https://github.com/strands-labs/strands-decider" \
  --extra-index-url https://pip.repos.neuron.amazonaws.com
```

The container needs the host's Neuron devices (`--device /dev/neuron0` ... `--device /dev/neuron15`).
Run it with host networking or with no published ports: the servers below listen on `127.0.0.1`
only.

The engine sets the runtime variables it needs unless they are set: `VLLM_NEURON_BACKEND=neuron_native`
(before `libtorch_neuronx_lite` is imported), a `NEURON_RT_ROOT_COMM_ID` port that follows the
first visible core, `NEURON_RT_IO_RING_CACHE_SIZE=32` and
`NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=32`. Set `NEURON_LIBTORCH_CACHE_ROOT` to a directory
that outlives the container, so the compiled graphs are kept between starts. Do not share
torch's own compile cache (`TORCHINDUCTOR_CACHE_DIR`) between hosts: its entries name NKI kernel
files under `/var/tmp/nki-intermediate-cache` on the host that compiled them, and on another host
the compile fails with `NCC_EVRF059`.

## One server

```bash
NEURON_RT_VISIBLE_CORES=0 strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --device neuron --port 8100
curl -s localhost:8100/health
```

`serve` compiles every graph the engine uses before it answers `/health`: whole prompts in
token buckets of 256 to 4,096 (with several rows per forward for short ones), a shared state in
the same buckets, and question suffixes of 128 tokens in batches of 2 to 8 (`neuron_engine.NeuronConfig`).
With the default 67 shapes, a first start took 42 minutes; a start with the compiled graphs in
`NEURON_LIBTORCH_CACHE_ROOT` took 16, most of it torch tracing each shape again. Allow for it in
any health-check timeout. `--no-warmup` serves at once and compiles each shape on the first
request that needs it.

## A server per core

[`scripts/serve_replicas.sh`](scripts/serve_replicas.sh) starts one server per logical core,
on consecutive loopback ports. The first one compiles; the rest start together once it is
healthy and load its graphs from the shared cache (four replicas: 45 minutes for the first, 17
more for the others):

```bash
deploy/trainium2/scripts/serve_replicas.sh StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 64 0 8100
# 64 servers on 127.0.0.1:8100-8163, on logical cores 0-63
kill $(cat replica-logs/pids)
```

Each server takes one request at a time on its core, so spread requests over the ports with a
load balancer of your choice running on the instance and listening on `127.0.0.1` too.

## Tensor parallel (experimental)

Over two adjacent cores of one chip, a 4,096-token forward takes 88 ms instead of 134 ms. That
pays for long states only: the collectives cost more of a short forward, and two cores serving
separately answer more short requests than one group of two.

```bash
python -m strands_decider.neuron_tp serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --tensor-parallel 2 --core-base 60 --port 8200
```

The group takes cores `--core-base` to `--core-base + N - 1`. Keep a group within one chip, whose
four logical cores are `4k` to `4k + 3`: start a TP4 group on a multiple of 4, a TP2 group on a
multiple of 2. The ranks run in one process each and serve from rank 0.

## Choosing a layout

[`cost_model/cost_model.py`](cost_model/cost_model.py) estimates a forward's latency, MFU and an
instance's throughput for each way of splitting the model over cores (data parallel, tensor
parallel with and without sequence parallel, context parallel, Ulysses, pipeline). It uses the
measured trn2 collective latencies in [`cost_model/trn2_lnc2_collectives.json`](cost_model/trn2_lnc2_collectives.json),
and `--measured` calibrates its kernel efficiencies to the measured single-core forwards:

```bash
python deploy/trainium2/cost_model/cost_model.py --measured
```

Calibrated, it fits the seven measured single-core points within 3% and predicts the four
measured tensor-parallel points, which it was not fitted to, within 2.5 to 12.6%.

With JevBench's lengths (median 178 tokens, 90th percentile 2,471, 99th 3,072), the servers
per core take 96.6% of requests: route a request over about 2,800 tokens, or one with several
questions over a state of more than about 2,000 tokens, to a TP2 group. The measurements are in
[evaluation/results.md](../../evaluation/results.md#serving-on-aws-trainium2-accuracy-and-latency).

## Access

The servers have no authentication and listen on `127.0.0.1`
([Security](../../README.md#security)). To call them from another machine, forward a port
over SSH or AWS Systems Manager Session Manager, and do not bind them to other interfaces. To
serve other callers, put an authenticated front in front of them, as the
[SageMaker guide](../sagemaker/README.md) does with IAM-authenticated requests.
