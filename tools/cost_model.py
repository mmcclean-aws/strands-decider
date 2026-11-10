"""Analytic cost model: Strands Decider 2B (Qwen3.5-2B hybrid torso) prefill on trn2.48xlarge.

Every request is ONE prefill forward (no decode). The model estimates per-forward latency,
instance throughput and MFU for each sharding strategy over a group of P logical cores
(LNC=2), using measured collective latencies (collectives/trn2_lnc2_collectives.json,
nightly nccom-test data) and per-kernel efficiency factors that are calibrated on device
(CALIB below; update them from measurements).

Strategies (P = ranks in one model replica; the instance holds 64/P replicas):
  DP        P=1: whole model on one LNC2 core (replicas only).
  TP        Megatron TP: column/row-sharded projections + MLP, 2 all-reduces/layer of [T,H].
  TPSP      Megatron TP + SP: reduce-scatter + all-gather instead of each all-reduce.
  CP        Context parallel: tokens sharded T/P, weights replicated. Attention all-gathers
            K/V (GQA: 2 KV heads x 256). GDN uses the two-pass linear state correction
            (local pass with S0=0 and a transition pass with v=0,S0=I, all-gather of
            (S_local, M), prefix-compose, correction matmul) -> GDN core cost 2*core(T/P).
  ULY       Ulysses: tokens sharded for projections/MLP; all-to-all to head-sharding for the
            GDN core (16 heads) and attention core (8 q heads, KV replicated as needed).
  HYB       Ulysses for GDN layers + CP (K/V all-gather) for attention layers.
  TSP       Zyphra TSP (arXiv:2604.26294): tokens AND weights sharded on one axis; weights
            are broadcast / ring-passed every layer (P_L*(P-1)/P bytes per rank per layer)
            plus the CP K/V all-gather; GDN handled as in CP.
  PP        Pipeline over layers: no latency gain for one request; throughput == DP.
  EP        Not applicable: every layer is a dense SwiGLU MLP (no experts).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
COLL = json.load(open(os.path.join(HERE, "..", "collectives", "trn2_lnc2_collectives.json")))


# ---------------------------------------------------------------- model
@dataclass
class Model:
    H: int = 2048
    n_layers: int = 24
    n_gdn: int = 18
    n_attn: int = 6
    I: int = 6144  # SwiGLU intermediate
    # GDN
    gdn_hk: int = 16
    gdn_hv: int = 16
    gdn_dk: int = 128
    gdn_dv: int = 128
    conv_k: int = 4
    # attention
    nq: int = 8
    nkv: int = 2
    hd: int = 256
    vocab: int = 248320
    pointer_dim: int = 768
    dtype_bytes: int = 2

    @property
    def gdn_proj_params(self) -> int:
        qkv = (2 * self.gdn_hk * self.gdn_dk + self.gdn_hv * self.gdn_dv) * self.H
        z = self.gdn_hv * self.gdn_dv * self.H
        ab = 2 * self.gdn_hv * self.H
        out = self.gdn_hv * self.gdn_dv * self.H
        return qkv + z + ab + out

    @property
    def attn_proj_params(self) -> int:
        q = 2 * self.nq * self.hd * self.H  # query + sigmoid output gate
        kv = 2 * self.nkv * self.hd * self.H
        o = self.nq * self.hd * self.H
        return q + kv + o

    @property
    def mlp_params(self) -> int:
        return 3 * self.H * self.I

    @property
    def nonembed_params(self) -> int:
        return (self.n_gdn * (self.gdn_proj_params + self.mlp_params)
                + self.n_attn * (self.attn_proj_params + self.mlp_params))

    def gdn_core_flops_per_token(self, chunk: int = 128) -> float:
        # per head: K K^T, Q K^T, A^-1 (beta V), A^-1 (beta K decayed), intra attn @ v_new
        # (5 x 2*C*D) + k_cumdecay@S, Q@S, K^T@v_new (3 x 2*D*D)
        d = self.gdn_dk
        return self.gdn_hv * (10 * chunk * d + 6 * d * d)

    def attn_core_flops(self, T: int, ctx: int | None = None) -> float:
        """Causal attention FLOPs (QK^T + PV) for T queries attending to a prefix of ctx."""
        ctx = T if ctx is None else ctx
        # sum over queries of (#keys) ~ T*(ctx - T) + T^2/2
        pairs = T * (ctx - T) + T * T / 2
        return 4 * self.nq * self.hd * pairs

    def flops(self, T: int) -> dict[str, float]:
        f = {
            "gdn_proj": 2 * self.gdn_proj_params * T * self.n_gdn,
            "attn_proj": 2 * self.attn_proj_params * T * self.n_attn,
            "mlp": 2 * self.mlp_params * T * self.n_layers,
            "gdn_core": self.gdn_core_flops_per_token() * T * self.n_gdn,
            "attn_core": self.attn_core_flops(T) * self.n_attn,
        }
        f["total"] = sum(f.values())
        return f


# ---------------------------------------------------------------- hardware
@dataclass
class HW:
    name: str = "trn2.48xlarge (LNC=2)"
    cores: int = 64  # logical LNC2 cores
    peak_bf16_core: float = 667e12 / 4  # 166.75 TFLOP/s per LNC2 core (667 TF/chip, 4 cores)
    hbm_bw_core: float = 2.9e12 / 4  # 725 GB/s per LNC2 core
    hbm_bytes_core: float = 24 * 2**30


@dataclass
class Calib:
    """Kernel efficiency factors. Defaults are priors; replace with on-device measurements."""

    # GEMM efficiency vs tokens per rank (fraction of bf16 peak), piecewise-linear in log2(T)
    gemm_eff: dict[int, float] = field(default_factory=lambda: {
        64: 0.30, 128: 0.45, 256: 0.60, 512: 0.72, 1024: 0.80, 2048: 0.84, 4096: 0.86})
    attn_eff: float = 0.45  # head_dim 256 flash attention, fraction of peak
    # GDN chunked kernel (nkilib gdn_cte, schur): fixed + per (head x token) on ONE physical NC
    # From nkilib README trn2 bf16: (1,2,4,768) 192.7us, (1,1,1,8192) 441.9us, (1,1,1,128) 33.7us
    gdn_fixed_us: float = 25.0
    gdn_ns_per_head_token_nc: float = 55.0
    gdn_lnc_speedup: float = 2.0  # heads split over the 2 physical NCs of an LNC2 core
    # elementwise / norm / conv / activation traffic not fused into GEMMs: bytes per token per
    # layer moved through HBM (read+write) for RMSNorm x2, residual adds, conv1d, l2norm, gates
    elementwise_bytes_per_token_layer: float = 2 * 2048 * 2 * 6
    per_layer_overhead_us: float = 8.0  # kernel launch / sync / DMA setup per layer
    collective_scale: float = 1.0  # in-graph collective cost relative to nccom-test numbers
    overlap: float = 0.0  # fraction of collective time hidden behind compute
    elem_replicated_frac: float = 0.2  # share of elementwise work that TP cannot shard (norms, residual)


# On-device measurements (SDK 2.33 native backend, one LNC2 core, final v3 kernels: fused ops + gdn_fast):
# full forward ms by (T, B). Used to fit the two free parameters of measured_calib().
MEASURED_DP = {(512, 1): 20.72, (2048, 1): 67.28, (512, 4): 67.12, (4096, 1): 133.58,
               (1024, 1): 35.20, (2560, 1): 85.65, (256, 8): 68.28}
# Measured TP (v3, intra-chip adjacent cores, 4-way chunked all-reduce where noted) for validation.
MEASURED_TP = {("TP", 2, 2048): 47.8, ("TP", 2, 4096): 87.9, ("TP", 4, 2048): 27.3, ("TP", 4, 4096): 56.9}
# In-graph all-reduce vs nccom-test (4r intra-chip proxy): 402 us vs 129 us at 16 MiB -> x3.1.
IN_GRAPH_COLLECTIVE_SCALE = 3.1


def measured_calib(fit: bool = True) -> "Calib":
    """GEMM efficiency from the per-GEMM sweep (pre-transposed weights: 82-89% at T>=2048), the GDN
    core from gdn_fast timings (B=1, 16 heads, LNC2: 0.18/0.31/0.55/1.07 ms at T=512/1024/2048/4096 ->
    53 us fixed + 15.5 ns per head-token per LNC2 core), in-graph collectives at x3.1 the nccom numbers,
    and elementwise/per-layer overheads fitted to MEASURED_DP by least squares on a small grid."""
    c = Calib(gemm_eff={64: 0.12, 128: 0.25, 256: 0.45, 512: 0.65, 1024: 0.78, 2048: 0.86, 4096: 0.88},
              attn_eff=0.40, gdn_fixed_us=106.0, gdn_ns_per_head_token_nc=31.0, gdn_lnc_speedup=2.0,
              elementwise_bytes_per_token_layer=40e3, per_layer_overhead_us=20.0,
              collective_scale=IN_GRAPH_COLLECTIVE_SCALE)
    if not fit:
        return c
    best = None
    for eb in [x * 5e3 for x in range(0, 121)]:
        for ov in [x * 5.0 for x in range(0, 41)]:
            c.elementwise_bytes_per_token_layer, c.per_layer_overhead_us = eb, ov
            err = sum((math.log(estimate("DP", 1, T, B, cal=c).latency_ms / v)) ** 2 for (T, B), v in MEASURED_DP.items())
            if best is None or err < best[0]:
                best = (err, eb, ov)
    c.elementwise_bytes_per_token_layer, c.per_layer_overhead_us = best[1], best[2]
    return c


def interp_eff(table: dict[int, float], T: float) -> float:
    ks = sorted(table)
    if T <= ks[0]:
        return table[ks[0]] * T / ks[0]
    if T >= ks[-1]:
        return table[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= T <= b:
            w = (math.log2(T) - math.log2(a)) / (math.log2(b) - math.log2(a))
            return table[a] + w * (table[b] - table[a])
    raise AssertionError


def coll_us(op: str, nbytes: float, P: int, cal: Calib) -> float:
    """Collective latency (us), log-log interpolation of measured nccom data.

    nbytes = AG output / RS input / AR buffer / A2A total per-rank buffer.
    P=2 uses the 4-rank intra-chip numbers (no 2r data exists; conservative proxy).
    """
    if P <= 1 or nbytes <= 0:
        return 0.0
    table = COLL[op]
    key = str(min(16, max(4, 2 ** math.ceil(math.log2(P)))))
    ys = table[key]
    xs = COLL["sizes_bytes"]
    floor = COLL["small_msg_floor_us"].get(op, {}).get(key, ys[0])
    if nbytes <= xs[0]:
        y = floor + (ys[0] - floor) * nbytes / xs[0]
    elif nbytes >= xs[-1]:
        y = ys[-1] * nbytes / xs[-1]
    else:
        for i in range(len(xs) - 1):
            if xs[i] <= nbytes <= xs[i + 1]:
                w = (math.log(nbytes) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
                y = math.exp(math.log(ys[i]) + w * (math.log(ys[i + 1]) - math.log(ys[i])))
                break
    return y * cal.collective_scale


# ---------------------------------------------------------------- per-strategy latency
@dataclass
class Result:
    strategy: str
    P: int
    T: int
    B: int
    latency_ms: float
    compute_ms: float
    comm_ms: float
    breakdown: dict[str, float]
    mfu: float
    replicas: int
    throughput_tok_s: float
    throughput_req_s: float
    feasible: bool
    note: str = ""


def gemm_ms(flops: float, weight_bytes: float, T_loc: float, hw: HW, cal: Calib) -> float:
    t_c = flops / (hw.peak_bf16_core * interp_eff(cal.gemm_eff, T_loc))
    t_m = weight_bytes / hw.hbm_bw_core
    return max(t_c, t_m) * 1e3


def gdn_core_ms(heads: float, tokens: float, B: int, cal: Calib) -> float:
    per_nc = cal.gdn_fixed_us + cal.gdn_ns_per_head_token_nc * 1e-3 * heads * tokens * B
    return per_nc / cal.gdn_lnc_speedup * 1e-3


def estimate(strategy: str, P: int, T: int, B: int = 1, m: Model | None = None,
             hw: HW | None = None, cal: Calib | None = None) -> Result:
    m, hw, cal = m or Model(), hw or HW(), cal or Calib()
    Hb = m.dtype_bytes
    note = ""
    bd: dict[str, float] = {}
    comm = 0.0
    tok = T * B  # tokens in the forward (B sequences of length T)

    def layer_common(tok_loc: float, w_shard: float) -> tuple[float, float, float, float]:
        """GEMM + elementwise time for one GDN layer and one attention layer (ms)."""
        gdn_p = gemm_ms(2 * m.gdn_proj_params * tok_loc / w_shard, m.gdn_proj_params * Hb / w_shard, tok_loc, hw, cal)
        att_p = gemm_ms(2 * m.attn_proj_params * tok_loc / w_shard, m.attn_proj_params * Hb / w_shard, tok_loc, hw, cal)
        mlp = gemm_ms(2 * m.mlp_params * tok_loc / w_shard, m.mlp_params * Hb / w_shard, tok_loc, hw, cal)
        elem = cal.elementwise_bytes_per_token_layer * tok_loc / hw.hbm_bw_core * 1e3 + cal.per_layer_overhead_us * 1e-3
        return gdn_p, att_p, mlp, elem

    if strategy in ("DP", "PP"):
        P_eff = 1 if strategy == "DP" else P
        gdn_p, att_p, mlp, elem = layer_common(tok, 1)
        g_core = gdn_core_ms(m.gdn_hv, T, B, cal)
        a_core = m.attn_core_flops(T) * B / (hw.peak_bf16_core * cal.attn_eff) * 1e3
        bd = {"gdn_proj": gdn_p * m.n_gdn, "attn_proj": att_p * m.n_attn, "mlp": mlp * m.n_layers,
              "gdn_core": g_core * m.n_gdn, "attn_core": a_core * m.n_attn, "elementwise": elem * m.n_layers}
        if strategy == "PP":
            # stage boundary send of [T,H] per stage; latency equals DP plus P-1 hops
            hop = coll_us("allgather", tok * m.H * Hb, 2, cal) * 1e-3
            bd["p2p"] = hop * (P - 1)
            comm = bd["p2p"]
            note = "pipeline: same single-request latency as DP (+hops); throughput no better than DP"
        P = P_eff if strategy == "DP" else P
    elif strategy in ("TP", "TPSP"):
        # attention: 8 q heads -> P<=8 ; KV (2 heads) replicated when P>2
        gdn_p, att_p, mlp, elem = layer_common(tok, P)
        # elementwise work: ~80% is head/feature-sharded under TP (GDN prologue, gated norm, conv,
        # activations, attention elementwise); RMSNorm + residual (~20%) is replicated unless SP
        # (TP+SP also shards the norms/residual over tokens, so all of it divides by P)
        elem = elem / P if strategy == "TPSP" else elem * (cal.elem_replicated_frac + (1 - cal.elem_replicated_frac) / P)
        g_core = gdn_core_ms(m.gdn_hv / P, T, B, cal)
        kv_rep = max(1, P / m.nkv)
        a_core = m.attn_core_flops(T) * B / P / (hw.peak_bf16_core * cal.attn_eff) * 1e3
        att_p += (kv_rep - 1) * 0  # replicated KV proj compute is negligible (small)
        msg = tok * m.H * Hb
        if strategy == "TP":
            per = 2 * coll_us("allreduce", msg, P, cal)
        else:
            per = 2 * (coll_us("reducescatter", msg, P, cal) + coll_us("allgather", msg, P, cal))
        comm = per * m.n_layers * 1e-3 * (1 - cal.overlap)
        bd = {"gdn_proj": gdn_p * m.n_gdn, "attn_proj": att_p * m.n_attn, "mlp": mlp * m.n_layers,
              "gdn_core": g_core * m.n_gdn, "attn_core": a_core * m.n_attn, "elementwise": elem * m.n_layers,
              "collectives": comm}
        if P > m.nq:
            note = f"P>{m.nq}: attention heads cannot be split further"
    elif strategy in ("CP", "ULY", "HYB", "TSP"):
        tok_loc = tok / P
        T_loc = T / P
        w_shard = 1
        gdn_p, att_p, mlp, elem = layer_common(tok_loc, w_shard)
        # attention core with zigzag load balance: each rank T/P queries, average ctx T/2
        a_core_cp = m.attn_core_flops(T) * B / P / (hw.peak_bf16_core * cal.attn_eff) * 1e3
        kv_bytes = tok * 2 * m.nkv * m.hd * Hb  # AG output: K and V of all tokens
        ag_kv = coll_us("allgather", kv_bytes, P, cal) * 1e-3
        # GDN two-pass (CP/TSP): local + transition pass on T/P tokens, AG of (S_loc, M) fp32
        state_bytes = 2 * m.gdn_hv * m.gdn_dk * m.gdn_dv * 4 * B * P
        g_core_cp = 2 * gdn_core_ms(m.gdn_hv, T_loc, B, cal)
        g_fix_cp = coll_us("allgather", state_bytes, P, cal) * 1e-3
        corr = gemm_ms(2 * m.gdn_hv * T_loc * B * m.gdn_dk * m.gdn_dv, 0, tok_loc, hw, cal)
        # Ulysses GDN: a2a in (q,k,v,g,beta) and out (core output) -> heads/P over full T
        a2a_in = tok_loc * (2 * m.gdn_hk * m.gdn_dk + m.gdn_hv * m.gdn_dv + 2 * m.gdn_hv) * Hb
        a2a_out = tok_loc * m.gdn_hv * m.gdn_dv * Hb
        g_core_uly = gdn_core_ms(m.gdn_hv / P, T, B, cal)
        g_comm_uly = (coll_us("alltoall", a2a_in, P, cal) + coll_us("alltoall", a2a_out, P, cal)) * 1e-3
        # Ulysses attention: a2a q(+gate stays local) k v in, out back
        att_in = tok_loc * (m.nq * m.hd + 2 * m.nkv * m.hd) * Hb
        att_out = tok_loc * m.nq * m.hd * Hb
        a_comm_uly = (coll_us("alltoall", att_in, P, cal) + coll_us("alltoall", att_out, P, cal)) * 1e-3
        if strategy == "CP":
            g_core, g_comm, a_comm = g_core_cp + corr, g_fix_cp, ag_kv
        elif strategy == "ULY":
            g_core, g_comm, a_comm = g_core_uly, g_comm_uly, a_comm_uly
            if P > m.nq:
                note = f"P>{m.nq}: attention Ulysses degree capped"
        elif strategy == "HYB":
            g_core, g_comm, a_comm = g_core_uly, g_comm_uly, ag_kv
        else:  # TSP: CP data flow + weights moved every layer (no replication)
            g_core, g_comm, a_comm = g_core_cp + corr, g_fix_cp, ag_kv
            w_layer = (m.gdn_proj_params * m.n_gdn + m.attn_proj_params * m.n_attn + m.mlp_params * m.n_layers) * Hb
            # each rank receives (P-1)/P of every layer's weights; ring at ~AG busbw of the group
            w_bytes = w_layer * (P - 1) / P
            w_ms = coll_us("allgather", w_bytes / m.n_layers * P / (P - 1), P, cal) * 1e-3 * m.n_layers
            # overlap of weight traffic with GEMMs (TSP's premise); cap by compute
            gemm_total = gdn_p * m.n_gdn + att_p * m.n_attn + mlp * m.n_layers
            exposed = max(0.0, w_ms - gemm_total * 0.9)
            bd["weight_stream_exposed"] = exposed
            note = f"weights streamed {w_bytes/1e9:.2f} GB/forward/rank ({w_ms:.1f} ms raw)"
        comm_l = (g_comm * m.n_gdn + a_comm * m.n_attn) * (1 - cal.overlap)
        bd.update({"gdn_proj": gdn_p * m.n_gdn, "attn_proj": att_p * m.n_attn, "mlp": mlp * m.n_layers,
                   "gdn_core": g_core * m.n_gdn, "attn_core": a_core_cp * m.n_attn,
                   "elementwise": elem * m.n_layers, "collectives": comm_l})
        comm = comm_l + bd.get("weight_stream_exposed", 0.0)
    else:
        raise ValueError(strategy)

    # pointer head + embedding gather (tiny)
    bd["embed+head"] = (tok * m.H * Hb / hw.hbm_bw_core + 0.02e-3) * 1e3 / max(1, P if strategy not in ("DP", "PP") else 1)
    lat = sum(bd.values())
    compute = lat - comm
    fl = m.flops(T)["total"] * B
    cores = P if strategy != "DP" else 1
    mfu = fl / (lat * 1e-3 * hw.peak_bf16_core * cores)
    replicas = hw.cores // cores
    if strategy == "PP":
        # each stage busy 1/P of the forward: steady-state throughput equals DP on P cores
        mfu = fl / (lat * 1e-3 * hw.peak_bf16_core * cores) * P
    thr_req = replicas * B / (lat * 1e-3) * (P if strategy == "PP" else 1)
    feasible = True
    w_bytes_rank = (m.nonembed_params * Hb) / (P if strategy in ("TP", "TPSP", "TSP") else 1) + m.vocab * m.H * Hb
    if w_bytes_rank > hw.hbm_bytes_core * 0.8:
        feasible = False
    return Result(strategy, P, T, B, lat, compute, comm, bd, mfu, replicas, thr_req * T, thr_req, feasible, note)


STRATS = [("DP", 1), ("TP", 2), ("TP", 4), ("TP", 8), ("TPSP", 2), ("TPSP", 4), ("CP", 2), ("CP", 4),
          ("ULY", 2), ("ULY", 4), ("ULY", 8), ("HYB", 2), ("HYB", 4), ("TSP", 2), ("TSP", 4), ("PP", 2)]


def table(Ts=(128, 256, 512, 1024, 2048, 4096), B=1, cal: Calib | None = None) -> list[Result]:
    return [estimate(s, P, T, B, cal=cal) for T in Ts for s, P in STRATS]


if __name__ == "__main__":
    import sys
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cal = measured_calib() if "--measured" in sys.argv else None
    if cal is not None:
        print(f"fitted: elementwise {cal.elementwise_bytes_per_token_layer/1e3:.0f} KB/token/layer, overhead {cal.per_layer_overhead_us:.0f} us/layer")
        for (T, Bm), v in MEASURED_DP.items():
            print(f"  DP T={T} B={Bm}: measured {v:.1f} ms, model {estimate('DP', 1, T, Bm, cal=cal).latency_ms:.1f} ms")
        for (st, P, T), v in MEASURED_TP.items():
            print(f"  {st}{P} T={T}: measured {v:.1f} ms, model {estimate(st, P, T, 1, cal=cal).latency_ms:.1f} ms (validation, not fitted)")
    m = Model()
    print(f"non-embedding params {m.nonembed_params/1e9:.3f} B; FLOP/token @1024 = {m.flops(1024)['total']/1024/1e9:.2f} GF")
    print(f"{'T':>5} {'strategy':>8} {'P':>2} {'lat_ms':>7} {'comm':>6} {'MFU':>5} {'req/s/inst':>10} note")
    for r in table(B=B, cal=cal):
        print(f"{r.T:5d} {r.strategy:>8} {r.P:2d} {r.latency_ms:7.2f} {r.comm_ms:6.2f} {r.mfu:5.1%} {r.throughput_req_s:10.0f} {r.note}")
