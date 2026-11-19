"""Per-kernel / per-op-category device time from a neuron-explorer parquet export.
usage: PYTHONPATH=<dir with duckdb> python 06f_attribute.py <parquet_dir>
Category time = union of instruction intervals (any engine) on physical core 0 of the LNC pair."""
import sys, re, duckdb
d = sys.argv[1]
con = duckdb.connect()
total = con.execute(f"select total_time from '{d}/Summary.parquet'").fetchone()[0]
rows = con.execute(f"""select hlo_name, regexp_extract(nki_source_location, '([a-z_0-9]+)\\.py', 1), start_ts, end_ts
                       from '{d}/Instruction.parquet' where pcore_idx = 0 and hlo_name is not null""").fetchall()
def cat(h, src):
    if "custom-call" in h:
        if src in ("gdn_cte_schur", "gdn_cte_utils", "gdn_cte"): return "NKI gdn_cte"
        if src.startswith("gdn_conv1d"): return "NKI gdn_conv1d_prefill"
        if src.startswith("attention_cte"): return "NKI attention_cte"
        return "custom-call (compiler, e.g. softplus/silu lowering)"
    op = re.match(r"%([a-z\-]+)", h)
    op = op.group(1) if op else h
    return {"dot": "dot (matmul)", "transpose": "transpose"}.get(op, "other elementwise/reduce")
from collections import defaultdict
iv = defaultdict(list)
for h, src, s, e in rows:
    iv[cat(h, src or "")].append((s, e))
def union(v):
    v.sort(); tot = 0; cs, ce = v[0]
    for s, e in v[1:]:
        if s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    return tot + ce - cs
print(f"graph total_time = {total*1e3:.2f} ms")
for k, v in sorted(iv.items(), key=lambda kv: -union(kv[1])):
    print(f"  {k:52s} {union(v)/1e3:9.1f} us busy  ({union(v)/1e9/total*100:5.1f}% of wall; categories overlap)")
