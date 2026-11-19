#!/bin/bash
# Q6: profile compiled graphs. (1) NEFF discovery + in-process NRT capture, (2) offline neuron-explorer
# capture + summary-json + parquet, (3) per-kernel attribution of the 4-layer hybrid graph.
source $HOME/howto/scripts/common_env.sh
export NEURON_RT_VISIBLE_CORES=2 NEURON_RT_ROOT_COMM_ID=127.0.0.1:61236
F="aliasing_analysis|INFO|UserWarning|_manual_seed|guards.py|^\s*$|warnings.warn|PJRT_DEVICE|WARNING"
OUT=/tmp/howto_profile; rm -rf $OUT; mkdir -p $OUT; cd $OUT
python $HOME/howto/scripts/06_profile.py $OUT 2>&1 | grep -v -E "$F" | tail -5
# in-process session: copy the NEFFs next to the NTFF, then summarize the whole session dir
SESS=$(dirname $(find $OUT/inproc -name "*.ntff" | head -1))
find $OUT/inproc/neffs -name "*.neff" -exec cp {} $SESS/ \;
neuron-explorer view --output-format summary-json -d $SESS > $OUT/inproc_summary.json 2>/dev/null
python3 - $OUT/inproc_summary.json <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["device_profile_profile_nc_0_session_0"]
s = s[0] if isinstance(s, list) else s; s = s.get("summary", s); s = s[0] if isinstance(s, list) else s
print("in-process session: total_time %.2f ms over all executions in the window" % (s["total_time"] * 1e3))
PY
summ() {  # $1=neff $2=ntff $3=label
  neuron-explorer view --output-format summary-json -n $1 -s $2 > $OUT/$3.summary.json 2>/dev/null
  python3 - $OUT/$3.summary.json $3 <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); s = next(iter(d.values()))
s = s[0] if isinstance(s, list) else s; s = s.get("summary", s); s = s[0] if isinstance(s, list) else s
print(f"{sys.argv[2]:10s} device total_time {s['total_time']*1e6:8.1f} us  MFU {s['mfu_estimated_percent']*100:5.1f}%  "
      f"MBU {s['mbu_estimated_percent']*100:5.1f}%  PE-active {s['tensor_engine_active_time_percent']*100:5.1f}%")
PY
}
for name in matmul gdn_cte; do
  NEFF=$(cat $OUT/$name.neffpath); mkdir -p $OUT/cap_$name
  neuron-explorer capture -n $NEFF -s $OUT/cap_$name/profile.ntff --profile-nth-exec=2 > /dev/null 2>&1   # writes profile_exec_2.ntff
  summ $NEFF $OUT/cap_$name/profile_exec_2.ntff $name
done
# 4-layer hybrid graph: capture, export parquet (--ingest-only => no UI server), attribute per kernel
export NEURON_LIBTORCH_CACHE_ROOT=/tmp/howto_cold_cache TORCHINDUCTOR_CACHE_DIR=/tmp/howto_cold_cache/inductor
NEFF=$(python $HOME/howto/scripts/06e_find_neff.py 2>/dev/null | grep "^NEFF" | awk '{print $2}')
mkdir -p $OUT/hybrid
neuron-explorer capture -n $NEFF -s $OUT/hybrid/profile.ntff --profile-nth-exec=2 > /dev/null 2>&1
summ $NEFF $OUT/hybrid/profile_exec_2.ntff hybrid4L
neuron-explorer view -n $NEFF -s $OUT/hybrid/profile_exec_2.ntff --output-format parquet --output-file $OUT/hybrid/pq --ingest-only > /dev/null 2>&1
[ -d /tmp/howto_pylib/duckdb ] || pip install --quiet --target /tmp/howto_pylib duckdb > /dev/null 2>&1
PYTHONPATH=/tmp/howto_pylib python3 $HOME/howto/scripts/06f_attribute.py $OUT/hybrid/pq
