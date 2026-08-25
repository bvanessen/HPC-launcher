#!/usr/bin/env bash
# run_flux_step_packing.sh
#
# Manual mirror of tests/flux_step_resources_test.py: from inside a 1-node
# Flux allocation with >= 4 GPUs (El Capitan family, Tioga, ...), launch 7
# concurrent nested Flux jobs --
#   1 GPU job:        2 GPUs + 8 cores       (launch -n 1 --gpus-per-proc 2 -c 8)
#   2 GPU jobs:       1 GPU  + 8 cores each  (launch -n 1 --gpus-per-proc 1 -c 8)
#   4 CPU-only jobs:  cores sized to the node (launch -n 1 --gpus-per-proc 0 -c N)
# -- each running hpc_launcher/schedulers/flux_step_resources.sh bracketed
# by timestamps, and then check that:
#   * every launch ran as a job of THIS Flux instance (7 distinct
#     FLUX_JOB_IDs),
#   * all 7 jobs were alive at the same time (Flux queues jobs whose
#     footprints don't fit, so serialization shows up as non-overlapping
#     lifetimes),
#   * CPU affinity sets are sized to the footprint and pairwise disjoint,
#   * the GPU jobs see 2/1/1 pairwise-disjoint GPUs (ROCR_VISIBLE_DEVICES
#     on AMD, CUDA_VISIBLE_DEVICES on NVIDIA) and CPU-only jobs see none.
#
# Usage:
#   flux alloc -N 1 --exclusive
#   ./tests/run_flux_step_packing.sh
#
# Output goes to a temp dir (printed at the end); job reports are kept for
# inspection.

set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESOURCES_SH="$REPO_ROOT/hpc_launcher/schedulers/flux_step_resources.sh"
STEP_SLEEP=${STEP_SLEEP:-30}
PYTHON=${PYTHON:-python3}

# Per-job GPU counts: job i requests GPU_COUNTS[i] GPUs (2+1+1 = 4).
GPU_COUNTS=(2 1 1)
GPU_STEPS=${#GPU_COUNTS[@]}
CPU_STEPS=4
GPU_STEP_CORES=8

fail() { echo "FAIL: $*" >&2; exit 1; }
note() { echo "== $*"; }

[[ -n "${FLUX_URI:-}" ]] || fail "not inside a Flux allocation (FLUX_URI unset)"
info=$(flux resource info 2>/dev/null) || fail "flux resource info failed"
read -r NODES CORES GPUS <<< "$(awk '{print $1, $3, $5}' <<< "$info")"
[[ "$NODES" == "1" ]] || fail "requires a 1-node allocation (have $NODES)"
total_gpus=0; for g in "${GPU_COUNTS[@]}"; do total_gpus=$((total_gpus + g)); done
[[ "$GPUS" -ge "$total_gpus" ]] || fail "requires >= $total_gpus GPUs (have $GPUS)"
[[ -x "$RESOURCES_SH" ]] || chmod +x "$RESOURCES_SH" || fail "cannot make $RESOURCES_SH executable"

# Cores per CPU-only job: as large as possible while all 7 footprints still
# fit (Tioga's 64-core nodes get 4x8, El Capitan's 96-core nodes get 4x16).
CPU_STEP_CORES=$(( (CORES - GPU_STEPS * GPU_STEP_CORES) / CPU_STEPS ))
[[ $CPU_STEP_CORES -gt 16 ]] && CPU_STEP_CORES=16
[[ $CPU_STEP_CORES -ge 1 ]] || fail "node too small: $CORES cores cannot fit the workload"

OUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/flux_step_packing.XXXXXX")
note "Flux instance on $(hostname -s): $NODES node, $CORES cores, $GPUS GPUs; output in $OUT_DIR"
note "CPU-only jobs get $CPU_STEP_CORES cores each"

# Bracket the report with wall-clock stamps so overlap can be checked later.
WRAPPER="$OUT_DIR/step_wrapper.sh"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
echo "STEP_START \$(date +%s.%N)"
"$RESOURCES_SH"
sleep "\${1:-$STEP_SLEEP}"
echo "STEP_END \$(date +%s.%N)"
EOF
chmod +x "$WRAPPER"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------- launch ----
pids=()
note "launching $GPU_STEPS GPU jobs (${GPU_COUNTS[*]} GPUs, $GPU_STEP_CORES cores each)"
for i in $(seq 0 $((GPU_STEPS - 1))); do
  "$PYTHON" -m hpc_launcher.cli.launch \
      --scheduler flux \
      -N 1 -n 1 --gpus-per-proc "${GPU_COUNTS[$i]}" -c "$GPU_STEP_CORES" \
      "$WRAPPER" "$STEP_SLEEP" \
      > "$OUT_DIR/step_$i.out" 2> "$OUT_DIR/step_$i.err" &
  pids+=($!)
done

note "launching $CPU_STEPS CPU-only jobs ($CPU_STEP_CORES cores each)"
for i in $(seq "$GPU_STEPS" $((GPU_STEPS + CPU_STEPS - 1))); do
  "$PYTHON" -m hpc_launcher.cli.launch \
      --scheduler flux \
      -N 1 -n 1 --gpus-per-proc 0 -c "$CPU_STEP_CORES" \
      "$WRAPPER" "$STEP_SLEEP" \
      > "$OUT_DIR/step_$i.out" 2> "$OUT_DIR/step_$i.err" &
  pids+=($!)
done

note "waiting for ${#pids[@]} launches (sleep=${STEP_SLEEP}s each) ..."
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "job $i exited non-zero; stderr:" >&2
    sed 's/^/    /' "$OUT_DIR/step_$i.err" >&2
    rc=1
  fi
done
[[ $rc -eq 0 ]] || fail "one or more launches failed"

# ---------------------------------------------------------------- verify ----
report() { awk -v k="$2" '$1 == k {print $2; exit}' "$1"; }

total=$((GPU_STEPS + CPU_STEPS))
declare -a job_ids starts ends cpu_lists
for i in $(seq 0 $((total - 1))); do
  out="$OUT_DIR/step_$i.out"

  jid=$(report "$out" FLUX_JOB_ID)
  [[ -n "$jid" && "$jid" != "<unset>" ]] || fail "job $i reports no FLUX_JOB_ID -- it did not run as a job of the enclosing Flux instance (see $out)"
  job_ids[$i]=$jid

  starts[$i]=$(awk '/^STEP_START/{print $2; exit}' "$out")
  ends[$i]=$(awk '/^STEP_END/{print $2; exit}' "$out")
  [[ -n "${starts[$i]}" && -n "${ends[$i]}" ]] || fail "job $i is missing STEP_START/STEP_END stamps (see $out)"

  cpu_lists[$i]=$(report "$out" Cpus_allowed_list)
  [[ -n "${cpu_lists[$i]}" ]] || fail "job $i reports no Cpus_allowed_list (see $out)"
done

dup=$(printf '%s\n' "${job_ids[@]}" | sort | uniq -d)
[[ -z "$dup" ]] || fail "duplicate FLUX_JOB_IDs: $dup"
note "OK: $total distinct Flux jobs (ids: ${job_ids[*]})"

# All alive at once: the latest start must precede the earliest end.
"$PYTHON" - "${starts[@]}" -- "${ends[@]}" <<'PY' || fail "jobs did not all overlap in time -- the instance queued them instead of packing (footprints not stated?)"
import sys
sep = sys.argv.index("--")
starts = [float(x) for x in sys.argv[1:sep]]
ends = [float(x) for x in sys.argv[sep + 1:]]
sys.exit(0 if max(starts) < min(ends) else 1)
PY
note "OK: all $total jobs were alive simultaneously"

# Disjoint, correctly sized CPU sets. Two node-level realities to absorb:
#   * SMT: mpibind may or may not include a core's sibling hardware
#     threads in a job's mask, so normalize hwthread ids to physical cores
#     (sibling = core id + physical-core count on these nodes) before
#     counting or comparing.
#   * Core specialization: LC reserves the first core of each 8-core group
#     for the OS/runtime on El Capitan family nodes, so an 8-core request
#     yields 7 usable cores. Sizes are therefore checked as a range
#     [cores - cores/4, cores], not an exact count.
# Disjointness (on physical cores) is the strict check -- that is what
# proves the instance packed the jobs instead of stacking them.
THREADS_PER_CORE=$(lscpu 2>/dev/null | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
THREADS_PER_CORE=${THREADS_PER_CORE:-1}
NPROC_ONLN=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)
"$PYTHON" - "$GPU_STEPS" "$GPU_STEP_CORES" "$CPU_STEP_CORES" "$THREADS_PER_CORE" "$NPROC_ONLN" "${cpu_lists[@]}" <<'PY' || fail "CPU footprints are wrong or overlap (see the step_*.out reports)"
import sys
gpu_steps, gpu_cores, cpu_cores, tpc, nproc = (int(x) for x in sys.argv[1:6])
phys_count = max(1, nproc // tpc)
def parse(text):
    cpus = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return {c % phys_count for c in cpus}   # hwthread -> physical core
sets = [parse(t) for t in sys.argv[6:]]
ok = True
for i, s in enumerate(sets):
    cores = gpu_cores if i < gpu_steps else cpu_cores
    lo = cores - cores // 4   # allow reserved (specialized) cores
    if not lo <= len(s) <= cores:
        print(f"job {i}: {len(s)} physical cores ({sorted(s)}), expected "
              f"{lo}..{cores}", file=sys.stderr)
        ok = False
for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
        shared = sets[i] & sets[j]
        if shared:
            print(f"jobs {i} and {j} share physical cores {sorted(shared)}",
                  file=sys.stderr)
            ok = False
sys.exit(0 if ok else 1)
PY
note "OK: CPU footprints are sized to the request ($GPU_STEP_CORES/$CPU_STEP_CORES cores, minus any OS-reserved cores) and pairwise disjoint"

# GPU visibility: job i must see GPU_COUNTS[i] devices, pairwise disjoint;
# CPU-only jobs must see none. Flux's shell plugin sets the visibility
# variable (ROCR_VISIBLE_DEVICES on AMD, CUDA_VISIBLE_DEVICES on NVIDIA;
# HIP_VISIBLE_DEVICES if a wrapper moved it).
visible_gpus() {
  local out=$1 v
  for var in ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; do
    v=$(report "$out" "$var")
    [[ "$v" == "<unset>" ]] && v=""
    [[ -n "$v" ]] && { tr ',' ' ' <<< "$v"; return; }
  done
  echo ""
}
note "GPU visibility per job:"
gpus=()
for i in $(seq 0 $((GPU_STEPS - 1))); do
  g=$(visible_gpus "$OUT_DIR/step_$i.out")
  g=$(xargs <<< "$g")
  printf '     job %-2s sees: %s\n' "$i" "${g:-none}"
  n=$(wc -w <<< "$g")
  [[ "$n" == "${GPU_COUNTS[$i]}" ]] || fail "GPU job $i sees $n GPU(s) (${g:-none}), expected ${GPU_COUNTS[$i]} (see $OUT_DIR/step_$i.out)"
  gpus+=("$g")
done
all_gpus=$(printf '%s\n' "${gpus[@]}" | tr ' ' '\n' | grep -c .)
uniq_gpus=$(printf '%s\n' "${gpus[@]}" | tr ' ' '\n' | grep . | sort -u | wc -l)
[[ "$all_gpus" == "$uniq_gpus" ]] || fail "GPU jobs share GPU(s): $(printf '[%s] ' "${gpus[@]}")"
note "OK: the $GPU_STEPS GPU jobs see ${GPU_COUNTS[*]} disjoint GPUs: $(printf '[%s] ' "${gpus[@]}")"

for i in $(seq "$GPU_STEPS" $((GPU_STEPS + CPU_STEPS - 1))); do
  g=$(visible_gpus "$OUT_DIR/step_$i.out")
  g=$(xargs <<< "$g")
  [[ -z "$g" ]] || fail "CPU-only job $i sees GPU(s) $g despite --gpus-per-proc 0 (see $OUT_DIR/step_$i.out)"
done
note "OK: the $CPU_STEPS CPU-only jobs see no GPUs"

echo
echo "ALL CHECKS PASSED -- reports kept in $OUT_DIR"
