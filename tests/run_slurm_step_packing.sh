#!/usr/bin/env bash
# run_slurm_step_packing.sh
#
# Manual mirror of tests/slurm_step_resources_test.py: from inside a 1-node
# matrix (CTS-2 H100: 112 cores, 4 GPUs) Slurm allocation, launch 7
# concurrent nested job steps --
#   1 GPU step:       2 GPUs + 8 cores       (launch -n 1 --gpus-per-proc 2 -c 8)
#   2 GPU steps:      1 GPU  + 8 cores each  (launch -n 1 --gpus-per-proc 1 -c 8)
#   4 CPU-only steps: 16 cores each          (launch -n 1 --gpus-per-proc 0 -c 16)
# -- each running hpc_launcher/schedulers/slurm_step_resources.sh bracketed
# by timestamps, and then check that:
#   * every launch ran as a step of THIS allocation (same SLURM_JOB_ID,
#     7 distinct SLURM_STEP_IDs),
#   * all 7 steps were alive at the same time (no serialization on
#     "step creation temporarily disabled"),
#   * CPU affinity sets are sized to the footprint (8 vs 16) and disjoint,
#   * the GPU steps hold the expected number of GPUs (2/1/1), pairwise
#     disjoint, covering all 4 GPUs.
#
# Usage:
#   salloc -N 1 --exclusive -p pbatch
#   ./tests/run_slurm_step_packing.sh
#
# Output goes to a temp dir (printed at the end); step reports are kept for
# inspection.

set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESOURCES_SH="$REPO_ROOT/hpc_launcher/schedulers/slurm_step_resources.sh"
STEP_SLEEP=${STEP_SLEEP:-30}
PYTHON=${PYTHON:-python3}

# Per-step GPU counts: step i requests GPU_COUNTS[i] GPUs (2+1+1 = all 4).
GPU_COUNTS=(2 1 1)
GPU_STEPS=${#GPU_COUNTS[@]}
CPU_STEPS=4
GPU_STEP_CORES=8
CPU_STEP_CORES=16

fail() { echo "FAIL: $*" >&2; exit 1; }
note() { echo "== $*"; }

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "not inside a Slurm allocation (SLURM_JOB_ID unset)"
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || fail "requires a 1-node allocation (SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-unset})"
[[ -x "$RESOURCES_SH" ]] || chmod +x "$RESOURCES_SH" || fail "cannot make $RESOURCES_SH executable"

OUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/step_packing.XXXXXX")
note "allocation $SLURM_JOB_ID on $(hostname -s); output in $OUT_DIR"

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
note "launching $GPU_STEPS GPU steps (${GPU_COUNTS[*]} GPUs, $GPU_STEP_CORES cores each)"
for i in $(seq 0 $((GPU_STEPS - 1))); do
  "$PYTHON" -m hpc_launcher.cli.launch \
      -N 1 -n 1 --gpus-per-proc "${GPU_COUNTS[$i]}" -c "$GPU_STEP_CORES" \
      "$WRAPPER" "$STEP_SLEEP" \
      > "$OUT_DIR/step_$i.out" 2> "$OUT_DIR/step_$i.err" &
  pids+=($!)
done

note "launching $CPU_STEPS CPU-only steps ($CPU_STEP_CORES cores each)"
for i in $(seq "$GPU_STEPS" $((GPU_STEPS + CPU_STEPS - 1))); do
  "$PYTHON" -m hpc_launcher.cli.launch \
      -N 1 -n 1 --gpus-per-proc 0 -c "$CPU_STEP_CORES" \
      "$WRAPPER" "$STEP_SLEEP" \
      > "$OUT_DIR/step_$i.out" 2> "$OUT_DIR/step_$i.err" &
  pids+=($!)
done

note "waiting for ${#pids[@]} launches (sleep=${STEP_SLEEP}s each) ..."
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "step $i exited non-zero; stderr:" >&2
    sed 's/^/    /' "$OUT_DIR/step_$i.err" >&2
    rc=1
  fi
done
[[ $rc -eq 0 ]] || fail "one or more launches failed"

# ---------------------------------------------------------------- verify ----
report() { awk -v k="$2" '$1 == k {print $2; exit}' "$1"; }

total=$((GPU_STEPS + CPU_STEPS))
declare -a step_ids starts ends cpu_lists
for i in $(seq 0 $((total - 1))); do
  out="$OUT_DIR/step_$i.out"
  err="$OUT_DIR/step_$i.err"

  if grep -q "step creation temporarily disabled" "$err"; then
    fail "step $i had to retry step creation -- footprints did not pack (see $err)"
  fi

  jid=$(report "$out" SLURM_JOB_ID)
  [[ "$jid" == "$SLURM_JOB_ID" ]] || fail "step $i ran under job '${jid:-none}' instead of allocation $SLURM_JOB_ID -- it became a new allocation (see $out)"

  sid=$(report "$out" SLURM_STEP_ID)
  [[ -n "$sid" && "$sid" != "<unset>" ]] || fail "step $i reports no SLURM_STEP_ID (see $out)"
  step_ids[$i]=$sid

  starts[$i]=$(awk '/^STEP_START/{print $2; exit}' "$out")
  ends[$i]=$(awk '/^STEP_END/{print $2; exit}' "$out")
  [[ -n "${starts[$i]}" && -n "${ends[$i]}" ]] || fail "step $i is missing STEP_START/STEP_END stamps (see $out)"

  cpu_lists[$i]=$(report "$out" Cpus_allowed_list)
  [[ -n "${cpu_lists[$i]}" ]] || fail "step $i reports no Cpus_allowed_list (see $out)"
done

dup=$(printf '%s\n' "${step_ids[@]}" | sort | uniq -d)
[[ -z "$dup" ]] || fail "duplicate SLURM_STEP_IDs: $dup"
note "OK: $total distinct steps of allocation $SLURM_JOB_ID (step ids: ${step_ids[*]})"

# All alive at once: the latest start must precede the earliest end.
"$PYTHON" - "${starts[@]}" -- "${ends[@]}" <<'PY' || fail "steps did not all overlap in time -- they ran (partially) serialized"
import sys
sep = sys.argv.index("--")
starts = [float(x) for x in sys.argv[1:sep]]
ends = [float(x) for x in sys.argv[sep + 1:]]
sys.exit(0 if max(starts) < min(ends) else 1)
PY
note "OK: all $total steps were alive simultaneously"

# Disjoint, correctly sized CPU sets. Slurm allocates whole physical
# cores, so on a hyperthreaded node (matrix: 112 cores / 224 hardware
# threads) a request for N cores shows up as N * threads-per-core entries
# in Cpus_allowed_list -- each core brings its HT sibling(s).
THREADS_PER_CORE=$(lscpu 2>/dev/null | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
THREADS_PER_CORE=${THREADS_PER_CORE:-1}
"$PYTHON" - "$GPU_STEPS" "$GPU_STEP_CORES" "$CPU_STEP_CORES" "$THREADS_PER_CORE" "${cpu_lists[@]}" <<'PY' || fail "CPU footprints are wrong or overlap (see the step_*.out reports)"
import sys
gpu_steps, gpu_cores, cpu_cores, tpc = (int(x) for x in sys.argv[1:5])
def parse(text):
    cpus = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return cpus
sets = [parse(t) for t in sys.argv[5:]]
ok = True
for i, s in enumerate(sets):
    cores = gpu_cores if i < gpu_steps else cpu_cores
    expected = cores * tpc
    if len(s) != expected:
        print(f"step {i}: {len(s)} CPUs ({sorted(s)}), expected {expected} "
              f"({cores} cores x {tpc} threads/core)", file=sys.stderr)
        ok = False
for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
        shared = sets[i] & sets[j]
        if shared:
            print(f"steps {i} and {j} share CPUs {sorted(shared)}", file=sys.stderr)
            ok = False
sys.exit(0 if ok else 1)
PY
note "OK: CPU footprints are sized correctly ($GPU_STEP_CORES/$CPU_STEP_CORES cores x $THREADS_PER_CORE threads/core) and pairwise disjoint"

# GPU confinement. The env vars (SLURM_STEP_GPUS / CUDA_VISIBLE_DEVICES)
# are not reliably exported into the step on all systems, but the report's
# device-open tests are ground truth: with cgroup ConstrainDevices, only
# the GPUs the step actually holds are openable ("OPEN OK"), the rest are
# "DENIED". So check:
#   * GPU step i can open exactly GPU_COUNTS[i] /dev/nvidiaN devices,
#   * no two GPU steps share a device,
#   * each CPU-only step (--gres=none) can open none.
open_gpus() { # device numbers this step can open, space-separated
  awk '$1 ~ /^\/dev\/nvidia[0-9]+$/ && $2 == "OPEN" && $3 == "OK" {
         sub(/^\/dev\/nvidia/, "", $1); printf "%s ", $1
       }' "$1"
}
gpus=()
for i in $(seq 0 $((GPU_STEPS - 1))); do
  g=$(open_gpus "$OUT_DIR/step_$i.out")
  g=${g% }
  n=$(wc -w <<< "$g")
  [[ "$n" == "${GPU_COUNTS[$i]}" ]] || fail "GPU step $i can open $n GPU(s) (${g:-none}), expected ${GPU_COUNTS[$i]} (see $OUT_DIR/step_$i.out)"
  gpus+=("$g")
done
all_gpus=$(printf '%s\n' "${gpus[@]}" | tr ' ' '\n' | grep -c .)
uniq_gpus=$(printf '%s\n' "${gpus[@]}" | tr ' ' '\n' | grep . | sort -u | wc -l)
[[ "$all_gpus" == "$uniq_gpus" ]] || fail "GPU steps share GPU(s): $(printf '[%s] ' "${gpus[@]}")"
note "OK: the $GPU_STEPS GPU steps hold ${GPU_COUNTS[*]} disjoint GPUs: $(printf '[nvidia %s] ' "${gpus[@]}")"

for i in $(seq "$GPU_STEPS" $((GPU_STEPS + CPU_STEPS - 1))); do
  g=$(open_gpus "$OUT_DIR/step_$i.out")
  g=${g% }
  [[ -z "$g" ]] || fail "CPU-only step $i can open GPU(s) $g despite --gres=none (see $OUT_DIR/step_$i.out)"
done
note "OK: the $CPU_STEPS CPU-only steps hold no GPUs"

# Diagnose whether Slurm auto-injects CUDA_VISIBLE_DEVICES into GPU steps
# on this system. Slurm's gres/gpu plugin sets it per step -- UNLESS the
# variable is already set in the environment srun is invoked from, in
# which case Slurm leaves the inherited value untouched. So:
#   * report what the launching shell had (a pre-set value here, even an
#     empty one, explains an unset/stale value inside the steps);
#   * run one bare srun control step (no hpc-launcher involvement) and see
#     whether Slurm sets it there.
note "CUDA_VISIBLE_DEVICES injection diagnostics:"
if [[ ${CUDA_VISIBLE_DEVICES+x} == x ]]; then
  echo "     launching shell: CUDA_VISIBLE_DEVICES='"$CUDA_VISIBLE_DEVICES"' (SET -- Slurm will NOT override an inherited value; unset it before launching)"
else
  echo "     launching shell: CUDA_VISIBLE_DEVICES unset (good: Slurm may inject per step)"
fi
control=$(env -u CUDA_VISIBLE_DEVICES srun -N 1 -n 1 --gpus-per-task=1 -c 1 \
            --exact --mem=0 --overlap \
            /bin/sh -c 'echo "${CUDA_VISIBLE_DEVICES-UNSET}"' 2>/dev/null)
echo "     bare srun control step (launcher not involved): CUDA_VISIBLE_DEVICES='$control'"
if [[ "$control" == "UNSET" ]]; then
  echo "     -> this site's Slurm does not inject CUDA_VISIBLE_DEVICES into steps at all;"
  echo "        cgroup device confinement (verified above) is what isolates the GPUs"
fi

# CUDA_VISIBLE_DEVICES vs the physically openable devices. The device-open
# tests above are ground truth (the cgroup denies open() on every GPU the
# step does not hold, so the CUDA runtime enumerates only the held devices
# regardless of the variable). Whether Slurm additionally exports
# CUDA_VISIBLE_DEVICES into the step is site configuration (gres.conf
# Flags/AutoDetect), so unset is acceptable -- but when it IS set it must
# not contradict the cgroup: valid spellings are the physical device
# number(s) (matching /dev/nvidiaN) or cgroup-local indices 0..n-1.
note "CUDA_VISIBLE_DEVICES vs physically openable devices:"
for i in $(seq 0 $((GPU_STEPS + CPU_STEPS - 1))); do
  v=$(report "$OUT_DIR/step_$i.out" CUDA_VISIBLE_DEVICES)
  [[ "$v" == "<unset>" ]] && v=""
  phys=$(open_gpus "$OUT_DIR/step_$i.out"); phys=${phys% }
  printf '     step %-2s CUDA_VISIBLE_DEVICES=%-10s openable: %s\n' \
         "$i" "'${v}'" "${phys:-none}"

  if [[ -z "$v" ]]; then
    continue  # not exported on this system; cgroup confinement governs
  fi
  if [[ $i -lt $GPU_STEPS ]]; then
    n_vis=$(tr ',' '\n' <<< "$v" | grep -c .)
    n_phys=$(wc -w <<< "$phys")
    [[ "$n_vis" == "$n_phys" ]] || fail "GPU step $i: CUDA_VISIBLE_DEVICES='$v' lists $n_vis device(s) but $n_phys are openable ($phys)"
    sorted_v=$(tr ',' '\n' <<< "$v" | sort -n | xargs)
    local_ids=$(seq 0 $((n_phys - 1)) | xargs)
    if [[ "$sorted_v" != "$phys" && "$sorted_v" != "$local_ids" ]]; then
      fail "GPU step $i: CUDA_VISIBLE_DEVICES='$v' matches neither the physical device(s) ($phys) nor cgroup-local numbering ($local_ids)"
    fi
  else
    fail "CPU-only step $i: CUDA_VISIBLE_DEVICES='$v' but no GPU is openable"
  fi
done
note "OK: CUDA_VISIBLE_DEVICES (where exported) is consistent with the devices each step can open"

echo
echo "ALL CHECKS PASSED -- reports kept in $OUT_DIR"
