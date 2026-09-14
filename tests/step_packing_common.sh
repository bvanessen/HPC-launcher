# step_packing_common.sh
#
# Shared harness for the manual step/job packing tests. Not executable on
# its own: sourced by run_slurm_step_packing.sh and run_flux_step_packing.sh
# after they set the scheduler policy variables below.
#
# The workload contract (identical for both schedulers): from inside a 1-
# or 2-node allocation, launch a mix of concurrent nested steps/jobs, one
# task per node, each declared as a "nodes:gpus_per_task:cores_per_task"
# spec. Each task runs the scheduler's step-resources report into its own
# file (stdout of concurrent tasks interleaves, so per-task files keep the
# reports parseable), bracketed by STEP_START/STEP_END timestamps. Then
# verify that:
#   * every launch ran nested inside THIS allocation (per-job IDs distinct
#     across steps, consistent across a spanning step's tasks),
#   * all steps were alive at the same time (serialization shows up as
#     non-overlapping lifetimes),
#   * per-task CPU affinity sets are sized to the footprint and disjoint
#     between tasks on the same node,
#   * each task holds/sees exactly the requested number of GPUs, disjoint
#     per node; CPU-only tasks hold/see none.
#
# Policy variables a wrapper must set before sourcing:
#   RESOURCES_SH     absolute path to the per-scheduler report script
#   RANK_VAR         env var holding the task rank inside a step
#                    (SLURM_PROCID / FLUX_TASK_RANK)
#   LAUNCH_FLAGS     extra `launch` arguments (may be empty), e.g.
#                    "--scheduler flux"
#   VERIFY_ID_KEY    report key identifying the nested step/job
#                    (SLURM_STEP_ID / FLUX_JOB_ID)
#   VERIFY_ALLOC_KEY report key that must equal VERIFY_ALLOC_VALUE in every
#                    task, proving the step stayed inside the enclosing
#                    allocation ("" to skip; Flux job IDs are simply
#                    distinct, there is no inherited allocation ID)
#   VERIFY_ALLOC_VALUE  see above
#   GPU_MODE         "device-open": ground truth from the report's
#                    /dev/nvidiaN open tests, plus a CUDA_VISIBLE_DEVICES
#                    consistency check (Slurm + cgroup ConstrainDevices)
#                    "visible-env": ROCR/CUDA/HIP_VISIBLE_DEVICES (Flux)
#   CPU_MODE         "exact": exactly cores x threads/core CPUs (Slurm
#                    allocates whole physical cores with their SMT
#                    siblings)
#                    "normalized-range": normalize hwthreads to physical
#                    cores and allow up to cores/4 OS-reserved
#                    (specialized) cores (Flux on LC nodes)
#
# The wrapper must also set SPECS (the workload array) and may override
# STEP_SLEEP. After sourcing, it calls:
#   step_packing_run  <label>   # "step" or "job", for messages

STEP_SLEEP=${STEP_SLEEP:-30}
PYTHON=${PYTHON:-python3}

fail() { echo "FAIL: $*" >&2; exit 1; }
note() { echo "== $*"; }

step_packing_run() {
  local label=${1:-step}
  local total=${#SPECS[@]}

  [[ -x "$RESOURCES_SH" ]] || chmod +x "$RESOURCES_SH" \
    || fail "cannot make $RESOURCES_SH executable"

  # The work dir must be on a SHARED filesystem: the wrapper script and the
  # per-task reports are written/read by tasks on every node of a spanning
  # step, and TMPDIR//tmp are node-local on LC systems (a wrapper created
  # there is "No such file or directory" on the second node). Home is
  # shared; override with STEP_PACKING_DIR for another shared location.
  local out_dir
  out_dir=$(mktemp -d "${STEP_PACKING_DIR:-$HOME}/step_packing.XXXXXX")
  note "workload: ${SPECS[*]} (nodes:gpus:cores per $label); output in $out_dir"

  # Per-task wrapper: writes this task's report and START/END stamps to its
  # own file. Args: sleep seconds, step index.
  local wrapper="$out_dir/step_wrapper.sh"
  cat > "$wrapper" <<EOF
#!/usr/bin/env bash
r="$out_dir/step_\${2}_task\${${RANK_VAR}:-0}.report"
echo "STEP_START \$(date +%s.%N)" > "\$r"
"$RESOURCES_SH" >> "\$r"
sleep "\${1:-$STEP_SLEEP}"
echo "STEP_END \$(date +%s.%N)" >> "\$r"
EOF
  chmod +x "$wrapper"

  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

  # -------------------------------------------------------------- launch ----
  local pids=() i s_nodes s_gpus s_cores
  for i in $(seq 0 $((total - 1))); do
    IFS=: read -r s_nodes s_gpus s_cores <<< "${SPECS[$i]}"
    note "$label $i: -N $s_nodes, ${s_gpus} GPU(s)/task, ${s_cores} cores/task"
    # shellcheck disable=SC2086  # LAUNCH_FLAGS is deliberately word-split
    "$PYTHON" -m hpc_launcher.cli.launch $LAUNCH_FLAGS \
        -N "$s_nodes" -n 1 --gpus-per-proc "$s_gpus" -c "$s_cores" \
        "$wrapper" "$STEP_SLEEP" "$i" \
        > "$out_dir/step_$i.out" 2> "$out_dir/step_$i.err" &
    pids+=($!)
  done

  note "waiting for ${#pids[@]} launches (sleep=${STEP_SLEEP}s each) ..."
  local rc=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "$label $i exited non-zero; stderr:" >&2
      sed 's/^/    /' "$out_dir/step_$i.err" >&2
      rc=1
    fi
    if grep -q "step creation temporarily disabled" "$out_dir/step_$i.err"; then
      fail "$label $i had to retry step creation -- footprints did not pack (see $out_dir/step_$i.err)"
    fi
  done
  [[ $rc -eq 0 ]] || fail "one or more launches failed"

  # -------------------------------------------------------------- verify ----
  local tpc nproc
  tpc=$(lscpu 2>/dev/null | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
  tpc=${tpc:-1}
  nproc=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)

  "$PYTHON" "$(dirname "${BASH_SOURCE[0]}")/verify_step_packing.py" \
      --out-dir "$out_dir" --label "$label" \
      --id-key "$VERIFY_ID_KEY" \
      --alloc-key "$VERIFY_ALLOC_KEY" --alloc-value "$VERIFY_ALLOC_VALUE" \
      --gpu-mode "$GPU_MODE" --cpu-mode "$CPU_MODE" \
      --threads-per-core "$tpc" --nproc "$nproc" \
      "${SPECS[@]}" \
    || fail "verification failed (reports kept in $out_dir)"

  echo
  echo "ALL CHECKS PASSED -- reports kept in $out_dir"
}
