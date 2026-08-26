#!/usr/bin/env bash
# run_flux_step_packing.sh
#
# Manual mirror of tests/flux_step_resources_test.py: from inside a 1- or
# 2-node Flux allocation with >= 4 GPUs per node (El Capitan family,
# Tioga, ...), launch a mix of concurrent nested Flux jobs and verify the
# instance packs them. The harness, workload contract, and checks live in
# step_packing_common.sh / verify_step_packing.py; this wrapper only
# supplies the Flux policy:
#   * skip guard and workload from FLUX_URI / flux resource info,
#   * nested identity = FLUX_JOB_ID (no inherited allocation ID to check),
#   * GPU footprint from ROCR/CUDA_VISIBLE_DEVICES; CPU sizing normalized
#     to physical cores with tolerance for OS-reserved (specialized) cores.
#
# 1-node workload (7 jobs):  2/1/1-GPU x 8 cores + 4 CPU-only jobs whose
# core count is sized to the node (Tioga 64-core: 8; El Capitan 96: 16).
# 2-node workload (8 jobs):  adds a spanning GPU job (1 GPU + 8 cores per
# task) and a spanning CPU job; worst-case per-node GPU packing is 4, so
# no job can be squeezed out.
#
# Usage:
#   flux alloc -N 1 --exclusive   (or flux alloc -N 2 ...)
#   ./tests/run_flux_step_packing.sh

set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# shellcheck source=step_packing_common.sh
source "$(dirname "$0")/step_packing_common.sh"

[[ -n "${FLUX_URI:-}" ]] || fail "not inside a Flux allocation (FLUX_URI unset)"
info=$(flux resource info 2>/dev/null) || fail "flux resource info failed"
read -r NODES TOTAL_CORES TOTAL_GPUS <<< "$(awk '{print $1, $3, $5}' <<< "$info")"
[[ "$NODES" == "1" || "$NODES" == "2" ]] || fail "requires a 1- or 2-node allocation (have $NODES)"
CORES=$((TOTAL_CORES / NODES))   # per node
GPUS=$((TOTAL_GPUS / NODES))     # per node
[[ "$GPUS" -ge 4 ]] || fail "requires >= 4 GPUs per node (have $GPUS)"
note "Flux instance on $(hostname -s): $NODES node(s), $CORES cores/node, $GPUS GPUs/node"

# Workload: one "nodes:gpus_per_task:cores_per_task" spec per job. CPU-only
# jobs get as many cores as possible while the per-node worst case (all
# singles plus one task of each spanning job) still fits.
GPU_STEP_CORES=8
if [[ "$NODES" == "1" ]]; then
  CPU_STEP_CORES=$(( (CORES - 3 * GPU_STEP_CORES) / 4 ))
  [[ $CPU_STEP_CORES -gt 16 ]] && CPU_STEP_CORES=16
  [[ $CPU_STEP_CORES -ge 1 ]] || fail "node too small: $CORES cores/node cannot fit the workload"
  SPECS=( "1:2:$GPU_STEP_CORES" "1:1:$GPU_STEP_CORES" "1:1:$GPU_STEP_CORES"
          "1:0:$CPU_STEP_CORES" "1:0:$CPU_STEP_CORES" "1:0:$CPU_STEP_CORES" "1:0:$CPU_STEP_CORES" )
else
  CPU_STEP_CORES=$(( (CORES - 3 * GPU_STEP_CORES) / 5 ))
  [[ $CPU_STEP_CORES -gt 16 ]] && CPU_STEP_CORES=16
  [[ $CPU_STEP_CORES -ge 1 ]] || fail "node too small: $CORES cores/node cannot fit the workload"
  SPECS=( "2:1:$GPU_STEP_CORES" "1:2:$GPU_STEP_CORES" "1:1:$GPU_STEP_CORES"
          "2:0:$CPU_STEP_CORES"
          "1:0:$CPU_STEP_CORES" "1:0:$CPU_STEP_CORES" "1:0:$CPU_STEP_CORES" "1:0:$CPU_STEP_CORES" )
fi
note "CPU-only jobs get $CPU_STEP_CORES cores per task"

# Flux policy for the shared harness.
RESOURCES_SH="$REPO_ROOT/hpc_launcher/schedulers/flux_step_resources.sh"
RANK_VAR="FLUX_TASK_RANK"
LAUNCH_FLAGS="--scheduler flux"
VERIFY_ID_KEY="FLUX_JOB_ID"
VERIFY_ALLOC_KEY=""
VERIFY_ALLOC_VALUE=""
GPU_MODE="visible-env"
CPU_MODE="normalized-range"

step_packing_run "job"
