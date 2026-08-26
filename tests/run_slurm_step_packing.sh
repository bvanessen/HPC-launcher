#!/usr/bin/env bash
# run_slurm_step_packing.sh
#
# Manual mirror of tests/slurm_step_resources_test.py: from inside a 1- or
# 2-node matrix (CTS-2 H100: 112 cores, 4 GPUs per node) Slurm allocation,
# launch a mix of concurrent nested job steps and verify they pack. The
# harness, workload contract, and checks live in step_packing_common.sh /
# verify_step_packing.py; this wrapper only supplies the Slurm policy:
#   * skip guard and workload from SLURM_JOB_ID / SLURM_JOB_NUM_NODES,
#   * nested identity = SLURM_STEP_ID, all inside this SLURM_JOB_ID,
#   * GPU ground truth from cgroup device-open tests (+ CUDA_VISIBLE_DEVICES
#     consistency), exact cores x threads/core CPU sizing.
#
# 1-node workload (7 steps):  2/1/1-GPU x 8 cores + 4 CPU-only x 16 cores.
# 2-node workload (8 steps):  adds a spanning GPU step (1 GPU + 8 cores per
# task) and a spanning CPU step (16 cores per task); worst-case per-node
# GPU packing is 4, so no step can be squeezed out.
#
# Usage:
#   salloc -N 1 --exclusive -p pbatch   (or salloc -N 2 ...)
#   ./tests/run_slurm_step_packing.sh

set -u

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# shellcheck source=step_packing_common.sh
source "$(dirname "$0")/step_packing_common.sh"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "not inside a Slurm allocation (SLURM_JOB_ID unset)"
NODES=${SLURM_JOB_NUM_NODES:-0}
[[ "$NODES" == "1" || "$NODES" == "2" ]] || fail "requires a 1- or 2-node allocation (SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-unset})"
note "allocation $SLURM_JOB_ID ($NODES node(s)) on $(hostname -s)"

# Workload: one "nodes:gpus_per_task:cores_per_task" spec per step.
if [[ "$NODES" == "1" ]]; then
  SPECS=( "1:2:8" "1:1:8" "1:1:8"
          "1:0:16" "1:0:16" "1:0:16" "1:0:16" )
else
  SPECS=( "2:1:8" "1:2:8" "1:1:8"
          "2:0:16" "1:0:16" "1:0:16" "1:0:16" "1:0:16" )
fi

# Slurm policy for the shared harness.
RESOURCES_SH="$REPO_ROOT/hpc_launcher/schedulers/slurm_step_resources.sh"
RANK_VAR="SLURM_PROCID"
LAUNCH_FLAGS=""
VERIFY_ID_KEY="SLURM_STEP_ID"
VERIFY_ALLOC_KEY="SLURM_JOB_ID"
VERIFY_ALLOC_VALUE="$SLURM_JOB_ID"
GPU_MODE="device-open"
CPU_MODE="exact"

step_packing_run "step"
