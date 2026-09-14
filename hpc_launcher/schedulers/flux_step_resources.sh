#!/usr/bin/env bash
# flux_step_resources.sh
#
# Report the resources actually visible to a single flux-run task (a nested
# job of the enclosing Flux instance), as opposed to what the allocation
# owns.
#
# Usage:
#   flux alloc -N2 ...
#   flux run --label-io ./flux_step_resources.sh      # every task reports
#   ONE_PER_NODE=1 flux run ./flux_step_resources.sh  # one report per node
#   VERBOSE=1 flux run -n1 ./flux_step_resources.sh   # + flux job dumps
#
# Compare against the allocation itself by running it without flux run:
#   ./flux_step_resources.sh
#
# The scheduler-agnostic sections (CPU affinity, memory/NUMA, cgroups, GPU
# device access) live in step_resources_common.sh, shared with the Slurm
# variant (slurm_step_resources.sh).

set -u

# shellcheck source=step_resources_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/step_resources_common.sh"

ONE_PER_NODE=${ONE_PER_NODE:-0}
VERBOSE=${VERBOSE:-0}

# Skip all but the first task on each node if asked.
if [[ "$ONE_PER_NODE" == "1" && "${FLUX_TASK_LOCAL_ID:-0}" != "0" ]]; then
  exit 0
fi

# ---------------------------------------------------------------- identity ---
hdr "Identity"
report_common_identity
kv "context"             "$([[ -n ${FLUX_JOB_ID:-} ]] && echo 'inside flux job' || echo 'allocation shell (no job)')"
kv "FLUX_JOB_ID"         "${FLUX_JOB_ID:-}"
kv "FLUX_URI"            "${FLUX_URI:-}"
kv "FLUX_JOB_SIZE"       "${FLUX_JOB_SIZE:-}"
kv "FLUX_JOB_NNODES"     "${FLUX_JOB_NNODES:-}"
kv "FLUX_TASK_RANK"      "${FLUX_TASK_RANK:-}"
kv "FLUX_TASK_LOCAL_ID"  "${FLUX_TASK_LOCAL_ID:-}"

# ------------------------------------------------------- generic resources ---
report_cpus FLUX_JOB_CPUS_PER_TASK
report_memory_numa
report_cgroups
report_gpu_env
report_gpu_devices
report_gpu_runtime

# ------------------------------------------------------------------ verbose --
if [[ "$VERBOSE" == "1" ]]; then
  hdr "All FLUX_* environment"
  env | grep -E '^FLUX' | sort | sed 's/^/  /'

  if have flux && [[ -n "${FLUX_JOB_ID:-}" ]]; then
    hdr "flux job info $FLUX_JOB_ID R (granted resource set)"
    flux job info "$FLUX_JOB_ID" R 2>&1 | sed 's/^/  /'
    hdr "flux job eventlog $FLUX_JOB_ID"
    flux job eventlog "$FLUX_JOB_ID" 2>&1 | sed 's/^/  /'
  fi
fi

echo
