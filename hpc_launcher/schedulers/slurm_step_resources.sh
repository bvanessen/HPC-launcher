#!/usr/bin/env bash
# slurm_step_resources.sh
#
# Report the resources actually visible to a single srun task (job step),
# as opposed to what the enclosing salloc allocation owns.
#
# Usage:
#   salloc -N2 -n8 --gpus-per-task=1 ...
#   srun --label ./slurm_step_resources.sh          # every task reports
#   ONE_PER_NODE=1 srun ./slurm_step_resources.sh   # one report per node
#   VERBOSE=1 srun -n1 ./slurm_step_resources.sh    # + scontrol/cgroup dumps
#
# Compare against the allocation itself by running it without srun:
#   ./slurm_step_resources.sh
#
# The scheduler-agnostic sections (CPU affinity, memory/NUMA, cgroups, GPU
# device access) live in step_resources_common.sh, shared with the Flux
# variant (flux_step_resources.sh).

set -u

# shellcheck source=step_resources_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/step_resources_common.sh"

ONE_PER_NODE=${ONE_PER_NODE:-0}
VERBOSE=${VERBOSE:-0}

# Skip all but the first task on each node if asked.
if [[ "$ONE_PER_NODE" == "1" && "${SLURM_LOCALID:-0}" != "0" ]]; then
  exit 0
fi

# ---------------------------------------------------------------- identity ---
hdr "Identity"
report_common_identity
kv "context"             "$([[ -n ${SLURM_STEP_ID:-} ]] && echo 'inside srun step' || echo 'allocation shell (no step)')"
kv "SLURM_JOB_ID"        "${SLURM_JOB_ID:-}"
kv "SLURM_STEP_ID"       "${SLURM_STEP_ID:-}"
kv "SLURM_NODEID"        "${SLURM_NODEID:-}"
kv "SLURM_PROCID"        "${SLURM_PROCID:-}"
kv "SLURM_LOCALID"       "${SLURM_LOCALID:-}"
kv "SLURM_NTASKS"        "${SLURM_NTASKS:-}"
kv "SLURM_JOB_NODELIST"  "${SLURM_JOB_NODELIST:-}"
kv "SLURM_STEP_NODELIST" "${SLURM_STEP_NODELIST:-}"

# ------------------------------------------------------- generic resources ---
report_cpus SLURM_CPUS_ON_NODE SLURM_CPUS_PER_TASK \
            SLURM_JOB_CPUS_PER_NODE SLURM_CPU_BIND
report_memory_numa SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
report_cgroups
report_gpu_env SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_GPUS SLURM_GPUS_ON_NODE \
               SLURM_GPUS_PER_TASK SLURM_GPUS_PER_NODE SLURM_GPU_BIND
report_gpu_devices
report_gpu_runtime

# ------------------------------------------------------------------ verbose --
if [[ "$VERBOSE" == "1" ]]; then
  hdr "All SLURM_* / SRUN_* environment"
  env | grep -E '^(SLURM|SRUN)' | sort | sed 's/^/  /'

  if have scontrol && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    hdr "scontrol show job $SLURM_JOB_ID"
    scontrol show job "$SLURM_JOB_ID" 2>&1 | sed 's/^/  /'
    if [[ -n "${SLURM_STEP_ID:-}" ]]; then
      hdr "scontrol show step $SLURM_JOB_ID.$SLURM_STEP_ID"
      scontrol show step "$SLURM_JOB_ID.$SLURM_STEP_ID" 2>&1 | sed 's/^/  /'
    fi
  fi
fi

echo
