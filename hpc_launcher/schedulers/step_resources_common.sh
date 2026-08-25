# step_resources_common.sh
#
# Shared, scheduler-agnostic sections of the step/job resource reports.
# Not executable on its own: sourced by slurm_step_resources.sh and
# flux_step_resources.sh, which add their scheduler's identity and verbose
# sections around these. Everything here reports ground truth the kernel
# exposes (affinity masks, cgroup limits, device-node access), which is the
# same regardless of which scheduler put the process there.

have() { command -v "$1" >/dev/null 2>&1; }
hdr()  { printf '\n=== %s ===\n' "$*"; }
kv()   { printf '  %-30s %s\n' "$1" "${2:-<unset>}"; }
show() { [[ -r "$1" ]] && kv "$(basename "$1")" "$(tr -d '\n' < "$1")"; }

# Can this process actually open the device node? (cgroup device controller
# denies at open() time, so a permission-bit test like [ -r ] is not enough.)
dev_access() {
  local d=$1
  [[ -e "$d" ]] || { echo "absent"; return; }
  if (exec 3<"$d") 2>/dev/null; then echo "OPEN OK"; else echo "DENIED"; fi
}

report_common_identity() {
  kv "hostname"            "$(hostname -s)"
  kv "pid"                 "$$"
  kv "user"                "$(id -un)"
}

# ------------------------------------------------------------------- CPUs ----
# Extra scheduler-specific env vars to report may be passed as arguments.
report_cpus() {
  hdr "CPU affinity (what this task may run on)"
  kv "Cpus_allowed_list"   "$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)"
  kv "allowed CPU count"   "$(nproc 2>/dev/null)"
  kv "CPUs on node (total)" "$(getconf _NPROCESSORS_ONLN 2>/dev/null)"
  local v
  for v in "$@"; do
    kv "$v" "${!v:-}"
  done
  kv "OMP_NUM_THREADS"     "${OMP_NUM_THREADS:-}"
  have taskset  && kv "taskset -cp"    "$(taskset -cp $$ 2>/dev/null | sed 's/.*: //')"
  have hwloc-bind && kv "hwloc-bind --get" "$(hwloc-bind --get 2>/dev/null)"
}

# ------------------------------------------------------------ memory / NUMA --
# Extra scheduler-specific env vars to report may be passed as arguments.
report_memory_numa() {
  hdr "Memory and NUMA"
  kv "Mems_allowed_list"   "$(awk '/Mems_allowed_list/{print $2}' /proc/self/status)"
  local v
  for v in "$@"; do
    kv "$v" "${!v:-}"
  done
  kv "MemTotal (node)"     "$(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  if have numactl; then
    echo "  numactl --show:"
    numactl --show 2>/dev/null | sed 's/^/    /'
  fi
}

# ----------------------------------------------------------------- cgroups ---
report_cgroups() {
  hdr "cgroup limits (how the scheduler actually enforces the above)"
  if grep -q '^0::' /proc/self/cgroup 2>/dev/null; then
    local rel base f
    rel=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)
    base="/sys/fs/cgroup${rel}"
    kv "cgroup version" "v2"
    kv "cgroup path"    "$rel"
    for f in cpuset.cpus.effective cpuset.mems.effective cpu.max \
             memory.max memory.high memory.current pids.max; do
      show "$base/$f"
    done
    # v2 uses eBPF for devices; there is no readable devices.list.
    kv "devices.list" "n/a on cgroup v2 (eBPF) - see device open tests below"
  else
    kv "cgroup version" "v1"
    local ctl rel base
    for ctl in cpuset memory devices; do
      rel=$(awk -F: -v c="$ctl" '$2 ~ ("(^|,)" c "($|,)") {print $3}' /proc/self/cgroup | head -1)
      [[ -n "$rel" ]] || continue
      base="/sys/fs/cgroup/$ctl$rel"
      kv "$ctl cgroup" "$rel"
      case $ctl in
        cpuset) show "$base/cpuset.cpus"; show "$base/cpuset.effective_cpus"
                show "$base/cpuset.mems" ;;
        memory) show "$base/memory.limit_in_bytes"
                show "$base/memory.usage_in_bytes" ;;
        devices) [[ -r "$base/devices.list" ]] && {
                   echo "  devices.list:"; sed 's/^/    /' "$base/devices.list"; } ;;
      esac
    done
  fi
}

# -------------------------------------------------------------------- GPUs ---
# Extra scheduler-specific GPU env vars to report may be passed as arguments;
# the cross-vendor visibility variables are always reported.
report_gpu_env() {
  hdr "GPU allocation (environment)"
  local v
  for v in "$@" \
           CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL \
           ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; do
    kv "$v" "${!v:-}"
  done
}

report_gpu_devices() {
  hdr "GPU device nodes (can this task open them?)"
  local d
  shopt -s nullglob
  for d in /dev/nvidia[0-9]* /dev/nvidiactl /dev/kfd /dev/dri/renderD*; do
    kv "$d" "$(dev_access "$d")"
  done
  shopt -u nullglob
}

report_gpu_runtime() {
  hdr "GPUs as the runtime sees them"
  if have nvidia-smi; then
    nvidia-smi -L 2>&1 | sed 's/^/  /'
  elif have rocm-smi; then
    rocm-smi --showid --showproductname 2>&1 | sed 's/^/  /'
  else
    kv "vendor tool" "neither nvidia-smi nor rocm-smi found"
  fi
  # Runtime-level view: only devices the step is allowed to see should appear.
  if have python3; then
    python3 - <<'PY' 2>/dev/null | sed 's/^/  /'
try:
    import torch
    print("torch.cuda.device_count():", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
except Exception as e:
    pass
PY
  fi
}
