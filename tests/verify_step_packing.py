# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
"""
Shared verifier for the step/job packing tests.

Consumes the per-task report files written by
``hpc_launcher/schedulers/{slurm,flux}_step_resources.sh`` (named
``step_<i>_task<rank>.report``, bracketed by STEP_START/STEP_END stamps)
and checks the packing contract described in ``step_packing_common.sh``.

Used two ways:
- as a script by the manual ``tests/run_{slurm,flux}_step_packing.sh``
  harness (see ``main()`` for the CLI), and
- as an importable module by the pytest files
  (``{slurm,flux}_step_resources_test.py``), which reuse the parsing
  helpers and per-policy checks on in-memory report text.

Scheduler differences are two policy knobs:
- ``gpu_mode``: ``device-open`` takes ground truth from the report's
  ``/dev/nvidiaN`` open tests (cgroup ConstrainDevices; Slurm) and also
  checks CUDA_VISIBLE_DEVICES consistency where exported; ``visible-env``
  reads ROCR/CUDA/HIP_VISIBLE_DEVICES (Flux).
- ``cpu_mode``: ``exact`` expects exactly cores x threads/core CPUs (Slurm
  allocates whole physical cores with their SMT siblings);
  ``normalized-range`` normalizes hwthreads to physical cores and allows
  up to cores/4 OS-reserved (specialized) cores (Flux on LC nodes).
"""
import argparse
import glob
import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------
def report_value(text: str, key: str) -> str:
    """Value of a ``kv`` report line, '' if absent or ``<unset>``."""
    m = re.search(rf"^\s+{re.escape(key)}\s+(.*)$", text, re.MULTILINE)
    if not m:
        return ""
    value = m.group(1).strip()
    return "" if value == "<unset>" else value


def parse_cpu_list(text: str) -> set[int]:
    """'0-3,56,58-59' -> {0,1,2,3,56,58,59}"""
    cpus: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def cpu_set(text: str) -> set[int]:
    """The task's allowed CPUs (hardware-thread ids) from its report."""
    return parse_cpu_list(report_value(text, "Cpus_allowed_list"))


def openable_gpus(text: str) -> set[str]:
    """
    GPU device numbers the task can open, from the report's device-open
    tests -- ground truth under cgroup ConstrainDevices.
    """
    return {
        m.group(1)
        for m in re.finditer(
            r"^\s+/dev/nvidia(\d+)\s+OPEN OK\s*$", text, re.MULTILINE
        )
    }


def visible_gpus(text: str) -> set[str]:
    """
    The GPUs a task sees via the visibility variable its scheduler sets:
    ROCR_VISIBLE_DEVICES on AMD, CUDA_VISIBLE_DEVICES on NVIDIA
    (HIP_VISIBLE_DEVICES if a wrapper moved it).
    """
    for var in (
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
    ):
        value = report_value(text, var)
        if value:
            return set(value.split(","))
    return set()


def threads_per_core() -> int:
    """Hardware threads per physical core on this node (1 if undetectable)."""
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, universal_newlines=True
        ).stdout
        m = re.search(r"^Thread\(s\) per core:\s*(\d+)", out, re.MULTILINE)
        if m:
            return int(m.group(1))
    except OSError:
        pass
    return 1


def physical_cores(cpus: set[int], tpc: int, nproc: int) -> set[int]:
    """
    Normalize hardware-thread ids to physical-core ids: on LC nodes a
    core's SMT sibling is core id + physical-core count, and the affinity
    plugin may or may not include siblings in a task's mask.
    """
    phys_count = max(1, nproc // tpc)
    return {c % phys_count for c in cpus}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
class Failures:
    def __init__(self):
        self.messages: list[str] = []

    def __call__(self, msg: str):
        self.messages.append(msg)
        print(f"FAIL: {msg}", file=sys.stderr)

    @property
    def ok(self) -> bool:
        return not self.messages


def load_tasks(out_dir, specs, id_key, alloc_key, alloc_value, label, err):
    """
    Load every task report of every step; check report count per step,
    nested identity (id_key consistent within a step, distinct across
    steps, alloc_key == alloc_value when given), and collect lifetimes.

    :return: (tasks, starts, ends) where tasks is [(step, rank, text)].
    """
    tasks, starts, ends = [], [], []
    step_ids: dict[int, set] = {}
    for step, (s_nodes, _, _) in enumerate(specs):
        files = sorted(
            glob.glob(os.path.join(out_dir, f"step_{step}_task*.report"))
        )
        if len(files) != s_nodes:
            err(
                f"{label} {step}: expected {s_nodes} task report(s) "
                f"(1/node), found {len(files)}: {files}"
            )
            continue
        for f in files:
            text = open(f).read()
            rank = int(re.search(r"task(\d+)\.report$", f).group(1))

            if alloc_key:
                got = report_value(text, alloc_key)
                if got != alloc_value:
                    err(
                        f"{label} {step} task {rank}: {alloc_key}="
                        f"'{got or 'none'}' instead of {alloc_value} -- it "
                        f"became a new allocation ({f})"
                    )
            sid = report_value(text, id_key)
            if not sid:
                err(f"{label} {step} task {rank}: no {id_key} ({f})")
            step_ids.setdefault(step, set()).add(sid)

            m_s = re.search(r"STEP_START ([\d.]+)", text)
            m_e = re.search(r"STEP_END ([\d.]+)", text)
            if not (m_s and m_e):
                err(f"{label} {step} task {rank}: missing START/END stamps ({f})")
            else:
                starts.append(float(m_s.group(1)))
                ends.append(float(m_e.group(1)))
            tasks.append((step, rank, text))

    for step, sids in step_ids.items():
        if len(sids) != 1:
            err(f"{label} {step}: its tasks disagree on {id_key}: {sorted(sids)}")
    all_ids = [next(iter(s)) for s in step_ids.values() if len(s) == 1]
    if len(set(all_ids)) != len(all_ids):
        err(f"{label}s share {id_key}s: {sorted(all_ids)}")
    if not step_ids:
        err(f"no task reports found in {out_dir}")
    else:
        print(f"== OK: {len(specs)} {label}s, distinct {id_key}s: {' '.join(all_ids)}")
    return tasks, starts, ends


def check_overlap(starts, ends, count, label, err):
    """All steps alive at once: latest start precedes earliest end."""
    if starts and ends and max(starts) >= min(ends):
        err(
            f"{label}s did not all overlap in time (latest start "
            f"{max(starts)} >= earliest end {min(ends)}): they ran "
            f"(partially) serialized"
        )
    else:
        print(f"== OK: all {count} {label}s were alive simultaneously")


def check_cpus(tasks, specs, cpu_mode, tpc, nproc, label, err):
    """
    Per-task CPU footprints sized to the request (policy: exact vs
    normalized-range) and disjoint between tasks on the SAME node
    (different nodes reuse the same CPU numbering).
    """
    by_host: dict[str, list] = {}
    for step, rank, text in tasks:
        cores = specs[step][2]
        host = report_value(text, "hostname")
        if cpu_mode == "exact":
            cpus = cpu_set(text)
            if len(cpus) != cores * tpc:
                err(
                    f"{label} {step} task {rank} on {host}: {len(cpus)} CPUs "
                    f"({sorted(cpus)}), expected {cores * tpc} "
                    f"({cores} cores x {tpc} threads/core)"
                )
        else:  # normalized-range
            cpus = physical_cores(cpu_set(text), tpc, nproc)
            low = cores - cores // 4  # allow OS-reserved (specialized) cores
            if not low <= len(cpus) <= cores:
                err(
                    f"{label} {step} task {rank} on {host}: {len(cpus)} "
                    f"physical cores ({sorted(cpus)}), expected {low}..{cores}"
                )
        for o_step, o_rank, o_cpus in by_host.get(host, []):
            shared = cpus & o_cpus
            if shared:
                err(
                    f"{label}s {step} and {o_step} share CPUs "
                    f"{sorted(shared)} on {host}"
                )
        by_host.setdefault(host, []).append((step, rank, cpus))
    print(
        f"== checked: CPU footprints sized to the request "
        f"({cpu_mode}) and disjoint per node"
    )


def check_gpus(tasks, specs, gpu_mode, label, err):
    """
    Per-task GPU footprints: exactly the requested count, disjoint between
    tasks on the same node; CPU-only tasks hold/see none. Policy picks the
    source: device-open ground truth (Slurm/cgroup) or the visibility env
    variable (Flux).
    """
    extract = openable_gpus if gpu_mode == "device-open" else visible_gpus
    by_host: dict[str, list] = {}
    print(f"== GPU footprint per task ({gpu_mode}):")
    for step, rank, text in tasks:
        want = specs[step][1]
        host = report_value(text, "hostname")
        gpus = extract(text)
        print(
            f"     {label} {step} task {rank} on {host}: "
            f"{','.join(sorted(gpus)) or 'none'}"
        )
        if len(gpus) != want:
            err(
                f"{label} {step} task {rank} on {host}: holds "
                f"{sorted(gpus) or 'no'} GPU(s), expected {want}"
            )
        for o_step, o_rank, o_gpus in by_host.get(host, []):
            shared = gpus & o_gpus
            if shared:
                err(
                    f"{label}s {step} and {o_step} share GPU(s) "
                    f"{sorted(shared)} on {host}"
                )
        by_host.setdefault(host, []).append((step, rank, gpus))
    print("== checked: GPU footprints match the request and are disjoint per node")


def check_cvd_consistency(tasks, specs, label, err):
    """
    Only meaningful with device-open ground truth: CUDA_VISIBLE_DEVICES,
    where exported, must agree with the openable devices -- spelled either
    as the physical device numbers or as cgroup-local indices 0..n-1.
    Whether it is exported at all is site configuration, so unset passes.
    """
    for step, rank, text in tasks:
        v = report_value(text, "CUDA_VISIBLE_DEVICES")
        if not v:
            continue  # not exported here; cgroup confinement governs
        phys = sorted(int(g) for g in openable_gpus(text))
        if specs[step][1] == 0:
            err(
                f"CPU-only {label} {step} task {rank}: "
                f"CUDA_VISIBLE_DEVICES='{v}' but no GPU should be held"
            )
            continue
        vis = sorted(int(x) for x in v.split(","))
        local_ids = list(range(len(phys)))
        if vis not in (phys, local_ids):
            err(
                f"{label} {step} task {rank}: CUDA_VISIBLE_DEVICES='{v}' "
                f"matches neither the physical device(s) {phys} nor "
                f"cgroup-local numbering {local_ids}"
            )
    print("== checked: CUDA_VISIBLE_DEVICES (where exported) is consistent")


def verify(out_dir, specs, *, label, id_key, alloc_key, alloc_value,
           gpu_mode, cpu_mode, tpc, nproc) -> bool:
    """Run every check; report all failures rather than the first."""
    err = Failures()
    tasks, starts, ends = load_tasks(
        out_dir, specs, id_key, alloc_key, alloc_value, label, err
    )
    check_overlap(starts, ends, len(specs), label, err)
    check_cpus(tasks, specs, cpu_mode, tpc, nproc, label, err)
    check_gpus(tasks, specs, gpu_mode, label, err)
    if gpu_mode == "device-open":
        check_cvd_consistency(tasks, specs, label, err)
    return err.ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default="step")
    parser.add_argument("--id-key", required=True)
    parser.add_argument("--alloc-key", default="")
    parser.add_argument("--alloc-value", default="")
    parser.add_argument(
        "--gpu-mode", choices=("device-open", "visible-env"), required=True
    )
    parser.add_argument(
        "--cpu-mode", choices=("exact", "normalized-range"), required=True
    )
    parser.add_argument("--threads-per-core", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "specs", nargs="+",
        help="one nodes:gpus_per_task:cores_per_task spec per step",
    )
    args = parser.parse_args()

    specs = [tuple(int(x) for x in s.split(":")) for s in args.specs]
    ok = verify(
        args.out_dir, specs,
        label=args.label,
        id_key=args.id_key,
        alloc_key=args.alloc_key,
        alloc_value=args.alloc_value,
        gpu_mode=args.gpu_mode,
        cpu_mode=args.cpu_mode,
        tpc=args.threads_per_core,
        nproc=args.nproc,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
