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
Concurrent nested Flux jobs must pack onto one node, mirroring
``slurm_step_resources_test.py`` for Flux-scheduled systems (El Capitan
family and other LC Flux machines).

Run from inside a 1-node Flux allocation, e.g.::

    flux alloc -N 1 --exclusive
    pytest tests/flux_step_resources_test.py -v

The test launches 7 concurrent ``launch`` invocations inside the
allocation -- 3 GPU jobs with mixed footprints (one 2-GPU job and two
1-GPU jobs, 8 cores each) and 4 CPU-only jobs (``--gpus-per-proc 0``,
cores sized to the node) -- each running
``hpc_launcher/schedulers/flux_step_resources.sh`` to report the resources
the job actually received, bracketed by a sleep so their lifetimes must
overlap.

Unlike Slurm -- where a nested step needs ``--exact``/``--mem=0``/
``--gres=none`` to avoid the step-serialization pitfalls -- ``flux run``
inside an allocation submits to the enclosing Flux instance, whose
scheduler natively packs jobs with stated footprints onto disjoint
resources. What this test pins down is that the launcher's command
construction actually states those footprints (``--cores-per-task``,
``--gpus-per-task``) so that packing happens.

Checks:

- every launch runs as a job of the enclosing Flux instance (distinct
  ``FLUX_JOB_ID``\\ s), not a new allocation;
- all 7 jobs are alive simultaneously (Flux queues, so serialization shows
  up as non-overlapping lifetimes rather than a retry message);
- the jobs' CPU affinity sets are pairwise disjoint and sized to the
  requested footprint;
- the GPU jobs see 2/1/1 pairwise-disjoint GPUs (via the visibility
  variable Flux sets: ``ROCR_VISIBLE_DEVICES`` on AMD,
  ``CUDA_VISIBLE_DEVICES`` on NVIDIA), and CPU-only jobs see none.

Skipped anywhere but inside a 1-node Flux allocation with >= 4 GPUs.
"""
import os
import re
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]
RESOURCES_SH = os.path.join(
    REPO_ROOT, "hpc_launcher", "schedulers", "flux_step_resources.sh"
)

# Per-job GPU counts: job i requests GPU_COUNTS[i] GPUs (2+1+1 = 4).
GPU_COUNTS = (2, 1, 1)
GPU_STEPS = len(GPU_COUNTS)
CPU_STEPS = 4
GPU_STEP_CORES = 8
# CPU-only job cores are sized to the node at runtime (Flux nodes range
# from 64 cores on Tioga to 96+ on El Capitan): see _cpu_step_cores().
STEP_SLEEP_S = 30
STEP_TIMEOUT_S = 240


def _flux_resource_info() -> tuple[int, int, int]:
    """(nodes, cores, gpus) of the enclosing Flux instance, or (0, 0, 0)."""
    try:
        proc = subprocess.run(
            ["flux", "resource", "info"],
            capture_output=True,
            universal_newlines=True,
        )
    except OSError:
        return (0, 0, 0)
    m = re.search(r"^(\d+) Nodes, (\d+) Cores, (\d+) GPUs$", proc.stdout)
    if not m:
        return (0, 0, 0)
    return tuple(int(g) for g in m.groups())


def _skip_unless_flux_allocation() -> tuple[int, int, int]:
    if not os.getenv("FLUX_URI"):
        pytest.skip("not inside a Flux allocation")
    nodes, cores, gpus = _flux_resource_info()
    if nodes != 1:
        pytest.skip(f"requires a 1-node allocation (have {nodes})")
    if gpus < sum(GPU_COUNTS):
        pytest.skip(f"requires >= {sum(GPU_COUNTS)} GPUs (have {gpus})")
    return nodes, cores, gpus


def _cpu_step_cores(node_cores: int) -> int:
    """
    Cores per CPU-only job: as large as possible while all 7 footprints
    still fit (Tioga's 64-core nodes get 4x8, El Capitan's 96-core nodes
    get 4x16).
    """
    remaining = node_cores - GPU_STEPS * GPU_STEP_CORES
    return max(1, min(16, remaining // CPU_STEPS))


def _parse_cpu_list(text: str) -> set[int]:
    """'0-3,56,58-59' -> {0,1,2,3,56,58,59}"""
    cpus: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def _threads_per_core() -> int:
    """Hardware threads per physical core (1 if undetectable)."""
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


def _physical_cores(cpus: set[int]) -> set[int]:
    """
    Normalize hardware-thread ids to physical-core ids: on these nodes a
    core's SMT sibling is core id + physical-core count, and mpibind may or
    may not include siblings in a job's mask.
    """
    tpc = _threads_per_core()
    nproc = os.cpu_count() or 1
    phys_count = max(1, nproc // tpc)
    return {c % phys_count for c in cpus}


def _report_value(output: str, key: str) -> str:
    """
    Value of a ``kv`` line from flux_step_resources.sh, '' if absent/unset.
    """
    m = re.search(rf"^\s+{re.escape(key)}\s+(.*)$", output, re.MULTILINE)
    if not m:
        return ""
    value = m.group(1).strip()
    return "" if value == "<unset>" else value


def _visible_gpus(output: str) -> set[str]:
    """
    The GPUs a Flux job sees, from the visibility variable Flux's shell
    plugin sets: ROCR_VISIBLE_DEVICES on AMD, CUDA_VISIBLE_DEVICES on
    NVIDIA (HIP_VISIBLE_DEVICES if a wrapper moved it).
    """
    for var in (
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
    ):
        value = _report_value(output, var)
        if value:
            return set(value.split(","))
    return set()


def _launch_step(wrapper: str, gpus_per_proc: int, cores: int):
    """
    Start one ephemeral blocking launch (no launch dir, so stdout carries
    the report) as a nested job of the enclosing Flux instance.
    """
    cmd = LAUNCH + [
        "--scheduler", "flux",
        "-N", "1",
        "-n", "1",
        "--gpus-per-proc", str(gpus_per_proc),
        "-c", str(cores),
        wrapper,
        str(STEP_SLEEP_S),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
    )


def test_concurrent_jobs_pack_one_flux_node(tmp_path):
    _, node_cores, _ = _skip_unless_flux_allocation()
    assert os.path.exists(RESOURCES_SH), RESOURCES_SH
    cpu_step_cores = _cpu_step_cores(node_cores)

    # Bracket the resource report with wall-clock stamps so overlapping
    # lifetimes can be asserted from the captured output alone.
    wrapper = tmp_path / "step_wrapper.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'echo "STEP_START $(date +%s.%N)"\n'
        f'"{RESOURCES_SH}"\n'
        'sleep "${1:-30}"\n'
        'echo "STEP_END $(date +%s.%N)"\n'
    )
    wrapper.chmod(0o755)

    procs = []
    for count in GPU_COUNTS:
        procs.append(_launch_step(str(wrapper), count, GPU_STEP_CORES))
    for _ in range(CPU_STEPS):
        procs.append(_launch_step(str(wrapper), 0, cpu_step_cores))

    deadline = time.monotonic() + STEP_TIMEOUT_S
    results = []
    for i, proc in enumerate(procs):
        try:
            out, err = proc.communicate(
                timeout=max(1, deadline - time.monotonic())
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            pytest.fail(
                f"job {i} did not finish within {STEP_TIMEOUT_S}s -- jobs "
                f"are serializing instead of running concurrently.\n"
                f"stderr:\n{err}"
            )
        results.append((proc.returncode, out, err))

    # --- every launch succeeded as a nested job of *this* instance ---------
    job_ids = []
    for i, (rc, out, err) in enumerate(results):
        assert rc == 0, f"job {i} failed (rc={rc}):\n{err}"
        jid = _report_value(out, "FLUX_JOB_ID")
        assert jid, (
            f"job {i} reports no FLUX_JOB_ID -- it did not run as a job of "
            f"the enclosing Flux instance:\n{out}"
        )
        job_ids.append(jid)
    assert len(set(job_ids)) == len(procs), (
        f"expected {len(procs)} distinct Flux jobs, got {job_ids}"
    )

    # --- all jobs were alive at the same time ------------------------------
    starts = [float(re.search(r"STEP_START ([\d.]+)", out).group(1))
              for _, out, _ in results]
    ends = [float(re.search(r"STEP_END ([\d.]+)", out).group(1))
            for _, out, _ in results]
    assert max(starts) < min(ends), (
        f"jobs did not all overlap in time (latest start {max(starts)} >= "
        f"earliest end {min(ends)}): they ran (partially) serialized -- "
        f"footprints were not stated so the instance queued them"
    )

    # --- disjoint, correctly-sized CPU footprints --------------------------
    # Two node-level realities to absorb: SMT (mpibind may or may not
    # include a core's sibling hardware threads, so normalize to physical
    # cores first) and core specialization (LC reserves the first core of
    # each 8-core group on El Capitan family nodes, so an 8-core request
    # yields 7 usable cores -- sizes are a range, not an exact count).
    # Disjointness on physical cores is the strict check: that is what
    # proves the instance packed the jobs instead of stacking them.
    cpu_sets = []
    for i, (_, out, _) in enumerate(results):
        cores = GPU_STEP_CORES if i < GPU_STEPS else cpu_step_cores
        phys = _physical_cores(
            _parse_cpu_list(_report_value(out, "Cpus_allowed_list"))
        )
        low = cores - cores // 4  # allow OS-reserved (specialized) cores
        assert low <= len(phys) <= cores, (
            f"job {i} was given {len(phys)} physical cores "
            f"({sorted(phys)}), expected {low}..{cores}"
        )
        cpu_sets.append((i, phys))
    for i, a in cpu_sets:
        for j, b in cpu_sets:
            if i < j:
                assert not (a & b), (
                    f"jobs {i} and {j} share physical cores {sorted(a & b)} "
                    f"-- the instance did not pack them onto disjoint cores"
                )

    # --- GPU visibility: 2/1/1 disjoint; CPU-only jobs see none ------------
    held = []
    for i, (_, out, _) in enumerate(results[:GPU_STEPS]):
        gpus = _visible_gpus(out)
        assert len(gpus) == GPU_COUNTS[i], (
            f"GPU job {i} sees {sorted(gpus) or 'no'} GPU(s), expected "
            f"{GPU_COUNTS[i]}"
        )
        held.append(gpus)
    for i in range(len(held)):
        for j in range(i + 1, len(held)):
            shared = held[i] & held[j]
            assert not shared, (
                f"GPU jobs {i} and {j} share GPU(s) {sorted(shared)}: "
                f"{[sorted(g) for g in held]}"
            )
    for i, (_, out, _) in enumerate(results[GPU_STEPS:], start=GPU_STEPS):
        gpus = _visible_gpus(out)
        assert not gpus, (
            f"CPU-only job {i} sees GPU(s) {sorted(gpus)} despite "
            f"--gpus-per-proc 0"
        )
