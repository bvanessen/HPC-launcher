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
Seven concurrent nested job steps must pack onto one matrix node.

Run from inside a 1-node matrix (CTS-2 H100: 112 cores, 4 GPUs) Slurm
allocation, e.g.::

    salloc -N 1 --exclusive -p pbatch
    pytest tests/slurm_step_resources_test.py -v

The test launches 7 concurrent ``launch`` invocations inside the
allocation -- 3 GPU steps with mixed footprints (one 2-GPU step and two
1-GPU steps, 8 cores each) and 4 CPU-only steps (16 cores each;
``--gpus-per-proc 0``), 88 of the 112 cores and all 4 GPUs in total --
each running ``hpc_launcher/schedulers/slurm_step_resources.sh`` to report
the resources the step actually received, bracketed by a sleep so their
lifetimes must overlap.

It verifies the whole nested-job-step feature end to end:

- every launch runs as a step of the enclosing allocation (same
  ``SLURM_JOB_ID``, 7 distinct ``SLURM_STEP_ID``\\ s), not a new allocation;
- all 7 steps are alive simultaneously (no serialization on "step creation
  temporarily disabled, retrying"), which is what ``--exact`` +
  per-step footprints (``-c``/``--gpus-per-proc``) buy;
- the steps' CPU affinity sets are pairwise disjoint and sized to the
  requested footprint (8 vs. 16), and the GPU steps hold 2/1/1 pairwise
  disjoint GPUs (a multi-GPU footprint packs alongside single-GPU ones).

Skipped anywhere but inside a 1-node matrix allocation.
"""
import os
import re
import socket
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]
RESOURCES_SH = os.path.join(
    REPO_ROOT, "hpc_launcher", "schedulers", "slurm_step_resources.sh"
)

# Per-step footprints: GPU step i requests GPU_COUNTS[i] GPUs (2+1+1 =
# all 4) + 8 cores; plus 4 x (0 GPUs + 16 cores). 88 of 112 cores total.
GPU_COUNTS = (2, 1, 1)
GPU_STEPS = len(GPU_COUNTS)
CPU_STEPS = 4
GPU_STEP_CORES = 8
CPU_STEP_CORES = 16
# Long enough that unless all 8 steps run concurrently, a late starter's
# window cannot overlap an early finisher's (steps start within seconds of
# each other when nothing blocks).
STEP_SLEEP_S = 30
# Scheduling + script + sleep, with generous slack for a busy slurmctld.
STEP_TIMEOUT_S = 240


def _skip_unless_matrix_allocation():
    if not os.getenv("SLURM_JOB_ID"):
        pytest.skip("not inside a Slurm allocation")
    if os.getenv("SLURM_JOB_NUM_NODES") != "1":
        pytest.skip("requires a 1-node allocation")
    if not re.sub(r"\d+", "", socket.gethostname()) == "matrix":
        pytest.skip("requires a matrix (CTS-2 H100) node")


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
    """
    Hardware threads per physical core on this node (1 if undetectable).

    Slurm allocates whole physical cores, so on a hyperthreaded node
    (matrix: 112 cores / 224 hardware threads) a step that asked for N
    cores legitimately reports N * threads-per-core entries in
    ``Cpus_allowed_list`` -- each core brings its HT sibling(s).
    """
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


def _report_value(output: str, key: str) -> str:
    """
    Value of a ``kv`` line from slurm_step_resources.sh, '' if absent/unset.
    """
    m = re.search(rf"^\s+{re.escape(key)}\s+(.*)$", output, re.MULTILINE)
    if not m:
        return ""
    value = m.group(1).strip()
    return "" if value == "<unset>" else value


def _launch_step(wrapper: str, gpus_per_proc: int, cores: int):
    """
    Start one ephemeral blocking launch (no launch dir, so stdout carries the
    report) as a nested step of the enclosing allocation.
    """
    cmd = LAUNCH + [
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


def test_concurrent_steps_pack_one_matrix_node(tmp_path):
    _skip_unless_matrix_allocation()
    assert os.path.exists(RESOURCES_SH), RESOURCES_SH

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
        procs.append(_launch_step(str(wrapper), 0, CPU_STEP_CORES))

    deadline = time.monotonic() + STEP_TIMEOUT_S
    results = []
    for i, proc in enumerate(procs):
        try:
            out, err = proc.communicate(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            pytest.fail(
                f"step {i} did not finish within {STEP_TIMEOUT_S}s -- steps "
                f"are serializing instead of running concurrently.\n"
                f"stderr:\n{err}"
            )
        results.append((proc.returncode, out, err))

    # --- every launch succeeded as a step of *this* allocation -------------
    job_id = os.environ["SLURM_JOB_ID"]
    step_ids = []
    for i, (rc, out, err) in enumerate(results):
        assert rc == 0, f"step {i} failed (rc={rc}):\n{err}"
        assert "step creation temporarily disabled" not in err, (
            f"step {i} had to retry step creation -- the footprints did not "
            f"pack:\n{err}"
        )
        assert _report_value(out, "SLURM_JOB_ID") == job_id, (
            f"step {i} ran under a different job than the enclosing "
            f"allocation {job_id} -- it was launched as a new allocation:\n"
            f"{out}"
        )
        sid = _report_value(out, "SLURM_STEP_ID")
        assert sid, f"step {i} reports no SLURM_STEP_ID (not inside srun?)"
        step_ids.append(sid)
    assert len(set(step_ids)) == len(procs), (
        f"expected {len(procs)} distinct job steps, got {step_ids}"
    )

    # --- all steps were alive at the same time ------------------------------
    starts = [float(re.search(r"STEP_START ([\d.]+)", out).group(1))
              for _, out, _ in results]
    ends = [float(re.search(r"STEP_END ([\d.]+)", out).group(1))
            for _, out, _ in results]
    assert max(starts) < min(ends), (
        f"steps did not all overlap in time (latest start {max(starts)} >= "
        f"earliest end {min(ends)}): they ran (partially) serialized"
    )

    # --- disjoint, correctly-sized CPU footprints ---------------------------
    tpc = _threads_per_core()
    cpu_sets = []
    for i, (_, out, _) in enumerate(results):
        cores = GPU_STEP_CORES if i < GPU_STEPS else CPU_STEP_CORES
        expected = cores * tpc
        cpus = _parse_cpu_list(_report_value(out, "Cpus_allowed_list"))
        assert len(cpus) == expected, (
            f"step {i} was given {len(cpus)} CPUs ({sorted(cpus)}), "
            f"expected {expected} ({cores} cores x {tpc} threads/core)"
        )
        cpu_sets.append((i, cpus))
    for i, a in cpu_sets:
        for j, b in cpu_sets:
            if i < j:
                assert not (a & b), (
                    f"steps {i} and {j} share CPUs {sorted(a & b)} -- "
                    f"--exact did not give them disjoint footprints"
                )

    # --- GPU confinement, judged by what each step can actually open -------
    # The env vars (SLURM_STEP_GPUS / CUDA_VISIBLE_DEVICES) are not reliably
    # exported into the step on all systems, but the report's device-open
    # tests are ground truth: with cgroup ConstrainDevices only the GPUs the
    # step holds are openable ("OPEN OK"), the rest are "DENIED".
    def openable_gpus(output: str) -> set[int]:
        return {
            int(m.group(1))
            for m in re.finditer(
                r"^\s+/dev/nvidia(\d+)\s+OPEN OK\s*$", output, re.MULTILINE
            )
        }

    held = []
    for i, (_, out, _) in enumerate(results[:GPU_STEPS]):
        gpus = openable_gpus(out)
        assert len(gpus) == GPU_COUNTS[i], (
            f"GPU step {i} can open {sorted(gpus) or 'no'} GPU device(s), "
            f"expected {GPU_COUNTS[i]} -- not confined to its footprint"
        )
        held.append(gpus)
    for i in range(len(held)):
        for j in range(i + 1, len(held)):
            shared = held[i] & held[j]
            assert not shared, (
                f"GPU steps {i} and {j} share GPU(s) {sorted(shared)}: "
                f"{[sorted(g) for g in held]}"
            )
    for i, (_, out, _) in enumerate(results[GPU_STEPS:], start=GPU_STEPS):
        gpus = openable_gpus(out)
        assert not gpus, (
            f"CPU-only step {i} can open GPU(s) {sorted(gpus)} despite "
            f"--gres=none"
        )

    # --- CUDA_VISIBLE_DEVICES agrees with the openable devices -------------
    # The device-open tests are ground truth: the cgroup denies open() on
    # every GPU the step does not hold, so the CUDA runtime enumerates only
    # the held devices regardless of the variable. Whether Slurm exports
    # CUDA_VISIBLE_DEVICES into the step is site configuration (gres.conf
    # Flags/AutoDetect), so unset is acceptable -- but when it IS set it
    # must not contradict the cgroup: valid spellings are the physical
    # device number(s) (matching /dev/nvidiaN) or cgroup-local indices
    # 0..n-1.
    for i, (_, out, _) in enumerate(results):
        visible = _report_value(out, "CUDA_VISIBLE_DEVICES")
        if not visible:
            continue  # not exported here; cgroup confinement governs
        phys = sorted(openable_gpus(out))
        if i < GPU_STEPS:
            vis = sorted(int(x) for x in visible.split(","))
            local_ids = list(range(len(phys)))
            assert vis in (phys, local_ids), (
                f"GPU step {i}: CUDA_VISIBLE_DEVICES={visible!r} matches "
                f"neither the physical device(s) {phys} nor cgroup-local "
                f"numbering {local_ids}"
            )
        else:
            pytest.fail(
                f"CPU-only step {i}: CUDA_VISIBLE_DEVICES={visible!r} but "
                f"no GPU is openable"
            )
