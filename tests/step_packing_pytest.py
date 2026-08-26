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
Shared pytest harness for the on-hardware step/job packing tests
(``slurm_step_resources_test.py`` and ``flux_step_resources_test.py``).

Launches the workload through ``hpc_launcher.cli.launch`` exactly like the
manual ``tests/run_{slurm,flux}_step_packing.sh`` wrappers, then hands the
per-task reports to the shared checks in ``verify_step_packing``. Each
test file supplies only its scheduler policy (skip guard, launch flags,
identity keys, GPU/CPU verification modes) and its workload specs.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from verify_step_packing import (
    Failures,
    check_cpus,
    check_cvd_consistency,
    check_gpus,
    check_overlap,
    load_tasks,
    threads_per_core,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]

STEP_SLEEP_S = 30
# Scheduling + script + sleep, with generous slack for a busy scheduler.
STEP_TIMEOUT_S = 240


@pytest.fixture
def shared_tmp_path():
    """
    A temp directory on a SHARED filesystem. pytest's ``tmp_path`` lives
    under ``/tmp``, which is node-local on LC systems -- the wrapper script
    and per-task reports are written/read by tasks on every node of a
    spanning step, so a node-local directory fails with "No such file or
    directory" on the second node. Home is shared; override with
    ``STEP_PACKING_DIR`` for another shared location.
    """
    base = os.environ.get("STEP_PACKING_DIR", os.path.expanduser("~"))
    try:
        path = pathlib.Path(tempfile.mkdtemp(prefix="step_packing.", dir=base))
    except OSError as e:
        # Fixtures run before the test's own skip guards, so a sandbox or
        # CI host without a writable home must skip, not error.
        pytest.skip(f"no writable shared directory at {base}: {e}")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def make_wrapper(tmp_path, resources_sh: str, rank_var: str) -> str:
    """
    The per-task wrapper: writes this task's resource report and
    STEP_START/STEP_END stamps to its own ``step_<i>_task<rank>.report``
    file (stdout of concurrent tasks interleaves, so per-task files keep
    the reports parseable). Args: sleep seconds, step index.
    """
    wrapper = tmp_path / "step_wrapper.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f'r="{tmp_path}/step_${{2}}_task${{{rank_var}:-0}}.report"\n'
        'echo "STEP_START $(date +%s.%N)" > "$r"\n'
        f'"{resources_sh}" >> "$r"\n'
        f'sleep "${{1:-{STEP_SLEEP_S}}}"\n'
        'echo "STEP_END $(date +%s.%N)" >> "$r"\n'
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def launch_specs(wrapper: str, specs, launch_flags):
    """
    Start one ephemeral blocking launch per ``(nodes, gpus_per_task,
    cores_per_task)`` spec, one task per node, all concurrently.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    procs = []
    for i, (s_nodes, s_gpus, s_cores) in enumerate(specs):
        cmd = LAUNCH + list(launch_flags) + [
            "-N", str(s_nodes),
            "-n", "1",
            "--gpus-per-proc", str(s_gpus),
            "-c", str(s_cores),
            wrapper,
            str(STEP_SLEEP_S),
            str(i),
        ]
        procs.append(
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env,
            )
        )
    return procs


def wait_for_launches(procs, label: str):
    """Wait for every launch; fail on non-zero exit or step-retry stderr."""
    deadline = time.monotonic() + STEP_TIMEOUT_S
    for i, proc in enumerate(procs):
        try:
            _, err = proc.communicate(
                timeout=max(1, deadline - time.monotonic())
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            _, err = proc.communicate()
            pytest.fail(
                f"{label} {i} did not finish within {STEP_TIMEOUT_S}s -- "
                f"{label}s are serializing instead of running concurrently."
                f"\nstderr:\n{err}"
            )
        assert proc.returncode == 0, (
            f"{label} {i} failed (rc={proc.returncode}):\n{err}"
        )
        assert "step creation temporarily disabled" not in err, (
            f"{label} {i} had to retry step creation -- the footprints did "
            f"not pack:\n{err}"
        )


def run_packing_test(
    tmp_path,
    *,
    specs,
    resources_sh: str,
    rank_var: str,
    launch_flags,
    label: str,
    id_key: str,
    alloc_key: str,
    alloc_value: str,
    gpu_mode: str,
    cpu_mode: str,
):
    """
    Launch the workload and run every shared packing check, failing the
    test with the full list of violations (not just the first).
    """
    assert os.path.exists(resources_sh), resources_sh
    wrapper = make_wrapper(tmp_path, resources_sh, rank_var)
    procs = launch_specs(wrapper, specs, launch_flags)
    wait_for_launches(procs, label)

    err = Failures()
    tasks, starts, ends = load_tasks(
        str(tmp_path), specs, id_key, alloc_key, alloc_value, label, err
    )
    check_overlap(starts, ends, len(specs), label, err)
    tpc = threads_per_core()
    check_cpus(tasks, specs, cpu_mode, tpc, os.cpu_count() or 1, label, err)
    check_gpus(tasks, specs, gpu_mode, label, err)
    if gpu_mode == "device-open":
        check_cvd_consistency(tasks, specs, label, err)

    assert err.ok, "packing violations:\n" + "\n".join(err.messages)
