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
Concurrent nested Flux jobs must pack onto a Flux allocation, mirroring
``slurm_step_resources_test.py`` for Flux-scheduled systems (El Capitan
family, Tioga, ...).

Run from inside a 1- or 2-node Flux allocation with >= 4 GPUs per node,
e.g.::

    flux alloc -N 1 --exclusive     (or flux alloc -N 2 ...)
    pytest tests/flux_step_resources_test.py -v

The harness and checks are shared with the Slurm variant (see
``step_packing_pytest.py`` / ``verify_step_packing.py``); this file
supplies the Flux policy: nested identity is ``FLUX_JOB_ID`` (there is no
inherited allocation ID to compare), the GPU footprint is read from the
visibility variable Flux sets (``ROCR_VISIBLE_DEVICES`` on AMD,
``CUDA_VISIBLE_DEVICES`` on NVIDIA), and CPU sizing is normalized to
physical cores with tolerance for OS-reserved (specialized) cores.

Workloads (one ``(nodes, gpus_per_task, cores_per_task)`` spec per job,
one task per node; CPU-only job cores are sized to the node -- Tioga's
64-core nodes get 8, El Capitan's 96-core nodes get 16):

- 1 node (7 jobs): 2/1/1-GPU x 8 cores + 4 CPU-only jobs.
- 2 nodes (8 jobs): adds a spanning GPU job (1 GPU + 8 cores per task) and
  a spanning CPU job; worst-case per-node GPU packing is 4, so no job can
  be squeezed out.

Skipped anywhere but inside a 1- or 2-node Flux allocation with >= 4 GPUs
per node.
"""
import os
import re
import subprocess

import pytest

from step_packing_pytest import (
    REPO_ROOT,
    run_packing_test,
    shared_tmp_path,  # noqa: F401  (pytest fixture)
)

RESOURCES_SH = os.path.join(
    REPO_ROOT, "hpc_launcher", "schedulers", "flux_step_resources.sh"
)

GPU_STEP_CORES = 8


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


def _skip_unless_flux_allocation() -> tuple[int, int]:
    """Skip unless usable; return (nodes, cores per node)."""
    if not os.getenv("FLUX_URI"):
        pytest.skip("not inside a Flux allocation")
    nodes, total_cores, total_gpus = _flux_resource_info()
    if nodes not in (1, 2):
        pytest.skip(f"requires a 1- or 2-node allocation (have {nodes})")
    if total_gpus // nodes < 4:
        pytest.skip(
            f"requires >= 4 GPUs per node (have {total_gpus // nodes})"
        )
    return nodes, total_cores // nodes


def _build_specs(nodes: int, node_cores: int):
    """
    The workload: one ``(nodes, gpus_per_task, cores_per_task)`` spec per
    job. CPU-only jobs get as many cores as possible (capped at 16) while
    the per-node worst case -- all single-node jobs plus one task of each
    spanning job -- still fits.
    """
    cpu_jobs = 4
    spanning = 1 if nodes == 2 else 0
    cpu_cores = (node_cores - 3 * GPU_STEP_CORES) // (cpu_jobs + spanning)
    cpu_cores = min(cpu_cores, 16)
    if cpu_cores < 1:
        pytest.skip(f"node too small: {node_cores} cores/node")
    specs = []
    if nodes == 2:
        specs += [(2, 1, GPU_STEP_CORES)]
    specs += [(1, 2, GPU_STEP_CORES), (1, 1, GPU_STEP_CORES)]
    if nodes == 1:
        specs += [(1, 1, GPU_STEP_CORES)]
    if nodes == 2:
        specs += [(2, 0, cpu_cores)]
    specs += [(1, 0, cpu_cores)] * cpu_jobs
    return specs


def test_concurrent_jobs_pack_flux_allocation(shared_tmp_path):
    nodes, node_cores = _skip_unless_flux_allocation()
    run_packing_test(
        shared_tmp_path,
        specs=_build_specs(nodes, node_cores),
        resources_sh=RESOURCES_SH,
        rank_var="FLUX_TASK_RANK",
        launch_flags=["--scheduler", "flux"],
        label="job",
        id_key="FLUX_JOB_ID",
        alloc_key="",
        alloc_value="",
        gpu_mode="visible-env",
        cpu_mode="normalized-range",
    )
