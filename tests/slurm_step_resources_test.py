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
Concurrent nested job steps must pack onto a matrix Slurm allocation.

Run from inside a 1- or 2-node matrix (CTS-2 H100: 112 cores, 4 GPUs per
node) Slurm allocation, e.g.::

    salloc -N 1 --exclusive -p pbatch     (or salloc -N 2 ...)
    pytest tests/slurm_step_resources_test.py -v

The harness and checks are shared with the Flux variant (see
``step_packing_pytest.py`` / ``verify_step_packing.py``); this file
supplies the Slurm policy: nested identity is ``SLURM_STEP_ID`` (all
inside this ``SLURM_JOB_ID``), GPU ground truth comes from the report's
cgroup device-open tests (plus ``CUDA_VISIBLE_DEVICES`` consistency), and
CPU sizing is exact (Slurm allocates whole physical cores, each bringing
its SMT siblings).

Workloads (one ``(nodes, gpus_per_task, cores_per_task)`` spec per step,
one task per node):

- 1 node (7 steps): 2/1/1-GPU x 8 cores + 4 CPU-only x 16 cores.
- 2 nodes (8 steps): adds a spanning GPU step (1 GPU + 8 cores per task)
  and a spanning CPU step (16 cores per task); worst-case per-node GPU
  packing is 4, so no step can be squeezed out.

Skipped anywhere but inside a 1- or 2-node matrix allocation.
"""
import os
import re
import socket

import pytest

from step_packing_pytest import (
    REPO_ROOT,
    run_packing_test,
    shared_tmp_path,  # noqa: F401  (pytest fixture)
)

RESOURCES_SH = os.path.join(
    REPO_ROOT, "hpc_launcher", "schedulers", "slurm_step_resources.sh"
)

SPECS_BY_NODES = {
    1: [(1, 2, 8), (1, 1, 8), (1, 1, 8),
        (1, 0, 16), (1, 0, 16), (1, 0, 16), (1, 0, 16)],
    2: [(2, 1, 8), (1, 2, 8), (1, 1, 8),
        (2, 0, 16),
        (1, 0, 16), (1, 0, 16), (1, 0, 16), (1, 0, 16)],
}


def _skip_unless_matrix_allocation() -> int:
    if not os.getenv("SLURM_JOB_ID"):
        pytest.skip("not inside a Slurm allocation")
    nodes = int(os.getenv("SLURM_JOB_NUM_NODES", "0"))
    if nodes not in SPECS_BY_NODES:
        pytest.skip("requires a 1- or 2-node allocation")
    if not re.sub(r"\d+", "", socket.gethostname()) == "matrix":
        pytest.skip("requires a matrix (CTS-2 H100) node")
    return nodes


def test_concurrent_steps_pack_matrix_allocation(shared_tmp_path):
    nodes = _skip_unless_matrix_allocation()
    run_packing_test(
        shared_tmp_path,
        specs=SPECS_BY_NODES[nodes],
        resources_sh=RESOURCES_SH,
        rank_var="SLURM_PROCID",
        launch_flags=[],
        label="step",
        id_key="SLURM_STEP_ID",
        alloc_key="SLURM_JOB_ID",
        alloc_value=os.environ["SLURM_JOB_ID"],
        gpu_mode="device-open",
        cpu_mode="exact",
    )
