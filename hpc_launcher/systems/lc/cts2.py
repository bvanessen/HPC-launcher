# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.system import System, SystemParams
import os


# Known LC systems
_system_params = {
    "ipa": (
        "a100",
        {
            "a100": SystemParams(32, 2, "sm_80", 40, 1, "slurm"),
            "aa100": SystemParams(16, 2, "sm_80", 40, 2, "slurm"),
            "av100": SystemParams(32, 2, "sm_70", 32, 2, "slurm"),
            "v100": SystemParams(16, 2, "sm_70", 32, 2, "slurm"),
        },
    ),
}


class CTS2(System):
    """
    LLNL LC Systems based on the Commodity Technology System platform.
    """

    def __init__(self, system_name):
        super().__init__(system_name, _system_params)

    def environment_variables(self) -> list[tuple[str, str]]:
        env_list = []
        env_list.append(("MPICH_OFI_NIC_POLICY", "GPU"))
        env_list.append(("OMP_NUM_THREADS", "21"))
        env_list.append(("OMP_PLACES", "threads"))
        env_list.append(("OMP_PROC_BIND", "spread"))

        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return SlurmScheduler
