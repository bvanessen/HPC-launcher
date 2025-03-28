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
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.systems.system import System, SystemParams
import os


# Known LC systems
_mi250x_node = SystemParams(64, 8, "gfx90a", 64.0, 4, "flux")
# APUs can run into a snarl where they OOM if too much GPU memory is allocated
_mi300a_node = SystemParams(96, 4, "gfx942", 128.0, 4, "flux", 0.8)
_system_params = {
    "tioga": (
        "pdebug",
        {
            "pdebug": _mi250x_node,
            "mi300a": _mi300a_node,
        },
    ),
    "tuolumne": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "elcap": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "rzadams": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "tenaya": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
}


class ElCapitan(System):
    """
    LLNL LC Systems based on the El Capitan MI300a architecture.
    """

    def __init__(self, system_name):
        super().__init__(system_name, _system_params)

    def environment_variables(self) -> list[tuple[str, str]]:
        env_list = []
        env_list.append(("NCCL_NET_GDR_LEVEL", "3"))  # From HPE to avoid hangs
        env_list.append(
            ("NCCL_MIN_NCHANNELS", "24")
        )  # From AMD to improve collective performance
        env_list.append(("MIOPEN_DEBUG_DISABLE_FIND_DB", "0"))
        env_list.append(("MIOPEN_DISABLE_CACHE", "0"))
        tmpdir = os.environ.get("TMPDIR")
        env_list.append(("MIOPEN_USER_DB_PATH", f"{tmpdir}/MIOpen_user_db"))
        env_list.append(("MIOPEN_CUSTOM_CACHE_DIR", f"{tmpdir}/MIOpen_custom_cache"))

        if os.getenv("CRAY_LD_LIBRARY_PATH") is not None:
            env_list.append(
                (
                    "LD_LIBRARY_PATH",
                    os.getenv("CRAY_LD_LIBRARY_PATH") + ":${LD_LIBRARY_PATH}",
                )
            )
        if os.getenv("ROCM_PATH") is not None:
            env_list.append(
                (
                    "LD_LIBRARY_PATH",
                    os.path.join(os.getenv("ROCM_PATH"), "llvm", "lib")
                    + ":${LD_LIBRARY_PATH}",
                )
            )

        different_ofi_plugin = os.getenv("LBANN_USE_THIS_OFI_PLUGIN")
        if different_ofi_plugin is not None:
            env_list.append(
                ("LD_LIBRARY_PATH", different_ofi_plugin + ":${LD_LIBRARY_PATH}")
            )

        env_list.append(("OMP_NUM_THREADS", "21"))
        env_list.append(("OMP_PLACES", "threads"))
        env_list.append(("OMP_PROC_BIND", "spread"))

        # Performance tuning for HPE Slingshot Cassini NIC
        env_list.append(("FI_CXI_RDZV_PROTO", "alt_read"))
        env_list.append(("FI_CXI_RDZV_THRESHOLD", "0"))
        env_list.append(("FI_CXI_RDZV_GET_MIN", "0"))
        env_list.append(("FI_CXI_RDZV_EAGER_SIZE", "0"))

        # Performance tuning for RCCL multi-threading
        env_list.append(("NCCL_IGNORE_CPU_AFFINITY", "1"))

        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    def customize_scheduler(self, scheduler):
        use_this_rccl = os.getenv("LBANN_USE_THIS_RCCL")
        scheduler.launcher_flags = ["--exclusive"]
        if type(scheduler) is FluxScheduler:
            # Performance tuning for HPE Slingshot Cassini NIC
            scheduler.launcher_flags.append("-ofastload")
            scheduler.launcher_flags.append("--setattr=rdzv_get_en=0")

        if use_this_rccl is not None:
            scheduler.ld_preloads = [f"{use_this_rccl}"]

        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return FluxScheduler
