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
from psutil import Process
import inspect

affinity = None
if hasattr(Process, "cpu_affinity") and inspect.isfunction(Process.cpu_affinity):
    # Save affinity before importing torch
    affinity = Process().cpu_affinity()

import torch

if affinity is not None:
    # Restore affinity after importing torch
    Process().cpu_affinity(affinity)
import os

if torch.cuda.is_available():
    fraction_max_gpu_mem = float(os.getenv("TORCHRUN_HPC_MAX_GPU_MEM"))
    if fraction_max_gpu_mem != 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction_max_gpu_mem)
