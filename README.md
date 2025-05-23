# HPC-launcher Repository

The HPC launcher repository contains a set of helpful scripts and
Python bindings for launching LBANN 2.0 (PyTorch-core) on multiple
leadership-class HPC systems.  There are optimized routines for FLUX,
SLURM, and LSF launchers.  Currently there are supported systems at:
 - LLNL Livermore Computing (LC)
 - LBL NERSC (Pending)
 - ORNL OLCF (Pending)
 - RIKEN (Pending)

## Example Usage

Using the launch command to execute a command in parallel
```
launch -N1 -n1 hostname
```

Using the torchrun-hpc command to execute a PyTorch Python file in parallel on two nodes and four processes per node (8 in total):
```
torchrun-hpc -N2 -n4 file.py [arguments to Python file]
```

Using HPC-Launcher within existing PyTorch code with explicity invoking it from the command line (CLI).  Within the top level Python file, import `hpc_launcher.torch` first to ensure that `torch` is configured per HPC-Launcher's specification.
```
import hpc_launcher.torch
```

# LBANN: Livermore Big Artificial Neural Network Toolkit

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training framework that is
optimized to compose multiple levels of parallelism.

LBANN provides model-parallel acceleration through domain
decomposition to optimize for strong scaling of network training.  It
also allows for composition of model-parallelism with both data
parallelism and ensemble training methods for training large neural
networks with massive amounts of data.  LBANN is able to advantage of
tightly-coupled accelerators, low-latency high-bandwidth networking,
and high-bandwidth parallel file systems.

![Block Diagram of LBANN v2.0](LBANN_2.0_Block_Diagram.png)

## Publications

A list of publications, presentations and posters are shown
[here](https://lbann.readthedocs.io/en/latest/publications.html).

## Reporting issues
Issues, questions, and bugs can be raised on the [Github issue
tracker](https://github.com/LBANN/HPC-launcher/issues).
