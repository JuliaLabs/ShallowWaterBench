#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:4        # Number of GPUs per node
#SBATCH --tasks-per-node=4        # Number of tasks per node
#SBATCH --cpus-per-task=4         # CPU cores per MPI process
#SBATCH --exclusive
#SBATCH --nodes=1                 # Number of nodes

SRCDIR=`dirname $BASH_SOURCE`

export JULIA_DEPOT_PATH="${HOME}/.julia"
export JULIA_LOAD_PATH="@,${SRCDIR}/gpuenv,@stdlib"
export OPENBLAS_NUM_THREADS=1
# this should be srun --accel-bin=gn --gres=gpu:volta:1
mpirun julia --project=. -e "using MPI; MPI.Init(); MPI.finalize_atexit(); comm = MPI.COMM_WORLD; @show MPI.Comm_rank(comm); @show MPI.Comm_size(comm); usign CUDAdrv; @show length(CUDAdrv.devices())"
