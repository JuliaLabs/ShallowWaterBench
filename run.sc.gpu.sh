#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:4        # Number of GPUs per node
#SBATCH --tasks-per-node=4        # Number of tasks per node
#SBATCH --cpus-per-task=4         # CPU cores per MPI process
#SBATCH --nodes=1                 # Number of nodes

SRCDIR=$SLURM_SUBMIT_DIR

# Initialize Modules
source /usr/share/Modules/init/sh

module load /home/gridsan/groups/llgrid_beta/OpenMPI/3.1.2/openmpi-3.1.2-cuda
module load julia-1.0
module load cuda-latest

export OMPI_MCA_btl=self,tcp

export JULIA_DEPOT_PATH="${HOME}/.julia"
export OPENBLAS_NUM_THREADS=1
srun --mpi=pmi2 --accel-bind=gn --gres=gpu:volta:1 julia --project=gpuenv -e "using MPI; MPI.Init(); MPI.finalize_atexit(); comm = MPI.COMM_WORLD; @show MPI.Comm_rank(comm); @show MPI.Comm_size(comm); using CUDAdrv; @show length(CUDAdrv.devices())"
