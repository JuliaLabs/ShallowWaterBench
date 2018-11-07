#!/bin/bash
# Nodes have 16 cpus. 
#SBATCH --tasks-per-node=16
#SBATCH --nodes=2

# Initialize Modules
source /usr/share/Modules/init/sh

module load /home/gridsan/groups/llgrid_beta/OpenMPI/3.1.2/openmpi-3.1.2-cuda
module load julia-1.0

export OMPI_MCA_mpi_cuda_support=0
export OMPI_MCA_btl=self,tcp

export JULIA_DEPOT_PATH="${HOME}/.julia"
export OPENBLAS_NUM_THREADS=1
srun --mpi=pmi2 julia --project=. -L shallower_water.jl
