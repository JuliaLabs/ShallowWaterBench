#!/bin/bash
# Nodes have 16 cpus. 
#SBATCH --tasks-per-node=16
#SBATCH --nodes=2

# Initialize Modules
source /usr/share/Modules/init/sh

module load mpi/openmpi-x86_64
module load julia-1.0

export JULIA_DEPOT_PATH="${HOME}/.julia"
export OPENBLAS_NUM_THREADS=1
# This should be srun or srun --mpi=pmi2
mpirun julia --project=. -L original/shallow_water.jl
