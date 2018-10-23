#!/bin/bash
# Nodes have 16 cpus. 
#SBATCH --tasks-per-node=16
#SBATCH --nodes=2

export JULIA_DEPOT_PATH="${HOME}/.julia"
export OPENBLAS_NUM_THREADS=1
mpirun julia --project=. -L src/shallow_water.jl
