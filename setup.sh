#!/bin/bash

source /usr/share/Modules/init/sh

module load julia-1.0
module load /home/gridsan/groups/llgrid_beta/OpenMPI/3.1.2/openmpi-3.1.2-cuda
module load cuda-latest

export JULIA_DEPOT_PATH="${HOME}/.julia"
julia updatepkg.jl
julia --project=.      -e "using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.API.precompile()"
julia --project=gpuenv -e "using Pkg; Pkg.instantiate()" || true
srun -p gpu -n 1 --gres="gpu:tesla:1" julia --project=gpuenv -e "using Pkg; Pkg.build(); Pkg.API.precompile()"
