# ShallowWaterBench

Shallow water benchmarking expirment based of 
https://github.com/climate-machine/Canary.jl/commit/a58ac057a7224f45bcc450ee533937f7bd06fe17

## Setup

```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```

## Running locally

```bash
mpirun -n 4 --oversubscribe julia --project=. src/shallow_water.jl
```

## Running on the supercloud

`run.sc.sh` is setup to start 32 mpi ranks per default, using up two full nodes. It is important that you run the `module load` steps each time and take care to `instantiate` the project, before starting the job since the compute nodes have no access to internet and you would hit the joint cache from all nodes as once.

```bash
module load julia-1.0
module load mpi/openmpi-x86_64

# Run this when you need to update packages
export JULIA_DEPOT_PATH="${HOME}/.julia"
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"

sbatch run.sc.sh
```

### Running the GPU code

Note that we are using stacked environments here
```bash
module load julia-1.0
module load cuda-latest
module load mpi/openmpi-x86_64

# Run this when you need to update packages
export JULIA_DEPOT_PATH="${HOME}/.julia"
julia --project=.      -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
julia --project=gpuenv -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"

sbatch run.sc.gpu.sh
```
