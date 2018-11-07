# ShallowWaterBench

Shallow water benchmarking expirment based of 
https://github.com/climate-machine/Canary.jl/commit/a58ac057a7224f45bcc450ee533937f7bd06fe17

## Setup

```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.API.precompile()"
```

## Running locally

```bash
mpirun -n 4 --oversubscribe julia --project=. original/shallow_water.jl
```

## Running on the supercloud

### Setup

Run `./setup.sh` everytime that you updated a package, on the login node.

### Running normally 

`run.sc.sh` is setup to start 32 mpi ranks per default,
using up two full nodes. It is important that you run the setup before starting
the job and after the `Manifest.toml` got updated since the compute nodes don't
have access to the internet.

```bash
sbatch run.sc.sh
```

### Running the GPU code
`run.sc.gpu.sh` is setup to use one node with 4 GPUs, in order to pull in the
GPU functionality we are using a stacked environment.

```bash
sbatch run.sc.gpu.sh
```

