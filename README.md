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