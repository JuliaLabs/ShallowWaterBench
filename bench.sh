INTEL=${HOME}/intel
JULIA=${HOME}/builds/julia/julia
export ENABLE_JITPROFILING=1

source ${INTEL}/vtune_amplifier/amplxe-vars.sh

mpirun -n 4 --oversubscribe amplxe-cl -trace-mpi -collect hotspots --result-dir vtune ${JULIA} --project=. src/shallow_water.jl
