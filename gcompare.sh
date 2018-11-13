#!/bin/bash

time julia --project=gpuenv -L original/swe_cuarray.jl -e "@time main()"
time julia --project=gpuenv -L shallower_water.jl -e "main(); @time main()"
