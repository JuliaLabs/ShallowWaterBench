#!/bin/bash

time julia --project=. -L original/swe_cuarray.jl -e " @time main()"
time julia --project=. -L shallower_water.jl -e "main(); @time main()"
