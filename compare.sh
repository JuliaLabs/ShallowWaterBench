#!/bin/bash

time julia --project=. -e "import Canary; include(joinpath(dirname(pathof(Canary)), \"..\", \"examples\", \"swe_array.jl\"))"
time julia --project=. shallower_water.jl
