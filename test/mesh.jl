include(joinpath(@__DIR__, "..", "src", "mesh.jl"))
include(joinpath(@__DIR__, "..", "src", "partionedmesh.jl"))

using .Meshing

const Dim = 2

globalInds = CartesianIndices(ntuple(i-> 1:1024, Dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))


