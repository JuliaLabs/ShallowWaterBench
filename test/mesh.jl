include(joinpath(@__DIR__, "..", "src", "mesh.jl"))
include(joinpath(@__DIR__, "..", "src", "partionedmesh.jl"))

using .Meshing

const Dim = 2

globalInds = CartesianIndices(ntuple(i-> 1:1024, Dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

storage = Array{Int64}(undef, map(length, axes(globalInds))...)

overelems(globalMesh, storage) do elem, mesh, out
    storage[elem] = 1
end


