using Test

include(joinpath(@__DIR__, "..", "src", "mesh.jl"))
include(joinpath(@__DIR__, "..", "src", "partionedmesh.jl"))

using .Meshing

const Dim = 2

globalInds = CartesianIndices(ntuple(i-> 1:1024, Dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

storage = Array{Int64}(undef, map(length, axes(globalInds))...)

overelems(globalMesh, storage) do elem, mesh, out
    storage[elem] = 1
    return
end

@test all(storage .== 1)

overelems(globalMesh, storage) do elem, mesh, out
    Fs = faces(elem, mesh)
    n = neighbor(elem, first(Fs), mesh)
    storage[n] = 2
    return
end
@test all(storage .== 2)

# First zero the storage
overelems(globalMesh, storage) do elem, mesh, out
    storage[elem] = 0 
    return
end
# check that writes end up in the right place
# we need to space them out quite a bit
overelems(globalMesh, storage) do elem, mesh, out
    all(i->i%4==0, Tuple(elem)) || return
    Fs = faces(elem, mesh)
    for (i, f) in enumerate(Fs)
        n = neighbor(elem, f, mesh)
        storage[n] = i
    end
    storage[elem] = -1 
    return
end
mask =  [0  0  0  1;
         0  0  0  0;
         0  0  0  3;
         2  0  4 -1;
]
for i in 1:4:size(storage, 1)
    for j in 1:4:size(storage, 2)
       @test storage[i:(i+3), j:(j+3)] == mask
    end
end

# First zero the storage
overelems(globalMesh, storage) do elem, mesh, out
    storage[elem] = 0 
    return
end
# check that writes end up in the right place
# we need to space them out quite a bit
overelems(globalMesh, storage) do elem, mesh, out
    all(i->i%4==0, Tuple(elem)) || return
    for (i, n) in enumerate(neighbors(elem, mesh))
        storage[n] = i
    end
    storage[elem] = -1 
    return
end
mask =  [0  0  0  1;
         0  0  0  0;
         0  0  0  3;
         2  0  4 -1;
]
for i in 1:4:size(storage, 1)
    for j in 1:4:size(storage, 2)
       @test storage[i:(i+3), j:(j+3)] == mask
    end
end