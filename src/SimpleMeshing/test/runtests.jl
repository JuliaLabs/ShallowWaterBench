using Test

using SimpleMeshing
using .Meshing
using .Partitions

dim = 2

globalInds = CartesianIndices(ntuple(i-> 1:1024, dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

data = storage(Int64, globalMesh)

overelems(globalMesh, data) do elem, mesh, out
    out[elem] = 1
    return
end

@test all(data .== 1)

overelems(globalMesh, data) do elem, mesh, out
    Fs = faces(elem, mesh)
    n = neighbor(elem, first(Fs), mesh)
    out[n] = 2
    return
end
@test all(data .== 2)

# First zero the data
overelems(globalMesh, data) do elem, mesh, out
    out[elem] = 0 
    return
end
# check that writes end up in the right place
# we need to space them out quite a bit
overelems(globalMesh, data) do elem, mesh, out
    all(i->i%4==0, Tuple(elem)) || return
    Fs = faces(elem, mesh)
    for (i, f) in enumerate(Fs)
        n = neighbor(elem, f, mesh)
        out[n] = i
    end
    out[elem] = -1 
    return
end
mask =  [0  0  0  1;
         0  0  0  0;
         0  0  0  3;
         2  0  4 -1;
]
for i in 1:4:size(data, 1)
    for j in 1:4:size(data, 2)
       @test data[i:(i+3), j:(j+3)] == mask
    end
end

# First zero the data
overelems(globalMesh, data) do elem, mesh, out
    out[elem] = 0 
    return
end
# check that writes end up in the right place
# we need to space them out quite a bit
overelems(globalMesh, data) do elem, mesh, out
    all(i->i%4==0, Tuple(elem)) || return
    for (i, n) in enumerate(neighbors(elem, mesh))
        data[n] = i
    end
    data[elem] = -1 
    return
end
mask =  [0  0  0  1;
         0  0  0  0;
         0  0  0  3;
         2  0  4 -1;
]
for i in 1:4:size(data, 1)
    for j in 1:4:size(data, 2)
       @test data[i:(i+3), j:(j+3)] == mask
    end
end

nranks = 4
sz = nranks ÷ dim
ranks = collect(reshape(0:(nranks-1), ntuple(i->sz, dim)...))
@test ranks == [0 2;
                1 3]

myrank = 2

P = CartesianPartition(globalInds, ranks)
inds = rankindices(P, myrank)

mesh = GhostCartesianMesh(CPU(), inds)
boundaries = Vector{Tuple{Int, CartesianIndices{dim}, CartesianIndices{dim}}}()
for elems in ghostboundary(mesh)
    other_elems = translate(globalMesh, elems)
    other = locate(P, other_elems)
    @test other !== nothing
    push!(boundaries, (other, elems, other_elems))
end

data = Dict(rank => storage(Int64, GhostCartesianMesh(CPU(), rankindices(P, rank))) for rank in ranks)
localdata = data[myrank]

for (other, elems, other_elems) in boundaries
    localdata[elems] = view(data[other], other_elems)
    # XXX: localdata[elems] .= view(data[other], other_elems)
end
