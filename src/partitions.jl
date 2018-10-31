module Partitions

using Base.Iterators

export CartesianPartition
export locate, rankindices

"""
    CartesianPartition

Describes a partition over the data.

```julia
mpirank = MPI.Comm_rank(mpicomm)
mpisize = MPI.Comm_size(mpicomm)

dim = 2
@assert mpisize % dim == 0
sz = mpisize ÷ dim
ranks = reshape(0:(mpisize-1), ntuple(i->sz, dim)...)

globalInds = CartesianIndices((1:1024, 1:1024))
globalMesh = PeriodicGhostCartesianMesh(CartesianIndices(globalInds))

P = CartesianPartition(globalInds, ranks)
inds = rankindices(P, mpirank)

mesh = GhostCartesianMesh(CPU(), inds)
for elems in ghostboundary(mesh)
    other_elems = translate(globalMesh, elems)
    other = locate(P, other_elems)
    @test other !== nothing
    @show other, elems, other_elems
end
```
"""
struct CartesianPartition{N}
    indices    :: CartesianIndices{N}
    ranks      :: Array{Int,N}                   # ranks[i]==r ⇒ rank r has piece i
    partitions :: Array{CartesianIndices{N}, N}  # the indicies of piece i

    function CartesianPartition(inds::CartesianIndices{N}, ranks :: Array{Int,N}) where N
        ps = partitions(inds, size(ranks))
        @assert length(unique(ranks)) == length(ranks)
        new{N}(inds, ranks, ps)
    end
end

function locate(p::CartesianPartition, I::CartesianIndices)
    i = findfirst(P->!isempty(intersect(I, P)), p.partitions)
    i === nothing && return nothing
    p.ranks[i]
end

function rankindices(p::CartesianPartition, r)
    i = findfirst(R-> r==R, p.ranks)
    p.partitions[i]
end

function partitions(inds::CartesianIndices{N}, n::NTuple{N,Int}) where N
    fI = first(inds)
    lI = last(inds)

    steps = ceil.(Int, size(inds) ./ n)
    starts = ntuple(N) do i
        fI[i]:steps[i]:lI[i]
    end
    map(product(starts...)) do cI 
        I = ntuple(N) do i
            l = min(lI[i], cI[i]+steps[i]-1)
            cI[i]:l
        end
        CartesianIndices(I)
    end
end

end # module
