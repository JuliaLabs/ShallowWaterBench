module PartitionedMeshing

using Base.Iterators
using ..Meshing
using OffsetArrays

export PartitionedCartesianMesh, CartesianPartition, CPU, GPU
export ghostboundary, backend, locate, rankindices, translate

import ..Meshing: elemindices, neighbor, overelems, storage

abstract type Backend end
struct CPU <: Backend end
struct GPU <: Backend end

"""
    PartitionedCartesianMesh{B, N, M} <: CartesianMesh{N}

Represents the local region of a [`CartesianMesh`](@ref) `M`.
The local storage is expected to be an `OffsetArray`
"""
struct PartitionedCartesianMesh{B<:Backend, N, M<:CartesianMesh{N}} <: CartesianMesh{N}
    inds :: CartesianIndices{N}
    parent :: M

    function PartitionedCartesianMesh(::B, inds::CartesianIndices{N}, parent::M) where {B, N, M}
        new{B, N, M}(inds, parent)
    end
end

elemindices(mesh::PartitionedCartesianMesh) = mesh.inds 
Base.parentindices(mesh::PartitionedCartesianMesh) = mesh.inds
Base.parent(mesh::PartitionedCartesianMesh) = mesh.parent

neighbor(elem, face, mesh::PartitionedCartesianMesh) = elem + face

backend(::PartitionedCartesianMesh{B}) where B = B()
function overelems(f::F, mesh::PartitionedCartesianMesh{CPU}, args...) where F
    for I in elemindicies(mesh) 
        f(I, mesh, args...)
    end
end

function storage(::Type{T}, mesh::PartitionedCartesianMesh{CPU, N}) where {T, N}
    inds = elemindices(mesh).indices
    inds = ntuple(N) do i 
        I = inds[i]
        (first(I)-1):(last(I)+1)
    end

    underlaying = Array{Int64}(undef, map(length, inds)...)
    return OffsetArray(underlaying, inds)
end

function translate(mesh::PartitionedCartesianMesh{B, N}, boundary) where {B, N}
    pI = axes(elemindices(parent(mesh)))
    b = ntuple(N) do i
        b = boundary.indices[i]
        length(b) > 1 ? b : mod(b[1], pI[i])
    end
    CartesianIndices(b)
end

"""
    ghostboundary(mesh::PartitionedCartesianMesh)

Gives the local indicies that need to be updated
"""
function ghostboundary(mesh::PartitionedCartesianMesh{B, N}) where {B, N}
    fI = first(parentindices(mesh))
    lI = last(parentindices(mesh))

    upper = ntuple(Val(N)) do i
        head, tail = select(fI, lI, i)
        CartesianIndices((head..., fI[i] - 1, tail...))
    end

    lower = ntuple(Val(N)) do i
        head, tail = select(fI, lI, i)
        CartesianIndices((head..., lI[i] + 1, tail...))
    end

    return (upper..., lower...)
end

@inline function select(fI, lI, i)
    head = ntuple(i-1) do j
        fI[j]:lI[j]
    end

    tail = ntuple(length(fI) - i) do j
        fI[j]:lI[j]
    end

    return head, tail
end

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
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

P = CartesianPartition(globalInds, ranks)
inds = rankindices(P, mpirank)

mesh = PartionedCartesianMesh(CPU(), inds, globalMesh)
for elems in ghostboundary(mesh)
    other = locate(P, elems)
    @show other, elems
end
```
"""
struct CartesianPartition{N}
    indices    :: CartesianIndices{N}
    ranks      :: Array{Int,N}                     # ranks[i]==r ⇒ rank r has piece i
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
