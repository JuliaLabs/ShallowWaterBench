module PartionedMeshing
    using .Meshing

    export PartionedCartesianMesh, CartesianPartition, CPU, GPU
    export ghostboundary, backend, locate, rankindices

    import Meshing: elemindicies

abstract type Backend end
struct CPU <: Backend end
struct GPU <: Backend end

"""
    PartionedCartesianMesh{B, N, M} <: CartesianMesh{N}

Represents the local region of a [`CartesianMesh`](@ref) `M`.
The local storage is expected to be an `OffsetArray`
"""
struct PartionedCartesianMesh{B::Backend, N, M::CartesianMesh{N}} <: CartesianMesh{N}
    inds :: CartesianIndices{N}
    parent :: M
end

Meshing.neighbor(elem, face, mesh::PartionedCartesianMesh) = elem + face
Base.parentindicies(mesh::PartionedCartesianMesh) = mesh.inds
elemindicies(mesh::PartionedCartesianMesh) = CartesianIndices(axes(parentindices(mesh)))

backend(::PartionedCartesianMesh{B}) where B = B()
function Meshing.overelems(f, mesh::PartionedCartesianMesh, args...)
    overelems(backend(mesh), f, mesh, args...)
end

function Meshing.overelems(::CPU, f::F, mesh::PartionedCartesianMesh, args...) where F
    for I in elemindicies(mesh) 
        f(I, mesh, args...)
    end
end

"""
    ghostboundary(mesh::PartionedCartesianMesh)

Gives the non-local indicies that need to be updated
"""
function ghostboundary(mesh::PartitionedCartesianMesh{N}) where N
    # TODO: translate these ranges into the right space we might be periodic
    return ghostboundary(Val(N), first(parentindices(mesh)), last(parentindices(mesh)))
    
end


function ghostboundary(::Val{N}, fI, lI) where N
    upper = ntuple(Val(N)) do i
        head, tail = select(fI, lI, i)
        (head..., fI[i] - 1, tail...)
    end

    lower = ntuple(Val(N)) do i
        head, tail = select(fI, lI, i)
        (head..., lI[i] + 1, tail...)
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
    Partition

Describes a partition over the data.

```julia
mpirank = MPI.Comm_rank(mpicomm)
mpisize = MPI.Comm_size(mpicomm)

dim = 2
@assert mpisize % dim == 0
sz = mpisize ÷ dim
ranks = reshape(0:(mpisize-1), ntuple(i->sz, dim)...)

globalInds = CartesianIndices((1:1024, 1:1024))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds)

P = CartesianPartition(CartesianIndices(cinds, ranks)
inds = rankindices(P, mpirank)

mesh = PartionedCartesianMesh(globalMesh, inds)
for elems in ghostboundary(mesh)
    other = locate(P, elems)
    @show other, elems
end
```
"""
struct CartesianPartition{N}
    indices    :: CartesianIndices{N}
    ranks      :: Array{Int,N}                     # ranks[i]==r ⇒ rank r has piece i
    partitions :: Array{CartesianIndices{N}, {N}}  # the indicies of piece i
    function Partition{N}(inds::CartesianIndices{N}, ranks :: Array{Int,N})
        partitions = partitions(inds, size(ranks))
        new(inds, ranks, partitions)
    end
end

function locate(p::CartesianPartition, I...)
    i = findfirst(P->I in P, p.partitions)
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
    map(CartesianIndices(starts)) do cI 
        I = ntuple(N) do i
            l = min(lI[i], cI[i]+steps[i])
            cI[i]:l
        end
        CartesianIndices(I)
    end
end

end # module
