using Base.Cartesian

# N is spatial dimension
# A is data type
abstract type Mesh{N} end

abstract type CartesianMesh{N} end

struct PeriodicCartesianMesh{N} <: CartesianMesh{N}
    inds::CartesianIndices{N}
end

struct CartesianNeighbors{N} <: AbstractVector{CartesianIndex{N}} end

Base.size(::CartesianNeighbors{N}) where {N} = (2N,)

Base.axes(::CartesianNeighbors{N}) where {N} = (1:2N,)

Base.getindex(::CartesianNeighbors{N}, i::Int) where {N} = CartesianIndex(ntuple(n -> i ==     n ?  1 :
                                                                                      i == n + N ? -1 :
                                                                                                    0 , N))

faces(cell, mesh::PeriodicCartesianMesh{N}) where {N} = CartesianNeighbors{N}()

Base.mod(x::T, y::AbstractUnitRange{T}) where {T<:Integer} = y[mod1(x - y[1] + 1, length(y))]
Base.mod(x::CartesianIndex{N}, y::CartesianIndices{N}) where {N} = CartesianIndex(ntuple(n->mod(x[n], axes(y)[n]), N))

neighbor(cell, face, mesh::PeriodicCartesianMesh) =
    (cell + face) in mesh.inds ? cell + face : mod(cell + face, mesh.inds)

opposite(face, mesh::PeriodicCartesianMesh) = face * -1

neighbors(cell, mesh::Mesh) = map(face -> neighbor(cell, face, mesh), faces(cell, mesh))

"""
    PartionedCartesianMesh

Represents the local region of a CartesianMesh. 
"""
struct PartionedCartesianMesh{N} <: CartesianMesh{N}
    inds :: CartesianIndices{N}
end
faces(cell, mesh::PartionedCartesianMesh{N}) where {N} = CartesianNeighbors{N}()
opposite(face, mesh::PartionedCartesianMesh) = face * -1
neighbor(cell, face, mesh::PartionedCartesianMesh) = cell + face

parentindicies(mesh::PartionedCartesianMesh) = mesh.inds

##
# TODO:
# 1. We probably need to translate from global indicies (parentindicies) to local indicies
#    of a storage struct.
##

"""
    ghostboundary(mesh::PartionedCartesianMesh)

Gives the non-local indicies that need to be updated
"""
function ghostboundary(mesh::PartionedCartesianMesh{N}) where N
    fI = first(parentindicies(mesh))
    lI = last(parentindicies(mesh))

    upper = ntuple(N) do i
        head, tail = select(fI, lI, i)
        (head..., fI[i] - 1, tail...)
    end

    lower = ntuple(N) do i
        head, tail = select(fI, lI, i)
        (head..., lI[i] + 1, tail...)
    end

    return (upper..., lower...)
end

function select(fI, lI, i)
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

P = Partition(CartesianIndices((1:1024, 1:1024)), ranks)
inds = rankindices(mpirank)

mesh = PartionedCartesianMesh(inds)
for cells in ghostboundary(mesh)
    other = locate(P, cells)
    @show other, cells
end
```
"""
struct Partition{N}
    indices    :: CartesianIndices{N}
    ranks      :: Array{Int,N}                     # ranks[i]==r ⇒ rank r has piece i
    partitions :: Array{CartesianIndices{N}, {N}}  # the indicies of piece i
    function Partition{N}(inds::CartesianIndices{N}, ranks :: Array{Int,N})
        partitions = partitions(inds, size(ranks))
        new(inds, ranks, partitions)
    end
end

function locate(p::Partition, I...)
    i = findfirst(P->I in P, p.partitions)
    p.ranks[i]
end

function rankindices(p::Partition, r)
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
