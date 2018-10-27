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
# 2. Associate a ghost boundary with a particular process
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
