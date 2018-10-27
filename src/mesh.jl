using Base.Cartesian

# N is spatial dimension
# A is data type
abstract type Mesh{N} end

struct PeriodicCartesianMesh{N} <: Mesh{N}
    inds::CartesianIndices{N}
end

struct CartesianNeighbors{N} <: AbstractVector{CartesianIndex{N}} end

Base.size(::CartesianNeighbors{N}) where {N} = (2N,)

Base.axes(::CartesianNeighbors{N}) where {N} = (1:2N,)

Base.getindex(::CartesianNeighbors{N}, i::Int) where {N} = CartesianIndex(ntuple(n -> i ==     n ?  1 :
                                                                                      i == n + N ? -1 :
                                                                                                    0 , N))

faces(cell, mesh::PeriodicCartesianMesh{N}) where {N} = CartesianNeighbors{N}()

neighbor(cell, face, mesh::PeriodicCartesianMesh{N}) = cell + face

opposite(face, mesh::PeriodicCartesianMesh{N}) = face * -1

neighbors(cell, mesh::Mesh{N}) = map(face -> neighbor(cell, face, mesh), faces(cell, mesh))
