module Meshing
    export Mesh, CartesianMesh, PartionedCartesianMesh, PeriodicCartesianMesh
    export faces, neighbor, neighbors, overelems

using Base.Cartesian

"""
    Mesh{N}

`Mesh` is a arbitrary and potentially unstructured mesh, consisting of many cells.
The mesh encodes the connectivity between different cells.

## Interface
- [`overelems`](@ref) apply a function over all elements of the mesh.
- [`neighbors`](@ref) query all neighbors of a given cell in the mesh.
- [`faces`](@ref) query all faces of a given cell in the mesh.
- [`neighbor`](@ref) given a cell and a face, query the neighboring cell in the mesh.
"""
abstract type Mesh{N} end

"""
    faces(cell, mesh::Mesh)
"""
faces(cell, mesh::Mesh)  = throw(MethodError(faces, (typeof(cell), typeof(mesh))))

"""
    neighbor(cell, face, mesh::Mesh)
"""
neighbor(cell, face, mesh::Mesh) = throw(MethodError(neighbor, (typeof(cell), typeof(face), typeof(mesh))))

"""
    neighbors(cell, mesh::Mesh)
"""
neighbors(cell, mesh::Mesh) = map(face -> neighbor(cell, face, mesh), faces(cell, mesh))

"""
    overelems(f::Function, mesh::Mesh, args...)
"""
overelems(f, mesh::Mesh, args...) = throw(MethodError(overelems, (typeof(f), typeof(mesh), typeof(args))))

"""
    CartesianMesh{N} <: Mesh{N} 

A `CartesianMesh` is a [`Mesh`](@ref) over a cartesian space.
The faces of a cell in a `CartesianMesh` are [`CartesianNeighbors`](@ref), 
and the cell indicies are [`CartesianIndex`](@ref).
"""
abstract type CartesianMesh{N} <: Mesh{N} end

faces(cell, mesh::CartesianMesh{N}) where {N} = CartesianNeighbors{N}()
cellindices(mesh::CartesianMesh) = throw(MethodError(cellindices, (CartesianMesh,)))

struct CartesianNeighbors{N} <: AbstractVector{CartesianIndex{N}} end
Base.size(::CartesianNeighbors{N}) where {N} = (2N,)
Base.axes(::CartesianNeighbors{N}) where {N} = (1:2N,)
Base.getindex(::CartesianNeighbors{N}, i::Int) where {N} = CartesianIndex(ntuple(n -> i ==     n ?  1 :
                                                                                      i == n + N ? -1 :
                                                                                                    0 , N))

"""
    PeriodicCartesianMesh{N} <: CartesianMesh{N}

A `PeriodicCartesianMesh{N}` is a [`CartesianMesh`](@ref) with a periodic boundary condition.
"""
struct PeriodicCartesianMesh{N} <: CartesianMesh{N}
    inds::CartesianIndices{N}
end

cellindices(mesh::PeriodicCartesianMesh) = mesh.inds

Base.mod(x::T, y::AbstractUnitRange{T}) where {T<:Integer} = y[mod1(x - y[1] + 1, length(y))]
Base.mod(x::CartesianIndex{N}, y::CartesianIndices{N}) where {N} = CartesianIndex(ntuple(n->mod(x[n], axes(y)[n]), N))

neighbor(cell, face, mesh::PeriodicCartesianMesh) =
    (cell + face) in mesh.inds ? cell + face : mod(cell + face, mesh.inds)

# opposite(face, mesh::PeriodicCartesianMesh) = face * -1

function overelems(f::F, mesh::PeriodicCartesianMesh, args...) where F
    for I in cellindices(mesh)
        f(I, mesh, args...)
    end
end

end # module
