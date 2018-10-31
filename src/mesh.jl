module Meshing
    export Mesh, CartesianMesh, PeriodicCartesianMesh
    export faces, neighbor, neighbors, overelems

using Base.Cartesian

"""
    Mesh{N}

`Mesh` is a arbitrary and potentially unstructured mesh, consisting of many elems.
The mesh encodes the connectivity between different elems.

## Interface
- [`overelems`](@ref) apply a function over all elements of the mesh.
- [`neighbors`](@ref) query all neighbors of a given elem in the mesh.
- [`faces`](@ref) query all faces of a given elem in the mesh.
- [`neighbor`](@ref) given a elem and a face, query the neighboring elem in the mesh.

## Terminology
- A face is intersection between two elements.
  - This entails that in an adaptive mesh, even though elements might be rectangular and have four sides
    the sides are split into multiple faces.
"""
abstract type Mesh{N} end

"""
    faces(elem, mesh::Mesh)
"""
faces(elem, mesh::Mesh)  = throw(MethodError(faces, (typeof(elem), typeof(mesh))))

"""
    neighbor(elem, face, mesh::Mesh)
"""
neighbor(elem, face, mesh::Mesh) = throw(MethodError(neighbor, (typeof(elem), typeof(face), typeof(mesh))))

"""
    neighbors(elem, mesh::Mesh)
"""
neighbors(elem, mesh::Mesh) = map(face -> neighbor(elem, face, mesh), faces(elem, mesh))

"""
    overelems(f::Function, mesh::Mesh, args...)
"""
overelems(f, mesh::Mesh, args...) = throw(MethodError(overelems, (typeof(f), typeof(mesh), typeof(args))))

elemindices(mesh::Mesh) = throw(MethodError(elemindices, (typeof(Mesh),)))

"""
    CartesianMesh{N} <: Mesh{N} 

A `CartesianMesh` is a [`Mesh`](@ref) over a cartesian space.
The faces of a elem in a `CartesianMesh` are [`CartesianNeighbors`](@ref), 
and the elem indicies are [`CartesianIndex`](@ref).
"""
abstract type CartesianMesh{N} <: Mesh{N} end

faces(elem, mesh::CartesianMesh{N}) where {N} = CartesianNeighbors{N}()

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

elemindices(mesh::PeriodicCartesianMesh) = mesh.inds

Base.mod(x::T, y::AbstractUnitRange{T}) where {T<:Integer} = y[mod1(x - y[1] + 1, length(y))]
Base.mod(x::CartesianIndex{N}, y::CartesianIndices{N}) where {N} = CartesianIndex(ntuple(n->mod(x[n], axes(y)[n]), N))

neighbor(elem, face, mesh::PeriodicCartesianMesh) =
    (elem + face) in mesh.inds ? elem + face : mod(elem + face, mesh.inds)

function overelems(f::F, mesh::PeriodicCartesianMesh, args...) where F
    for I in elemindices(mesh)
        f(I, mesh, args...)
    end
end

end # module
