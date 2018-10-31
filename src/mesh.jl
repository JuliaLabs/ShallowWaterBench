module Meshing
    export Mesh, CartesianMesh, PeriodicCartesianMesh, GhostCartesianMesh, CPU, GPU
    export faces, neighbor, neighbors, overelems, storage
    export ghostboundary, backend, translate

using Base.Cartesian
using OffsetArrays

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
storage(::Type{T}, mesh::Mesh) where T = throw(MethodError(storage, (T, typeof(Mesh),)))

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
Base.length(::CartesianNeighbors{N}) where N = 2N
Base.axes(::CartesianNeighbors{N}) where {N} = (1:2N,)
Base.getindex(::CartesianNeighbors{N}, i::Int) where {N} = CartesianIndex(ntuple(n -> i ==     n ?  1 :
                                                                                      i == n + N ? -1 :
                                                                                                    0 , N))
Base.iterate(cn::CartesianNeighbors) = (cn[1], 2)
Base.iterate(cn::CartesianNeighbors, i) = i <= length(cn) ? (cn[i], i+1) : nothing
Base.IteratorSize(::CartesianNeighbors) = Base.HasShape{1}()	
Base.eltype(cn::CartesianNeighbors{N}) where N = CartesianIndex{N}
Base.map(f::F, cn::CartesianNeighbors{N}) where {F,N} = ntuple(i->f(cn[i]), 2N)

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

function storage(::Type{T}, mesh::PeriodicCartesianMesh{N}) where {T, N}
    inds = elemindices(mesh)
    underlaying = Array{Int64}(undef, map(length, axes(inds))...)
    return OffsetArray(underlaying, inds.indices)
end

function translate(mesh::PeriodicCartesianMesh{N}, boundary) where N 
    pI = axes(elemindices(mesh))
    b = ntuple(N) do i
        b = boundary.indices[i]
        length(b) > 1 ? b : mod(b[1], pI[i])
    end
    CartesianIndices(b)
end

abstract type Backend end
struct CPU <: Backend end
struct GPU <: Backend end

# TODO: Add backend to PeriodicCartesianMesh

"""
    GhostCartesianMesh{B, N, M} <: GhostCartesianMesh{N}

Represents the local region of a [`GhostCartesianMesh`](@ref) `M`.
The local storage is expected to be an `OffsetArray`

The current boundary is of size 1, a future extension would be to make this configurable.
"""
struct GhostCartesianMesh{B<:Backend, N} <: CartesianMesh{N}
    inds :: CartesianIndices{N}

    function GhostCartesianMesh(::B, inds::CartesianIndices{N}) where {B, N}
        new{B, N}(inds)
    end
end

elemindices(mesh::GhostCartesianMesh) = mesh.inds 
neighbor(elem, face, mesh::GhostCartesianMesh) = elem + face

backend(::GhostCartesianMesh{B}) where B = B()

function overelems(f::F, mesh::GhostCartesianMesh{CPU}, args...) where F
    for I in elemindices(mesh) 
        f(I, mesh, args...)
    end
end

function storage(::Type{T}, mesh::GhostCartesianMesh{CPU, N}) where {T, N}
    inds = elemindices(mesh).indices
    inds = ntuple(N) do i 
        I = inds[i]
        (first(I)-1):(last(I)+1)
    end

    underlaying = Array{Int64}(undef, map(length, inds)...)
    return OffsetArray(underlaying, inds)
end


"""
    ghostboundary(mesh::GhostCartesianMesh)

Gives the local indicies that need to be updated
"""
function ghostboundary(mesh::GhostCartesianMesh{B, N}) where {B, N}
    fI = first(elemindices(mesh))
    lI = last(elemindices(mesh))

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
        fI[i+j]:lI[i+j]
    end

    return head, tail
end
end # module
