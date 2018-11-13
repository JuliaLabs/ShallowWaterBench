module Meshing
    export Mesh, CartesianMesh, PeriodicCartesianMesh, GhostCartesianMesh, CPU
    export faces, neighbor, opposite, neighbors, overelems, storage, elems
    export ghostboundaries, boundaries, backend, translate

using Base.Cartesian
using OffsetArrays

abstract type Backend end
struct CPU <: Backend end

"""
    Mesh{N, B<:Backend}

`Mesh` is a arbitrary and potentially unstructured mesh, consisting of many elems.
The mesh encodes the connectivity between different elems.

## Interface
- [`elems`](@ref) obtain the iterator over the elements.
- [`overelems`](@ref) apply a function over all elements of the mesh.
- [`neighbors`](@ref) query all neighbors of a given elem in the mesh.
- [`faces`](@ref) query all faces of a given elem in the mesh.
- [`neighbor`](@ref) given a elem and a face, query the neighboring elem in the mesh.
- [`storage`](@ref) allocate a storage that is compatible with a mesh, and can be indexed by [`elems`](@ref).
- [`backend`](@ref) query which backend is used by the mesh.

## Terminology
- A face is intersection between two elements.
  - This entails that in an adaptive mesh, even though elements might be rectangular and have four sides
    the sides are split into multiple faces.
"""
abstract type Mesh{N, B<:Backend} end

"""
    backend(Mesh)
"""
backend(::Mesh{N, B}) where {N, B} = B()

"""
    faces(elem, mesh::Mesh)
"""
faces(elem, mesh::Mesh)  = throw(MethodError(faces, (typeof(elem), typeof(mesh))))

"""
    neighbor(elem, face, mesh::Mesh)
"""
neighbor(elem, face, mesh::Mesh) = throw(MethodError(neighbor, (typeof(elem), typeof(face), typeof(mesh))))

"""
    opposite(face, elem, mesh::Mesh)
"""
opposite(face, elem, mesh::Mesh) = throw(MethodError(opposite, (typeof(face), typeof(elem), typeof(mesh))))

"""
    neighbors(elem, mesh::Mesh)
"""
neighbors(elem, mesh::Mesh) = map(face -> neighbor(elem, face, mesh), faces(elem, mesh))

"""
    overelems(f::Function, mesh::Mesh, args...)
"""
overelems(f, mesh::Mesh, args...) = throw(MethodError(overelems, (typeof(f), typeof(mesh), typeof(args))))

"""
    elems(mesh)
"""
elems(mesh::Mesh) = throw(MethodError(elems, (typeof(Mesh),)))

"""
    storage(T, mesh)
"""
storage(::Type{T}, mesh::Mesh) where T = throw(MethodError(storage, (T, typeof(Mesh),)))

function Base.map(f::F, mesh::Mesh) where F
    T = Base._return_type(f, Tuple{eltype(elems(mesh))})
    if !isconcretetype(T)
        I = first(elems(mesh))
        T = typeof(f(I))
    end
    out = storage(T, mesh)
    overelems(mesh, out) do I, mesh, out
        Base.@_inline_meta
        out[I] = f(I)
    end
    return out
end

"""
    CartesianMesh{N, B} <: Mesh{N, B}

A `CartesianMesh` is a [`Mesh`](@ref) over a cartesian space.
The faces of a elem in a `CartesianMesh` are [`CartesianNeighbors`](@ref),
and the elem indicies are [`CartesianIndex`](@ref).
"""
abstract type CartesianMesh{N, B} <: Mesh{N, B} end

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
Base.map(f::F, cn::CartesianNeighbors{N}) where {F,N} = ntuple(i->f(cn[i]), Val(2N))

opposite(face, elem, mesh::CartesianMesh) = -face

"""
    PeriodicCartesianMesh{N, B} <: CartesianMesh{N, B}

A `PeriodicCartesianMesh{N}` is a [`CartesianMesh`](@ref) with a periodic boundary condition.
"""
struct PeriodicCartesianMesh{N, B, I<:CartesianIndices{N}} <: CartesianMesh{N, B}
    inds::I
    function PeriodicCartesianMesh(::B, inds::CartesianIndices{N}) where {B, N}
        new{N, B, typeof(inds)}(inds)
    end
end
PeriodicCartesianMesh(inds::CartesianIndices; backend = CPU()) = PeriodicCartesianMesh(backend, inds)
PeriodicCartesianMesh(space::Tuple; backend = CPU()) = PeriodicCartesianMesh(backend, CartesianIndices(space))
PeriodicCartesianMesh(ranges::AbstractUnitRange...; backend = CPU()) = PeriodicCartesianMesh(ranges; backend = backend)

elems(mesh::PeriodicCartesianMesh) = mesh.inds

@inline Base.mod(x::T, y::AbstractUnitRange{T}) where {T<:Integer} = @inbounds y[mod1(x - y[1] + 1, length(y))]
@inline Base.mod(x::CartesianIndex{N}, y::CartesianIndices{N}) where {N} = CartesianIndex(ntuple(n->(Base.@_inline_meta; mod(x[n], axes(y)[n])), Val(N)))

opposite(face, elem, mesh::PeriodicCartesianMesh) = -face

@inline function neighbor(elem::T, face, mesh::PeriodicCartesianMesh) where {T}
    elem′ = (elem + face) in mesh.inds ? elem + face : mod(elem + face, mesh.inds)
    elem′::T
end

function overelems(f::F, mesh::PeriodicCartesianMesh{N, CPU}, args...) where {F, N}
    for I in elems(mesh)
        f(I, mesh, args...)
    end
end

function storage(::Type{T}, mesh::PeriodicCartesianMesh{N, CPU}) where {T, N}
    inds = elems(mesh)
    underlying = Array{T}(undef, map(length, axes(inds))...)
    return OffsetArray(underlying, inds.indices)
end

function translate(mesh::PeriodicCartesianMesh{N}, boundary) where N
    pI = axes(elems(mesh))
    b = ntuple(Val(N)) do i
        b = boundary.indices[i]
        if length(b) > 1
            b
        else
            s = mod(b[1], pI[i])
            s:s # because if all dimensions are single numbers,
                # we will call CartesianIndices(a::Int, b::Int)
                # which returns CartesianIndices((1:a, 1:b))
        end
    end
    CartesianIndices(b)
end

"""
    GhostCartesianMesh{B, N, M} <: GhostCartesianMesh{N}

Represents the local region of a [`GhostCartesianMesh`](@ref) `M`.
The local storage is expected to be an `OffsetArray`

The current ghost-boundary is of size 1, a future extension would be to make this configurable.
Use [`ghostboundaries`](@ref) to query it.
"""
struct GhostCartesianMesh{N, B} <: CartesianMesh{N, B}
    inds :: CartesianIndices{N}

    function GhostCartesianMesh(::B, inds::CartesianIndices{N}) where {N,B}
        new{N, B}(inds)
    end
end
GhostCartesianMesh(inds::CartesianIndices; backend = CPU()) = GhostCartesianMesh(backend, inds)
GhostCartesianMesh(space::Tuple; backend = CPU()) = GhostCartesianMesh(backend, CartesianIndices(space))
GhostCartesianMesh(ranges::AbstractUnitRange...; backend = CPU()) = GhostCartesianMesh(ranges; backend = backend)

elems(mesh::GhostCartesianMesh) = mesh.inds
neighbor(elem, face, mesh::GhostCartesianMesh) = elem + face

function overelems(f::F, mesh::GhostCartesianMesh{N, CPU}, args...) where {F, N}
    for I in elems(mesh)
        f(I, mesh, args...)
    end
end

function storage(::Type{T}, mesh::GhostCartesianMesh{N, CPU}) where {T, N}
    inds = elems(mesh).indices
    inds = ntuple(Val(N)) do i
        I = inds[i]
        (first(I)-1):(last(I)+1)
    end

    underlaying = Array{T}(undef, map(length, inds)...)
    return OffsetArray(underlaying, inds)
end

# TODO: refactor boundaries and ghostboundaries
"""
    boundaries(mesh::CartesianMesh)

Gives the boundaries of the cartesian region.
"""
function boundaries(mesh::CartesianMesh{N}) where N
    fI = first(elems(mesh))
    lI = last(elems(mesh))

    fs = faces(fI, mesh)

    Tuple(CartesianIndices(ntuple(Val(N)) do i
        if f[i] == -1
            fI[i]
        elseif f[i] == 1
            lI[i]
        else
            fI[i]:lI[i]
        end
    end) for f in fs)
end

"""
    ghostboundaries(mesh::GhostCartesianMesh)

Gives the boundaries of the cartesian region.
"""
function ghostboundaries(mesh::GhostCartesianMesh{N}) where N
    fI = first(elems(mesh))
    lI = last(elems(mesh))

    fs = faces(fI, mesh)

    Tuple(CartesianIndices(ntuple(Val(N)) do i
        if f[i] == -1
            fI[i]-1
        elseif f[i] == 1
            lI[i]+1
        else
            fI[i]:lI[i]
        end
    end) for f in fs)
end

include("mpi_mesh.jl")

end # module
