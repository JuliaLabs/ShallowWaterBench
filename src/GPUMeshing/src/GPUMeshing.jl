module GPUMeshing

using Adapt
using OffsetArrays

Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(adapt(to, parent(x)), x.offsets)

using StructsOfArrays
using CuArrays
using CUDAnative

StructsOfArrays._type_with_eltype(::Type{<:CuArray}, T, N) = CuArray{T, N}
StructsOfArrays._type_with_eltype(::Type{CuDeviceArray{_T,_N,AS}}, T, N) where{_T,_N,AS} = CuDeviceArray(T,N,AS)

StructsOfArrays._type(::Type{<:CuArray}) = CuArray
StructsOfArrays._type(::Type{<:CuDeviceArray}) = CuDeviceArray

using SimpleMeshing
using .Meshing

struct GPU <: Meshing.Backend end
import .Meshing: storage, overelems

function storage(::Type{T}, mesh::PeriodicCartesianMesh{N, GPU}) where {T, N}
    inds = elems(mesh)
    underlying = CuArray{T}(undef, map(length, axes(inds))...)
    return OffsetArray(underlying, inds.indices)
end

function overelems(f::F, mesh::PeriodicCartesianMesh{N, GPU}, args...) where {F, N}
    error("Not implemented yet")
end

function storage(::Type{T}, mesh::GhostCartesianMesh{N, GPU}) where {T, N}
    inds = elems(mesh).indices
    inds = ntuple(N) do i 
        I = inds[i]
        (first(I)-1):(last(I)+1)
    end

    underlaying = CuArray{T}(undef, map(length, inds)...)
    return OffsetArray(underlaying, inds)
end

function overelems(f::F, mesh::GhostCartesianMesh{N, GPU}, args...) where {F, N}
    error("Not implemented yet")
end

end # module
