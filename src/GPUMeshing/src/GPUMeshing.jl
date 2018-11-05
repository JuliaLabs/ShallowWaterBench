module GPUMeshing

export GPU

using Adapt
using OffsetArrays

Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(adapt(to, parent(x)), x.offsets)
Base.Broadcast.BroadcastStyle(::Type{<:OffsetArray{<:Any, <:Any, AA}}) where AA = Base.Broadcast.BroadcastStyle(AA)

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

function storage(::Type{T}, mesh::GhostCartesianMesh{N, GPU}) where {T, N}
    inds = elems(mesh).indices
    inds = ntuple(N) do i
        I = inds[i]
        (first(I)-1):(last(I)+1)
    end

    underlaying = CuArray{T}(undef, map(length, inds)...)
    return OffsetArray(underlaying, inds)
end

function overelems(f::F, mesh::CartesianMesh{N, GPU}, args...) where {F, N}

    function kernelf(f::F, elems, mesh, args...) where F
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        i > length(elems) && return nothing
        I = elems[i]
        f(I, mesh, args...)
        return nothing
    end

    # The below does this:
    # @cuda threads=length(elems(mesh)) kernelf(f, elems(mesh), mesh, args...)
    cuargs = (f, elems(mesh), mesh, args...)
    GC.@preserve cuargs begin
        kernel_args = map(x->adapt(CUDAnative.Adaptor(), x), cuargs)

        kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
        kernel = cufunction(kernelf, kernel_tt)

        n = length(elems(mesh))
        threads = min(n, CUDAnative.maxthreads(kernel))
        blocks = ceil(Int, n / threads)

        @info("kernel configuration", N, threads, blocks,
            CUDAnative.maxthreads(kernel),
            CUDAnative.registers(kernel),
            CUDAnative.memory(kernel))

        kernel(kernel_args...; threads=threads, blocks=blocks)
    end
    return nothing
end

##
# Hacks
##
import Base.Broadcast
Broadcast.broadcasted(::Broadcast.DefaultArrayStyle{1}, ::typeof(+), r::AbstractUnitRange, x::Real) = Base._range(first(r) + x, nothing, nothing, length(r))

##
# Compatibility between TotallyNotApproxFun and CUDAnative
##
import TotallyNotApproxFun: ComboFun, OrthoBasis, Fun

for op in (:(CUDAnative.:exp),)
    @eval begin
        function $(op)(a::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            ComboFun(a.basis, map($op, a.coeffs))
        end
    end
end
for op in (:(CUDAnative.:exp), )
    @eval begin
        function $op(a::ComboFun{T, N, B}, b::ComboFun{S, N, B}) where {T, S, N, B <: OrthoBasis}
            @assert a.basis == b.basis
            ComboFun(a.basis, $op.(a.coeffs, b.coeffs))
        end
        function $op(a::ComboFun{T, N, B}, b) where {T, N, B <: OrthoBasis}
            ComboFun(a.basis, $op.(a.coeffs, b))
        end
        function $op(a::ComboFun{T, N, B}, b::Fun) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
        function $op(a, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            ComboFun(b.basis, $op.(a, b.coeffs))
        end
        function $op(a::Fun, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
    end
end

end # module
