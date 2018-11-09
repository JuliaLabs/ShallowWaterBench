using StaticArrays

dimsmapslices(dims, f, x) = mapslices(f, x; dims = dims)

@inline function dimsmapslices(dims::Union{Int, Tuple}, f, x::StaticArray)
    return _mapslices_exposition(Val(dims), f, x)
end

@generated function _mapslices_exposition(::Val{dims}, f, a) where {dims}
    slicers = Array(collect(product((n in dims ? (:(Colon()),) : 1:size(a)[n] for n = 1:ndims(a))...)))
    slices = map(slicer -> :(f(a[$(slicer...)])), slicers)
    return quote
        Base.@_inline_meta
        _mapslices_denoument(Val(size(a)), Val(dims), $(slices...))
    end
end

@generated function _mapslices_denoument(::Val{oldsize}, ::Val{dims}, slices::StaticArray...) where {oldsize, dims}
    if all(isequal(size(slices[1])), map(size, slices))
        count = 0
        firstsize = (size(slices[1])..., ones(Int, length(oldsize))...)
        newsize = ((n in dims ? oldsize[n] : firstsize[count+=1] for n = 1:length(oldsize))...,)

        slicers = Array(collect(product((n in dims ? (:(Colon()),) : 1:newsize[n] for n = 1:length(newsize))...)))
        thunk = quote
            Base.@_inline_meta
            res = @MArray zeros($(eltype(slices[1])), $(newsize...))
        end
        for (i, slicer) in enumerate(slicers)
            push!(thunk.args, :(@inbounds res[$(slicer...)] = slices[$i]))
        end
        push!(thunk.args, :(return similar_type(slices[1], Size($(newsize...)))(res)))
        return thunk
    else
        return :(throw(DimensionMismatch()))
    end
end

dimscat(dims, as...) = cat(as...; dims = dims)

@inline function dimscat(dims::Union{Int, Tuple}, as::StaticArray...)
    return _cat(Val(dims), as...)
end

@generated function _cat(::Val{dims}, as...) where {dims}
    as = map(((n, a),)->map(i -> :(as[$n][$(Tuple(i)...)]), CartesianIndices(size(a))), enumerate(as))
    try
        res = cat(as..., dims=dims)
        return quote
            Base.@_inline_meta
            @inbounds return similar_type(as[1], promote_type(map(eltype, as)...), Size($(size(res)...)))(tuple($(res...)))
        end
    catch DimensionMismatch err
        return :(throw($err))
    end
end

StaticArrays.SVector(i::CartesianIndex) = SVector(Tuple(i))

function Base.collect(it::Base.Iterators.ProductIterator{<:Tuple{Vararg{SArray}}})
    SArray{Tuple{size(it)...},eltype(it),ndims(it),length(it)}(it...)
end

#=
function Base.collect(it::Base.Iterators.ProductIterator{TT}) where {TT<:Tuple{Vararg{LobattoPoints}}}
    sproduct(it.iterators)
end

_length(::Type{LobattoPoints{T, N}}) where {T, N} = N
_eltype(::Type{LobattoPoints{T, N}}) where {T, N} = T
using Base.Cartesian
@generated function sproduct(points::TT) where {N, TT<:Tuple{Vararg{LobattoPoints, N}}}
    lengths = map(_length, TT.parameters)
    eltypes = map(_eltype, TT.parameters)
    M = prod(lengths)
    I = CartesianIndices((lengths...,))
    quote
        Base.@_inline_meta
        @nexprs $N j->(P_j = points[j])
        @nexprs $N j->(S_j = length(P_j))
        @nexprs $M j->(elem_j = @ntuple $N k-> P_k[($I)[j][k]])
        @ncall $M SArray{Tuple{$(lengths...)}, Tuple{$(eltypes...)}, $N, $M} elem
    end
end

    @inbounds return similar_type(as[1], promote_type(map(eltype, as)...), Size($(size(res)...)))(tuple($(res...)))

        firstsize = (size(firstslice)..., $(ones(Int, length(oldsize))...))
        newsize = ($((n in dims ? oldsize[n] : :(firstsize[$(count+=1)]) for n = 1:length(oldsize))...),)
        count = 0

    count = 0

    return quote
        Base.@_inline_meta
        firstslice = $(slices[1])
        otherslices = $(slices[2:end])
        firstsize = $(ones(Int, ndims(a))...)
        if firstslice isa AbstractArray
            firstsize = (size(firstslice)..., firstsize...)
        end
        newsize = ($((n in dims ? size(a)[n] : :(firstsize[$(count += 1)]) for n = 1:ndims(a))...),)
        if firstslice isa StaticArray && all(isa(StaticArray
        return
    end
end

    for n in 1:ndims(a)
        if size(slices)[n] != 1
            slices = mapslices(tube -> :(dimscat($n, $(tube...))), slices; dims=n)
        end
    end
=#
