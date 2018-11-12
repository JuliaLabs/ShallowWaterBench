module TotallyNotApproxFun

using StaticArrays
using Base.Iterators
using LinearAlgebra
using Canary

export Fun, ComboFun, approximate, ProductFun, LagrangeFun, VectorFun, MultilinearFun
export Basis, OrthoBasis, ProductBasis, LagrangeBasis
export points, LobattoPoints
export normal
export ∇, ∫, ∫Ψ, ∫∇Ψ

const DEBUG = false

include("util.jl")

#A representation of a T-valued function in an N-Dimensional space.
abstract type Fun{T, N} end

summary(io, fn::Fun{T}) where {T} = show(IOContext(io, :typeinfo=>T), "$(nameof(f))()≈$(value(fn))")

function Base.show(io::IO, f::Fun{T}) where {T}
    if (get(io, :typeinfo, Any) <: typeof(f))
        print(io, "ƒ()≈")
    elseif get(io, :compact, false)
        print(io, "$(nameof(f))()::$T≈")
    else
        print(io, "$(typeof(f))()≈")
    end
    show(IOContext(io, :typeinfo=>T), value(f))
end

#An approximation of the value of a function (used for printing). For now, most of our funs happen to be defined on -1 to 1. In the future, defining funs over a `Space` type would allow us to implement something more precise here.
value(f::Fun{T, N}) where {T, N} = f(@SVector zeros(N))

#A discretization of a (T-valued function)-valued function in an N-Dimensional space.
abstract type Basis{T, N, F<:Fun{T, N}} <: AbstractArray{F, N} end

#A function represented by a linear combination of basis functions
struct ComboFun{T, N, B<:Basis{<:Any, N}, C<:AbstractArray{T}} <: Fun{T, N}
    basis::B
    coeffs::C
    function ComboFun(basis::B, coeffs::C) where {T, N, B<:Basis{<:Any, N}, C<:AbstractArray{T}}
        DEBUG && @assert length(basis) == length(coeffs)
        new{T, N, B, C}(basis, coeffs)
    end
end

(f::ComboFun)(x...) = apply(f, SVector(x...))
#f(x) is just sum_i(c_i * b_i(x))
@inline function apply(f::ComboFun, x::AbstractVector)
    #sum(f.coeffs .* apply.(f.basis, Ref(x)))
    i = first(eachindex(f.coeffs))
    y = f.coeffs[i] * apply(f.basis[i], x)
    y -= y
    for i in eachindex(f.coeffs)
        y += f.coeffs[i] * apply(f.basis[i], x)
    end
    return y
end

#A basis corresponding to a set of points, where the basis function i is one(T) at point i and zero(T) everywhere else
abstract type OrthoBasis{T, N, F} <: Basis{T, N, F} end
points(::OrthoBasis) = error("Subtypes of OrthoBasis must define the points function")
approximate(f, b::OrthoBasis) = ComboFun(b, map(f, points(b)))

for op in (:(Base.:+), :(Base.:-), :(Base.zero), :(LinearAlgebra.transpose), :(LinearAlgebra.adjoint), :(LinearAlgebra.norm), :(Base.exp), :(Base.sqrt), :(Base.abs))
    @eval begin
        function $(op)(a::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            ComboFun(a.basis, map($op, a.coeffs))
        end
    end
end
for op in (:(Base.:+), :(Base.:-), :(Base.:*), :(Base.:/), :(Base.exp), :(Base.:^), :(Base.max))
    @eval begin
        function $op(a::ComboFun{T, N, B}, b::ComboFun{S, N, B}) where {T, S, N, B <: OrthoBasis}
            DEBUG && @assert a.basis == b.basis
            ComboFun(a.basis, map($op, a.coeffs, b.coeffs))
        end
        function $op(a::ComboFun{T, N, B}, b) where {T, N, B <: OrthoBasis}
            ComboFun(a.basis, map(x->$op(x, b), a.coeffs))
        end
        function $op(a::ComboFun{T, N, B}, b::Fun) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
        function $op(a, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            ComboFun(b.basis, map(x->$op(a, x), b.coeffs))
        end
        function $op(a::Fun, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
    end
end

#A function which is a constant
struct ConstFun{T, N} <: Fun{T, N}
    val::T
end

(f::ConstFun)(x...) = apply(f, SVector(x...))
@inline function apply(f::ConstFun{T, N}, x::AbstractVector) where {T, N}
    f.val
end
WrapFun(x) = ConstFun(x)
WrapFun(x::Fun) = x

#A function which is a product of one-dimensional functions
struct ProductFun{T, N, F <: Tuple{Vararg{Fun{T, 1}, N}}} <: Fun{T, N}
    funs::F
    ProductFun(funs::Fun{T, 1}) where {T} = new{T, 1, Tuple{typeof(funs)}}((funs,))
    ProductFun(funs::Fun{T, 1}...) where {T} = new{T, length(funs), typeof(funs)}(funs)
end
function ProductFun(args...)
    if any(isa.(args, Fun))
        return ProductFun(WrapFun.(args))
    else
        return prod(args)
    end
end

(f::ProductFun)(x...) = apply(f, SVector(x...))
@inline function apply(f::ProductFun{T, N}, x::AbstractVector) where {T, N}
    #prod(apply.(f.funs, SVector.(x)))
    y = apply(f.funs[1], SVector(x[1]))
    for i in 2:N
        y *= apply(f.funs[i], SVector(x[i]))
    end
    y
end

#A basis which is an outer product of one-dimensional bases
struct ProductBasis{T, N, B <: Tuple{Vararg{OrthoBasis{T, 1}, N}}} <: OrthoBasis{T, N, Fun{T, N}}
    bases::B
    ProductBasis(basis::OrthoBasis{T, 1}) where {T} = new{T, 1, Tuple{typeof(basis)}}((basis,))
    ProductBasis(bases::OrthoBasis{T, 1}...) where {T} = new{T, length(bases), typeof(bases)}(bases)
end
#ProductBasis{T, N, B}() where {T, N, B} = ProductBasis((b() for b in B.parameters)...)
Base.size(b::ProductBasis) = map(length, b.bases)
Base.eltype(b::ProductBasis{T, N}) where {T, N} = ProductFun{T, N, Tuple{map(eltype, b.bases)...}}
Base.@propagate_inbounds function Base.getindex(b::ProductBasis, i::Int...)::eltype(b)
    ind = Tuple(CartesianIndices(b)[i...])
    ProductFun(map((b,j)->getindex(b, j), b.bases, ind)...)
end
points(b::ProductBasis) = SVector.(collect(product(map(points, b.bases)...)))
#Base.Broadcast.broadcastable(b::ProductBasis) = SArray{Tuple{size(b)...}}(b) #TODO generalize to non-static children

#=
function Base.getindex(f::ComboFun{<:Any, N, <:ProductBasis}, I::CartesianIndex{N}) where {N}
    I = Tuple(I)
    basis = ProductBasis(f.basis.bases[findall(iszero, I)]...)
    coeffs = f.coeffs[ntuple(n -> I[n] == 0 ? Colon() : (I[n] == 1 ? lastindex(f.coeffs, n) : 1), N)...]
    return ComboFun(basis, coeffs)
end

function Base.setindex!(f::ComboFun{<:Any, N, <:ProductBasis}, g::ComboFun{<:Any, M, <:ProductBasis}, I::CartesianIndex{N}) where {N, M}
    I = Tuple(I)
    DEBUG && @assert ProductBasis(f.basis.bases[findall(iszero, I)]...) == g.basis
    f.coeffs[ntuple(n -> I[n] == 0 ? Colon() : (I[n] == 1 ? lastindex(f.coeffs, n) : 1), N)...] = g.coeffs
    return f
end
=#

normal(face::CartesianIndex) = SVector(face)

@generated function Base.getindex(f::ComboFun{<:Any, N, <:ProductBasis}, I::CartesianIndex{N}) where {N}
    thunk = quote
        Base.@_propagate_inbounds_meta
    end
    for dim in 1:N
        for (offset, select) in ((-1, 1), (1, :end))
            push!(thunk.args, quote
                if I == CartesianIndex($(ntuple(n->n == dim ? offset : 0, N)...))
                    basis = ProductBasis($(ntuple(n->:(f.basis.bases[$(n >= dim ? n + 1 : n)]), N - 1)...))
                    coeffs = f.coeffs[$(ntuple(n->n == dim ? select : :(:), N)...)]
                    return ComboFun(basis, coeffs)
                end
            end)
        end
    end
    push!(thunk.args, :(error("Invalid ind")))
    return thunk
end

@generated function Base.setindex!(f::ComboFun{<:Any, N, <:ProductBasis}, g::ComboFun{<:Any, M, <:ProductBasis}, I::CartesianIndex{N}) where {N, M}
    thunk = quote
        Base.@_propagate_inbounds_meta
    end
    for dim in 1:N
        for (offset, select) in ((-1, 1), (1, :end))
            push!(thunk.args, quote
                if I == CartesianIndex($(ntuple(n->n == dim ? offset : 0, N)...))
                    DEBUG && @assert ProductBasis($(ntuple(n->:(f.basis.bases[$(n >= dim ? n + 1 : n)]), N - 1)...)) == g.basis
                    f.coeffs[$(ntuple(n->n == dim ? select : :(:), N)...)] = g.coeffs
                    return f
                end
            end)
        end
    end
    push!(thunk.args, :(error("Invalid ind")))
    return thunk
end

#The minimum-degree polynomial function which is 1 at the nth point and 0 at the other points
struct LagrangeFun{T, P <: AbstractVector{T}} <: Fun{T, 1}
    points::P
    n::Int
end

LagrangeFun(points::AbstractVector, n) = LagrangeFun{eltype(points), typeof(points)}(points, n)
(f::LagrangeFun)(x...) = apply(f, SVector(x...))
#This method is mostly here for clarity, it probably shouldn't be called (TODO specialize somewhere with a stable interpolation routine)
@inline function apply(f::LagrangeFun{T}, x::AbstractVector{S}) where {T, S}
    DEBUG && @assert length(x) == 1
    #return prod([(x[1] - f.points[i])/(f.points[f.n] - f.points[i]) for i in filter(!isequal(f.n), eachindex(f.points))])
    T′ = promote_type(T, S)
    T′ = Base.promote_op(/, T′, T′)
    y = one(T′)
    for i in eachindex(f.points)
        if i != f.n
            y *= (x[1] - f.points[i])/(f.points[f.n] - f.points[i])
        end
    end
    y::T′
end

#A basis of polynomials
struct LagrangeBasis{T, P <: AbstractVector{T}} <: OrthoBasis{T, 1, LagrangeFun{T, P}}
    points::P
end
#LagrangeBasis{T, P}() where {T, P <: AbstractVector{T}} = LagrangeBasis{T, P}(P())

Base.size(b::LagrangeBasis) = size(b.points)
@inline Base.getindex(b::LagrangeBasis, i::Int) = LagrangeFun(b.points, i)
points(b::LagrangeBasis) = b.points
#Base.Broadcast.broadcastable(b::LagrangeBasis) = SArray{Tuple{size(b)...}}(b) #TODO generalize to non-static children

#A vector representing Lobatto Points
struct LobattoPoints{T, N} <: AbstractVector{T} end

Base.size(p::LobattoPoints{T, N}) where {T, N} = (N,)
@generated function Base.getindex(p::LobattoPoints{T, N}, i::Int) where {T, N}
    return quote
        Base.@_inline_meta
        $(SVector{N, T}(lglpoints(T, N - 1)[1]))[i]
    end
end
LobattoPoints(n) = LobattoPoints{Float64, n + 1}()
#Base.Broadcast.broadcastable(p::LobattoPoints) = SArray{Tuple{size(p)...}}(p)

MultilinearFun(x₀, x₁, y₀, y₁) = MultilinearFun(SVector(x₀), SVector(x₁), SVector(y₀), SVector(y₁))
function MultilinearFun(x₀::SVector{N}, x₁::SVector{N}, y₀::SVector{N}, y₁::SVector{N}) where N
    basis = ProductBasis(ntuple(i->LagrangeBasis(SVector(x₀[i], x₁[i])), N)...)
    coeffs = map(SVector, collect(product(ntuple(i->SVector(y₀[i], y₁[i]), N)...)))
    ComboFun(basis, coeffs)
end

D(p) = spectralderivative(p)
D(p::SVector{N}) where {N} = SMatrix{N, N}(spectralderivative(p))
@generated function D(p::LobattoPoints{T, N}) where {T, N}
    return quote
        Base.@_inline_meta
        $(SMatrix{N, N}(spectralderivative(LobattoPoints{T, N}())))
    end
end

∫(f::ComboFun) = sum(map(∫, f.basis) .* f.coeffs)

∫(f::ProductFun) = prod(map(∫, f.funs))

@generated function Base.map(::typeof(∫), b::LagrangeBasis{T, <:LobattoPoints{T, N}}) where {T, N}
    return quote
        Base.@_inline_meta
        return $(SVector{N, T}(lglpoints(T, N - 1)[2]))
    end
end

@inline function Base.map(::typeof(∫), b::ProductBasis{T, N}) where {T, N}
    return map(prod, collect(product(ntuple(n->map(∫, b.bases[n]), N)...)))
end

@generated function ∫(f::LagrangeFun{T, <:LobattoPoints{T, N}}) where {T, N}
    return quote
        Base.@_inline_meta
        return $(SVector{N, T}(lglpoints(T, N - 1)[2]))[f.n]
    end
end

∇(f::ComboFun) = sum(map(∇, f.basis) .* f)

function ∇(f::LagrangeFun) where {T, N}
    return ComboFun(f.basis, D(f.points)[:, f.n])
end

function ∇(f::ComboFun{<:Any, 1, <:LagrangeBasis}) where {T, N}
    return ComboFun(f.basis, D(f.basis.points) * f.coeffs)
end

function ∇(f::ComboFun{<:Any, 1, <:LagrangeBasis{<:Any, <:SVector{2}}})
    dx = (f.coeffs[2] - f.coeffs[1]) / (f.basis.points[2] - f.basis.points[1])
    return ComboFun(f.basis, SVector(dx, dx))
end

function ∇(f::ComboFun{T, N, <:ProductBasis}) where {T, N}
    partials = [dimsmapslices(n, c-> ∇(ComboFun(b, c)).coeffs, f.coeffs) for (n, b) in enumerate(f.basis.bases)]
    ComboFun(f.basis, dimscat.(ndims(T) + 1, partials...))
end

function ∫∇Ψ(f::ComboFun) where {T, N}
    return ComboFun(f.basis, map(b -> ∫(∇(b)' * f), f.basis))
end

function ∫∇Ψ(f::ComboFun{<:Any, 1, <:LagrangeBasis})
    ω = map(∫, f.basis)
    return ComboFun(f.basis, D(f.basis.points)' * (ω .* f.coeffs))
end

function ∫∇Ψ(f::ComboFun{<:Any, 1, <:ProductBasis{<:Any, 1, <:Tuple{Vararg{<:LagrangeBasis}}}})
    ω = map(∫, f.basis)
    return ComboFun(f.basis, D(f.basis.bases[1].points)' * (ω .* f.coeffs))
end

function ∫∇Ψ(f::ComboFun{<:AbstractVector, 2, <:ProductBasis{<:Any, 2, <:Tuple{Vararg{<:LagrangeBasis}}}})
    ω = map(∫, f.basis)
    return ComboFun(f.basis, dimsmapslices(1, c->D(f.basis.bases[1].points)' * c, ω.*(getindex.(f.coeffs, 1))) +
                             dimsmapslices(2, c->D(f.basis.bases[2].points)' * c, ω.*(getindex.(f.coeffs, 2))))
end

function ∫∇Ψ(f::ComboFun{<:AbstractMatrix, 2, <:ProductBasis{<:Any, 2, <:Tuple{Vararg{<:LagrangeBasis}}}})
    ω = map(∫, f.basis)
    return ComboFun(f.basis, dimsmapslices(1, c->D(f.basis.bases[1].points)' * c, map(*, ω, map(c -> getindex(c, 1, :), f.coeffs))) +
                             dimsmapslices(2, c->D(f.basis.bases[2].points)' * c, map(*, ω, map(c -> getindex(c, 2, :), f.coeffs))))
end


function ∫Ψ(f::ComboFun)
    return ComboFun(f.basis, f.coeffs .* map(∫, f.basis))
end

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

end
