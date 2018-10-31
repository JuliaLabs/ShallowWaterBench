#using FastGaussQuadrature.jl
using StaticArrays

#A representation of a T-valued function in an N-Dimensional space.
abstract type Fun{T, N} end

#A discretization of a (T-valued function)-valued function in an N-Dimensional space.
abstract type Basis{T, N, F<:Fun{T, N}} <: AbstractArray{F, N} end

#A function represented by a linear combination of basis functions
struct ComboFun{T, N, B<:Basis{<:Any, N}, C<:AbstractArray{T}} <: Fun{T, N}
    basis::B
    coeffs::C
end

(f::ComboFun)(x) = apply(f, SVector(x))
(f::ComboFun)(x...)  = apply(f, SVector(x...))
#f(x) is just sum_i(c_i * b_i(x))
function apply(f::ComboFun, x::AbstractVector)
    return sum(f.coeffs .* (apply.(f.basis, Ref(x))))
end

#A basis corresponding to a set of points, where the basis function i is one(T) at point i and zero(T) everywhere else
abstract type OrthoBasis{T, N, F} <: Basis{T, N, F} end

for op in (:+, :-)
    @eval begin
        function Base.$(op)(a::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            ComboFun(a.basis, map($op, a.coeffs))
        end
    end
end
for op in (:(Base.:+), :(Base.:-), :(Base.:*), :(Base.:/))
    @eval begin
        function $op(a::ComboFun{T, N, B}, b::ComboFun{S, N, B}) where {T, S, N, B <: OrthoBasis}
            @assert a.basis == b.basis
            ComboFun(a.basis, map($op, a.coeffs, b.coeffs))
        end
        function $op(a::ComboFun{T, N, B}, b) where {T, N, B <: OrthoBasis}
            @assert a.basis == b.basis
            ComboFun(a.basis, map($op, a.coeffs, b))
        end
        function $op(a::ComboFun{T, N, B}, b::Fun) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
        function $op(a, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            @assert a.basis == b.basis
            ComboFun(a.basis, map($op, a.coeffs, b))
        end
        function $op(a::Fun, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
    end
end

#A function which is a product of one-dimensional functions
#
struct ProductFun{T, N, F <: Tuple{Vararg{Fun{T, 1}, N}}} <: Fun{T, N}
    funs::F
    ProductFun(x::Fun{T, 1}) where {T} = new{T, 1, Tuple{typeof(x)}}((x,))
    ProductFun(x::Fun{T, 1}...) where {T} = new{T, length(x), typeof(x)}(x)
end

#ProductFun(x...) = ProductFun(Tuple(x))
(f::ProductFun)(x) = apply(f, SVector(x))
(f::ProductFun)(x...) = apply(f, SVector(x...))
apply(f::ProductFun{T, N}, x::AbstractVector) where {T, N} = prod(apply.(SVector(f.funs), SVector.(x)))

#A basis which is a cartesian product of one-dimensional bases
struct ProductBasis{T, N, F, B <: Tuple{Vararg{OrthoBasis{T, 1, F}, N}}} <: OrthoBasis{T, N, F}
    bases::B
end

Base.size(b::ProductBasis) = Tuple(map(length, b.bases))
Base.getindex(b::ProductBasis, i::CartesianIndex) = ProductFun(map(getindex, b.basis, Tuple(i))...)

#The minimum-degree polynomial function which is 1 at the Rth point and 0 at the other points
struct LagrangeFun{T, P <: AbstractVector{T}, I} <: Fun{T, 1}
    points::P
end

LagrangeFun(points::AbstractVector, i) = LagrangeFun{eltype(points), typeof(points), i}(points)
(f::LagrangeFun)(x) = apply(f, SVector(x))
(f::LagrangeFun)(x...) = apply(f, SVector(x...))
#This method is mostly here for clarity, it probably shouldn't be called (specialize this somewhere with a stable interpolation routine)
apply(f::LagrangeFun, x::AbstractVector) = apply(f, x[1])
function apply(f::LagrangeFun{T, P, I}, x) where {T, P, I}
    @assert length(x) == 1
    res = prod(ntuple(n->n == I ? 1 : (x - f.points[n])/(f.points[I] - f.points[n]), length(f.points)))
end

#A basis of polynomials
struct LagrangeBasis{T, P <: AbstractVector{T}} <: OrthoBasis{T, 1, LagrangeFun{T, P}}
    points::P
end

Base.size(b::LagrangeBasis) = size(b.points)
Base.getindex(b::LagrangeBasis, i::Int) = LagrangeFun(b.points, i)

