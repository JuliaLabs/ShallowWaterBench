module TotallyNotApproxFun

using FastGaussQuadrature
using StaticArrays
using Base.Iterators

export Fun, ComboFun, ApproxFun, ProductFun, LagrangeFun, VectorFun, MultilinearFun
export Basis, OrthoBasis, ProductBasis, LagrangeBasis
export points, LobattoPoints

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

#We'll work on this.
value(f) = f(0)

#A discretization of a (T-valued function)-valued function in an N-Dimensional space.
abstract type Basis{T, N, F<:Fun{T, N}} <: AbstractArray{F, N} end

#A function represented by a linear combination of basis functions
struct ComboFun{T, N, B<:Basis{<:Any, N}, C<:AbstractArray{T}} <: Fun{T, N}
    basis::B
    coeffs::C
end

(f::ComboFun)(x...) = apply(f, SVector(x...))
#f(x) is just sum_i(c_i * b_i(x))
function apply(f::ComboFun, x::AbstractVector)
    return sum(f.coeffs .* (apply.(f.basis, Ref(x))))
end

#A basis corresponding to a set of points, where the basis function i is one(T) at point i and zero(T) everywhere else
abstract type OrthoBasis{T, N, F} <: Basis{T, N, F} end
points(::OrthoBasis) = error("Subtypes of OrthoBasis must define the points function")
ApproxFun(f, b::OrthoBasis) = ComboFun(b, map(f, points(b)))

for op in (:+, :-, :transpose)
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
            ComboFun(a.basis, $op.(a.coeffs, b.coeffs))
        end
        function $op(a::ComboFun{T, N, B}, b) where {T, N, B <: OrthoBasis}
            ComboFun(a.basis, $op.(a.coeffs, b))
        end
        function $op(a::ComboFun{T, N, B}, b::Fun) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
        function $op(a, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            ComboFun(a.basis, $op.(a.coeffs, b))
        end
        function $op(a::Fun, b::ComboFun{T, N, B}) where {T, N, B <: OrthoBasis}
            throw(NotImplementedError())
        end
    end
end

#A function which is a product of one-dimensional functions
struct ProductFun{T, N, F <: Tuple{Vararg{Fun{T, 1}, N}}} <: Fun{T, N}
    funs::F
    ProductFun(funs::Fun{T, 1}) where {T} = new{T, 1, Tuple{typeof(funs)}}((funs,))
    ProductFun(funs::Fun{T, 1}...) where {T} = new{T, length(funs), typeof(funs)}(funs)
end

(f::ProductFun)(x...) = apply(f, SVector(x...))
function apply(f::ProductFun, x::AbstractVector)
    prod(apply.(SVector(f.funs), SVector.(x)))
end

#A basis which is a cartesian product of one-dimensional bases
struct ProductBasis{T, N, B <: Tuple{Vararg{OrthoBasis{T, 1}, N}}} <: OrthoBasis{T, N, Fun{T, N}}
    bases::B
    ProductBasis(basis::OrthoBasis{T, 1}) where {T} = new{T, 1, Tuple{typeof(basis)}}((basis,))
    ProductBasis(bases::OrthoBasis{T, 1}...) where {T} = new{T, length(bases), typeof(bases)}(bases)
end

Base.size(b::ProductBasis) = map(length, b.bases)
Base.eltype(b::ProductBasis{T, N}) where {T, N} = ProductFun{T, N, Tuple{eltype.(b.bases)...}}
Base.getindex(b::ProductBasis, i::Int...)::eltype(b) = ProductFun(map(getindex, b.bases, i)...)
points(b::ProductBasis) = SArray{Tuple{length.(points.(b.bases))...}}(SVector.(product(points.(b.bases)...))) #TODO specialize `product` for static sizes instead of this function

#The minimum-degree polynomial function which is 1 at the nth point and 0 at the other points
struct LagrangeFun{T, P <: AbstractVector{T}} <: Fun{T, 1}
    points::P
    n::Int
end

LagrangeFun(points::AbstractVector, n) = LagrangeFun{eltype(points), typeof(points)}(points, n)
(f::LagrangeFun)(x...) = apply(f, SVector(x...))
#This method is mostly here for clarity, it probably shouldn't be called (TODO specialize somewhere with a stable interpolation routine)
function apply(f::LagrangeFun, x::AbstractVector)
    @assert length(x) == 1
    res = prod(ntuple(i->i == f.n ? 1 : (x[1] - f.points[i])/(f.points[f.n] - f.points[i]), length(f.points)))
end

#A basis of polynomials
struct LagrangeBasis{T, P <: AbstractVector{T}} <: OrthoBasis{T, 1, LagrangeFun{T, P}}
    points::P
end

Base.size(b::LagrangeBasis) = size(b.points)
Base.getindex(b::LagrangeBasis, i::Int) = LagrangeFun(b.points, i)
points(b::LagrangeBasis) = b.points

#A vector representing Lobatto Points
struct LobattoPoints{T, N} <: AbstractVector{T} end

Base.size(p::LobattoPoints{T, N}) where {T, N} = (N,)
@generated function Base.getindex(p::LobattoPoints{T, N}, i::Int) where {T, N}
    return :($(SVector{N, T}(gausslobatto(N)[1]))[i])
end
LobattoPoints(n) = LobattoPoints{Float64, n + 1}()

#=
#A vector-valued array composed of one-dimensional arrays
struct VectorArray{T, N, A <: Tuple{Vararg{AbstractVector{T}, N}}} <: AbstractArray{SVector{N, T}, N}
    data::A
    VectorArray(x::AbstractVector{T, 1}) where {T} = new{T, 1, Tuple{typeof(x)}}((x,))
    VectorArray(x::AbstractVector{T, 1}...) where {T} = new{T, length(x), typeof(x)}(x)
end

(f::VectorFun)(x) = apply(f, SVector(x))
(f::VectorFun)(x::AbstractVector) = apply(f, x)
(f::VectorFun)(x...) = apply(f, SVector(x...))
function apply(f::VectorFun, x::AbstractVector)
    apply.(SVector(f.funs), SVector.(x))
end
=#

#A vector-valued function composed of one-dimensional functions
struct VectorFun{T, N, F <: Tuple{Vararg{Fun{T, 1}, N}}} <: Fun{SVector{N, T}, N}
    funs::F
    VectorFun(x::Fun{T, 1}) where {T} = new{T, 1, Tuple{typeof(x)}}((x,))
    VectorFun(x::Fun{T, 1}...) where {T} = new{T, length(x), typeof(x)}(x)
end

(f::VectorFun)(x...) = apply(f, SVector(x...))
function apply(f::VectorFun, x::AbstractVector)
    apply.(SVector(f.funs), SVector.(x))
end

#WARNING DEFINING AN SARRAY METHOD
StaticArrays.SVector(i::CartesianIndex) = SVector(Tuple(i))

function MultilinearFun(x₀, x₁, y₀, y₁)
    x₀, x₁, y₀, y₁ = map(Tuple, (x₀, x₁, y₀, y₁))
    VectorFun(ComboFun.(LagrangeBasis.(SVector.(x₀, x₁)), SVector.(y₀, y₁))...)
end

#r = repositioner(SVector(2.0, 4.0), SVector(3.0, 5.0), SVector(-1.0, -1.0), SVector(1.0, 1.0))

#println(r([3.4, 4.5]))
#using InteractiveUtils
#@code_warntype(r([3.4, 4.5]))











#1 function to interpolate global coefficients to local -1 to 1 for basis
#  a) only store scale in type domain
#  b) store scale and offset

#2 implement ∫ψ(f, ω, J) and ∫∇ψ(f) on combination functions (and handle the fact that f is repositioned) (f = f(reposition(x)))

#3 implement ∫ and ∫∇ on combination functions (and handle the fact that f is repositioned) (f = f(reposition(x)))

end
