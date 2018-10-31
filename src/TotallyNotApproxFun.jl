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
    println(f.coeffs)
    println(f.basis)
    println(Ref(x))
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
for op in (:+, :-, :*, :/)
    @eval begin
        function Base.$(op)(a::ComboFun{T, N, B}, b::ComboFun{S, N, B}) where {T, S, N, B <: OrthoBasis}
            @assert a.basis == b.basis
            ComboFun(a.basis, map($op, a.coeffs, b.coeffs))
        end
    end
end

#A function which is a product of one-dimensional functions
struct ProductFun{T, N, F <: NTuple{N, <:Fun{<:Any, 1}}, Op} <: Fun{T, N}
    op::Op
    funs::F
end

(f::ProductFun)(x) = apply(f, SVector(x))
(f::ProductFun)(x...) = apply(f, SVector(x...))
apply(f::ProductFun{T, N}, x::AbstractVector) where {T, N} = f.op(apply.(f.funs, x))

#A basis which is a cartesian product of one-dimensional bases
struct ProductBasis{T, N, F, B <: NTuple{N, <:OrthoBasis{T, 1, <:F}}} <: OrthoBasis{T, N, F}
    bases::B
end

Base.size(b::ProductBasis) = map(length, b.bases)
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

Base.size(b::LagrangeBasis) = map(length, b.points)
Base.getindex(b::LagrangeBasis, i::Int) = LagrangeFun(b.points, i)

#return a linear function which maps x₀ to y₀ and x₁ to y₁
function repositioner(x₀::AbstractVector{T}, x₁::AbstractVector{T}, y₀::AbstractVector{T}, y₁::AbstractVector{T}) where {N, T}
    ProductFun(Tuple(ComboFun.(LagrangeBasis.(SVector.(x₀, x₁)), SVector.(y₀, y₁))))
end

f = LagrangeFun(SVector(2.0, 4.0), 1)
println(f(1))
g = ComboFun(LagrangeBasis(SVector(2.0, 4.0)), SVector(10, 12))
println(g(1))

using InteractiveUtils
@code_warntype(f(1))

r = repositioner(SVector(2.0, 4.0), SVector(3.0, 5.0), SVector(-1.0, -1.0), SVector(1.0, 1.0))

println(r(2.5, 4.1))










#=
for N = 1:16
    points = gensym()
    @eval begin
        (foo, _) = lglpoints(i)
        const points = SVector(foo)
        
        
        struct LobattoPoints{T, N} <: AbstractVector{T, N} end
        getindex(LobattoPoints{T, N}, i) = $points[i]
    end
end


struct LagrangeFun{T, P <: AbstractVector} <: Fun{T, N}
    points::P
end

struct LagrangeFun{T, P <: AbstractVector} <: Fun{T, N}
    points::P
end

struct LobattoPoints{T, P}

apply(f::ComboFun{T, 1, PolyBasis{S, P}}, x::SVector{1})::T where {T, S, P} = apply(f, x[1])

#Given M Lobatto points on [-1, 1], this is the order M polynomial which is 1 on the I^th point and zero everywhere else
struct LobattoBasis{T, M, I} <: OrthoBasis{T, N, P} end

#Given M Lobatto points on [-1, 1], this is the order M polynomial which is 1 on the I^th point and zero everywhere else
struct LobattoBasis{T, M, I} <: OrthoBasis{T, N, P} end

points(basis::LobattoBasis{T, M, I})

for M in 1:10
    @eval begin
        points(LobattoBasis
        
        (ξ, ω) gausslobatto(7)
        function Base.$(op)(a::ComboFun{T, N, B}, b::ComboFun{S, N, B}) where {T, S, N, B <: OrthoBasis}
            @assert a.basis == b.basis
            ComboFun(a.basis, map($op, a.coeffs, b.coeffs))
        end
    end
end

#Given M Lobatto points on [-1, 1], this is the order M polynomial which is 1 on the I^th point and zero everywhere else
struct LobattoFun{T, M, I} <: Fun{T, 1} end


@generated function
(ξ, ω) gausslobatto(7)


#apply(::LobattoFun, x) = ?
(ξ, ω) = lglpoints(DFloat, N)
D = spectralderivative(ξ)

∫ϕ(f::CartesianLobattoBasis) = 

#we can probably specialize this case.
#apply(f::ComboFun{T, N, M, B}, x::NTuple{N})::T where {T, N, M, B<:InterpolationBasis{M}} = sum(f.coeffs .* apply.(b, (x,)))
#funs(b::InterpolationBasis) = map(x->apply(b, x), points(b))

#A basis corresponding to a cartesian combination of linear Lobatto Basis Functions
struct CartesianLobattoBasis{T, N} <: OrthoBasis{T, N, N, SVector{N, Float64}}
    basis::B
end

#A basis corresponding to a cartesian combination of linear Lobatto Basis Functions
struct CartesianLobattoBasis{T, N, M} <: OrthoBasis{T, N, N, SVector{N, Float64}} end

size(::CartesianLobattoBasis{T, N, M}) = ntuple(n->M + 1, N)

getindex(::CartesianLobattoBasis{T, N, M}, ::CartesianIndex) = 


function Base.show(io::IO, e::RectFunk)
    print(io, "RectFunk containing ")
    show(io, MIME"text/plain"(), e.coeffs)
end

Base.@pure function face2idx(arr, f)
    @assert ndims(arr) == length(f)
    ntuple(ndims(arr)) do i
        dir = f[i]
        if dir == 0
            Colon()
        elseif dir == -1
            1
        elseif dir == 1
            size(arr, i)
        else
            error("Only 1, 0 and -1 are allowed in face index")
        end
    end
end

function face(e::RectFunk, f, m::Mesh)
    e.coeffs[face2idx(e.coeffs, f)...]
end

function face(e::RectFunk, f)
    e.coeffs[face2idx(e.coeffs, f)...]
end

# Volume RHS for 2D
function volumerhs!(h′, V⃗′, h, V⃗, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
    Nq = size(h, 1)
    J = metric.J
    dim=2
    ξ⃗ = metric.ξ⃗
    η⃗ = metric.η⃗
    fluxh = Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxV⃗ = Array{DFloat,3}(undef,dim,Nq,Nq)
    for e ∈ elems
        hb = bathymetry[e]
        hs = h[e]
        ht = hb + hs
        v⃗ = V⃗[e] / ht
        fluxh = V⃗[e]

        fluxV⃗ = ht * v⃗ * 

        ξ⃗ = dΩ

        u=U[:,:,e] ./ H
        v=V[:,:,e] ./ H
        fluxh[1,:,:]=U[:,:,e]
        fluxh[2,:,:]=V[:,:,e]
        fluxU[1,:,:]=(H .* u .* u + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb
        fluxU[2,:,:]=(H .* u .* v) .* δnl
        fluxV[1,:,:]=(H .* v .* u) .* δnl
        fluxV[2,:,:]=(H .* v .* v + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb

        # loop of ξ-grid lines
        for j = 1:Nq
            h′[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxh[1,:,j] + ξy[:,j,e] .* fluxh[2,:,j]))
            rhsU[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxU[1,:,j] + ξy[:,j,e] .* fluxU[2,:,j]))
            V′[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxV[1,:,j] + ξy[:,j,e] .* fluxV[2,:,j]))
        end #j
        # loop of η-grid lines
        for i = 1:Nq
            h′[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxh[1,i,:] + ηy[i,:,e] .* fluxh[2,i,:]))
            rhsU[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxU[1,i,:] + ηy[i,:,e] .* fluxU[2,i,:]))
            V′[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxV[1,i,:] + ηy[i,:,e] .* fluxV[2,i,:]))
        end #i
    end #e ∈ elems
end #function volumerhs-2d
=#
