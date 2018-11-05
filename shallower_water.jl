using SimpleMeshing
using .Meshing
using .Partitions
using TotallyNotApproxFun
using StaticArrays
using Base.Iterators
using LinearAlgebra
using Test

const dim = 2
const order = 3

if Base.find_package("GPUMeshing") !== nothing
    using GPUMeshing
    backend = GPU()
    GPUMeshing.CuArrays.allowscalar(false)
else
    backend = CPU()
end

mesh = PeriodicCartesianMesh(ntuple(i-> 1:10, dim); backend=backend)

# the whole mesh will go from X⃗₀ to X⃗₁
# (to add a vector arrow to a quantity like `v⃗`, type `v\vec` and then press tab.)
# (subscripts or superscripts like `v₀` can be added with `v\_1` followed by tab.)

X⃗₀ = SVector(2.0, 2.0)
X⃗₁ = SVector(123.0, 100.0)
I⃗₀ = first(elems(mesh))
I⃗₁ = last(elems(mesh))
const Î = one(I⃗₀)

# I⃗⁻¹      is a function which maps indices to coordinates
# I⃗        is a function which maps coordinates to indicies
# X⃗⁻¹[i]   is a function which maps element coordinates (-1.0 to 1.0) to coordinates
# X⃗[i] is a function which maps coordinates to element coordinates (-1.0 to 1.0)

const I⃗⁻¹      = MultilinearFun(I⃗₀, I⃗₁, X⃗₀, X⃗₁)
I⃗(x⃗)    = CartesianIndex(floor.(Int, MultilinearFun(X⃗₀, X⃗₁, I⃗₀, I⃗₁)(x⃗))...)
# test all(I == I⃗⁻¹(I⃗(I)) for I in elems(mesh))
# Due to floating-point imprecisions that does not hold exactly everywhere
X⃗⁻¹    = map(i -> MultilinearFun(-1.0, 1.0, I⃗(i), I⃗(i + Î)), mesh)
X⃗      = map(i -> MultilinearFun(I⃗(i), I⃗(i + Î), -1.0, 1.0), mesh)

# Here is where we construct our basis. In our case, we've chosen an order 3 Lagrange basis over 3 + 1 Lobatto points

Ψ = ProductBasis(repeat([LagrangeBasis(LobattoPoints(order))], dim)...)

# Set initial conditions

approximate(f) = map(i -> approximate(x->f(X⃗[i](x)), Ψ), mesh)

r          = approximate(x⃗ -> norm(x⃗ - 0.5))
bathymetry = approximate(x⃗ -> eltype(x⃗) * 0.2)
h          = approximate(x⃗ -> 0.5 * exp(-100.0 * r(x⃗)))
U⃗          = approximate(x⃗ -> zero(x⃗))
dX⃗         = ∇(X⃗⁻¹[1])
J          = det(dX⃗)

dt = 0.0025
nsteps = ceil(Int64, tend / dt)
dt = tend / nsteps

overelems(mesh) do elem, mesh
    #function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
    ht    = h[elem] + bathymetry[elem]
    u⃗     = U⃗[elem] / ht
    fluxh = U⃗[elem]
    Δh    = ComboFun(∫_Ω.(dot.(∇.(Ψ), dX⃗ * fluxh * J)), Ψ)
    fluxU⃗ = ((u⃗ * u⃗' * ht) + I * 0.5 * gravity * hs^2) * δnl + I * gravity * h[elem] * bathymetry[elem]
    ΔU⃗    = ComboFun(∫_Ω.(dot.(∇.(Ψ), dX⃗ * fluxU⃗ * J)), Ψ)
end

#↑ this stuff works
#↓ this stuff doesn't
#=

ht = h .+ bathymetry
#function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
overelems(mesh) do elem
    J = jacobian(X⃗[elem]) #?

    J = norm(ξ⃗[elem]) #?

    ht = h[elem] + bathymetry[elem]
    u⃗ = U⃗[elem]/ht
    fluxh⃗ = U⃗[elem]
    fluxU⃗ = U⃗[elem] * U⃗[elem]'/ht + I * 0.5g * h[elem]^2 * δnl + I*gravity * h[e] * bathymetry[e]



    Δh[e] = ∫∇Ψ(fluxh⃗[e], d(X⃗[e]))
    Δh[e] = ∫∇Ψ(fluxh⃗[e], I)

#    ΔU⃗ = ∫∇Ψ(fluxU⃗)

    function ∫∇(f::ComboFun{B::Basis, C}, J) where {B, C}
        coeffs = sum(integral(gradient(Psi_i ).*psij) * J)
        return ComboFun(f.basis, 

        for Ψ₁, c in zip(f.basis, f.coeffs)
          for Ψ₂ in f.basis
            res += ?fun fun2

        ω = weights(f.basis)

        

        # loop of ξ-grid lines
        for j = 1:Nq
            rhsh[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxh[1,:,j] + ξy[:,j,e] .* fluxh[2,:,j]))
            rhsU[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxU[1,:,j] + ξy[:,j,e] .* fluxU[2,:,j]))
            rhsV[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxV[1,:,j] + ξy[:,j,e] .* fluxV[2,:,j]))
        end #j
        # loop of η-grid lines
        for i = 1:Nq
            rhsh[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxh[1,i,:] + ηy[i,:,e] .* fluxh[2,i,:]))
            rhsU[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxU[1,i,:] + ηy[i,:,e] .* fluxU[2,i,:]))
            rhsV[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxV[1,i,:] + ηy[i,:,e] .* fluxV[2,i,:]))
        end #i
    end


    J*fluxh⃗ == (ξx[:,j,e] .* fluxh[1,:,j] + ξy[:,j,e] .* fluxh[2,:,j])
    
end #e ∈ elems

#rhsh = zero.(Qh)
#rhsU⃗ = zero.(QU⃗)
#if (advection)
#    δnl=1.0
#    gravity=0.0
#    if dim == 1
#        QU⃗ = Qh .+ bathymetry
#    elseif dim == 2
#        QU⃗ = Qh .+ (bathymetry.*Ref([1.0, 0.0]))
#    end
#end

#plot h now

#we can get the value of h at some position x⃗ by calling h[I⃗⁻¹(x⃗)](X⃗⁻¹(x⃗))
=#
