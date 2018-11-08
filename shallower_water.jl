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

function main(;backend=backend)
    mesh = PeriodicCartesianMesh(ntuple(i-> 1:10, dim); backend=backend)

    # the whole mesh will go from X⃗₀ to X⃗₁
    # (to add a vector arrow to a quantity like `v⃗`, type `v\vec` and then press tab.)
    # (subscripts or superscripts like `v₀` can be added with `v\_1` followed by tab.)

    X⃗₀ = SVector(2.0, 2.0)
    X⃗₁ = SVector(123.0, 100.0)
    I⃗₀ = first(elems(mesh))
    I⃗₁ = last(elems(mesh))
    Î = one(I⃗₀)

    # I⃗⁻¹      is a function which maps indices to coordinates
    # I⃗        is a function which maps coordinates to indicies
    # X⃗⁻¹[i]   is a function which maps element coordinates (-1.0 to 1.0) to coordinates
    # X⃗[i] is a function which maps coordinates to element coordinates (-1.0 to 1.0)

    I⃗⁻¹      = MultilinearFun(I⃗₀, I⃗₁, X⃗₀, X⃗₁)
    I⃗(x⃗)    = CartesianIndex(floor.(Int, MultilinearFun(X⃗₀, X⃗₁, I⃗₀, I⃗₁)(x⃗))...)
    # test all(I == I⃗⁻¹(I⃗(I)) for I in elems(mesh))
    # Due to floating-point imprecisions that does not hold exactly everywhere
    X⃗⁻¹    = map(i -> MultilinearFun((-1.0, -1.0), (1.0, 1.0), I⃗(i), I⃗(i + Î)), mesh)
    X⃗      = map(i -> MultilinearFun(I⃗(i), I⃗(i + Î), (-1.0, -1.0), (1.0, 1.0)), mesh)

    # Here is where we construct our basis. In our case, we've chosen an order 3 Lagrange basis over 3 + 1 Lobatto points

    Ψ = ProductBasis(repeat([LagrangeBasis(LobattoPoints(order))], dim)...)

    # Set initial conditions

    myapproximate(f) = map(i -> approximate(x->f(X⃗[i](x)), Ψ), mesh)
    myinterpolate(f, x⃗) = f[I⃗⁻¹(x⃗)](X⃗⁻¹(x⃗))

    r          = myapproximate(x⃗ -> norm(x⃗ - 0.5))
    bathymetry = myapproximate(x⃗ -> 0.2)
    h          = myapproximate(x⃗ -> 0.5 * exp(-100.0 * norm(x⃗ - 0.5)))
    U⃗          = myapproximate(x⃗ -> zero(x⃗))
    Δh         = myapproximate(x⃗ -> zero(eltype(x⃗)))
    ΔU⃗         = myapproximate(x⃗ -> zero(x⃗))
    dX⃗         = ∇(X⃗⁻¹[1])(zero(Î))
    J          = det(dX⃗)
    gravity    = 10.0
    #sJ         = det(∇(X⃗⁻¹[1][face]))

    #dt = 0.0025
    #nsteps = ceil(Int64, tend / dt)
    #dt = tend / nsteps

    overelems(mesh, h, bathymetry, U⃗, Δh, ΔU⃗) do elem, mesh, h, bathymetry, U⃗, Δh, ΔU⃗
        #function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
        ht         = h[elem] + bathymetry[elem]
        u⃗          = U⃗[elem] / ht
        fluxh      = U⃗[elem]
        Δh[elem]  += ∫∇Ψ(dX⃗ * fluxh * J)
        fluxU⃗      = (u⃗ * u⃗' * ht) + I * gravity * (0.5 * h[elem]^2 + h[elem] * bathymetry[elem])
        ΔU⃗[elem]  += ∫∇Ψ(dX⃗ * fluxU⃗ * J)
    end

    elem₁ = first(elems(mesh))
    faces₁ = faces(elem₁, mesh)
    Jfaces = SVector{length(faces₁)}([norm(∇(X⃗[elem₁][face])(zero(Î))) for face in faces₁])

    overelems(mesh, h, bathymetry, U⃗, Δh, ΔU⃗) do elem, mesh, h, bathymetry, U⃗, Δh, ΔU⃗
        myΔh = ComboFun(Δh[elem].basis, MArray(Δh[elem].coeffs))
        myΔU⃗ = ComboFun(ΔU⃗[elem].basis, MArray(ΔU⃗[elem].coeffs))
        for (face, Jface) in zip(faces(elem, mesh), Jfaces)
            elem′ = neighbor(elem, face, mesh)
            face′ = opposite(face, elem′, mesh)

            hs = h[elem][face]
            hb = bathymetry[elem][face]

            ht        = hs + hb
            u⃗         = U⃗[elem][face] / ht
            fluxh     = U⃗[elem][face]
            fluxU⃗     = (u⃗ * u⃗' * ht) + I * gravity * (0.5 * hs^2 + hs * hb)
            λ         = abs(normal(face)' * u⃗) + sqrt(gravity * hs)

            hs′ = h[elem′][face′]
            hb′ = bathymetry[elem′][face′]

            ht′        = hs′ + hb′
            u⃗′         = U⃗[elem′][face′] / ht′
            fluxh′     = U⃗[elem′][face′]
            fluxU⃗′     = (u⃗′ * u⃗′' * ht′) + I * gravity * (0.5 * hs′^2 + hs′ * hb′)
            λ′         = abs(normal(face′)' * u⃗′) + sqrt(gravity * hs′)

            myΔh[face] -= ∫Ψ(((fluxh + fluxh′)' * normal(face) - (max( λ, λ′ ) * (hs′ - hs)) / 2) * Jface)
            myΔU⃗[face] -= ∫Ψ(((fluxU⃗ + fluxU⃗′)' * normal(face) - (max( λ, λ′ ) * (U⃗[elem′][face′] - U⃗[elem][face])) / 2) * Jface)
        end
        Δh[elem] = ComboFun(myΔh.basis, SArray(myΔh.coeffs))
        ΔU⃗[elem] = ComboFun(myΔU⃗.basis, SArray(myΔU⃗.coeffs))
    end

    rkb = 1.0
    rka = 1.0
    dt = 1.0
    overelems(mesh, h, bathymetry, U⃗, Δh, ΔU⃗) do elem, mesh, h, bathymetry, U⃗, Δh, ΔU⃗
        ht = h[elem] + bathymetry[elem]
        u⃗ = U⃗[elem] / ht

        M = ∫Ψ(approximate(x⃗ -> J, Ψ))
        for e ∈  neighbors(elem, mesh)
            h[elem] += rkb * dt * Δh[elem] / M
            U⃗[elem] += rkb * dt * ΔU⃗[elem] / M
            Δh[elem] *= rka
            ΔU⃗[elem] *= rka
        end

        U⃗[elem] = (h[elem]+bathymetry[elem]) * u⃗
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
