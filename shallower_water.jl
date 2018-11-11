using SimpleMeshing
using .Meshing
using .Partitions
using TotallyNotApproxFun
using StaticArrays
using Base.Iterators
using LinearAlgebra
using Test
using MPI

const dim = 2
const order = 4

if Base.find_package("GPUMeshing") !== nothing
    using GPUMeshing
    backend = GPU()
    GPUMeshing.CuArrays.allowscalar(false)
else
    backend = CPU()
end

MPI.Initialized() || MPI.Init()
MPI.finalize_atexit()

const mpicomm = MPI.COMM_WORLD

function main(tend=0.32, backend=backend)
    params = setup(backend)
    compute(tend, params...)
end

function setup(backend)
    # Create a CPU mesh
    globalMesh = PeriodicCartesianMesh(ntuple(i-> 1:10, dim); backend=backend)
    mesh = localpart(globalMesh, mpicomm)

    # the whole mesh will go from X⃗₀ to X⃗₁
    # (to add a vector arrow to a quantity like `v⃗`, type `v\vec` and then press tab.)
    # (subscripts or superscripts like `v₀` can be added with `v\_1` followed by tab.)

    X⃗₀ = SVector(0.0, 0.0)
    X⃗₁ = SVector(1.0, 1.0)
    I⃗₀ = first(elems(mesh))
    I⃗₁ = last(elems(mesh))
    Î = one(I⃗₀)

    # I⃗⁻¹      is a function which maps indices to coordinates
    # I⃗        is a function which maps coordinates to indicies
    # X⃗⁻¹[i]   is a function which maps element coordinates (-1.0 to 1.0) to coordinates
    # X⃗[i] is a function which maps coordinates to element coordinates (-1.0 to 1.0)

    I⃗⁻¹      = MultilinearFun(I⃗₀, I⃗₁ + Î, X⃗₀, X⃗₁)
    I⃗(x⃗)    = CartesianIndex(floor.(Int, MultilinearFun(X⃗₀, X⃗₁, I⃗₀, I⃗₁ + Î)(x⃗))...)
    # test all(I == I⃗⁻¹(I⃗(I)) for I in elems(mesh))
    # Due to floating-point imprecisions that does not hold exactly everywhere
    X⃗⁻¹    = map(i -> MultilinearFun((-1.0, -1.0), (1.0, 1.0), I⃗⁻¹(i), I⃗⁻¹(i + Î)), mesh)
    X⃗      = map(i -> MultilinearFun(I⃗⁻¹(i), I⃗⁻¹(i + Î), (-1.0, -1.0), (1.0, 1.0)), mesh)

    # Here is where we construct our basis. In our case, we've chosen an order 3 Lagrange basis over 3 + 1 Lobatto points
    Ψ = ProductBasis(ntuple(i->LagrangeBasis(LobattoPoints(order)), dim)...)

    # Set initial conditions

    myapproximate(f) = map(i -> approximate(x->f(X⃗⁻¹[i](x)), Ψ), mesh)
    myinterpolate(f, x⃗) = f[I⃗⁻¹(x⃗)](X⃗⁻¹(x⃗))

    bathymetry = myapproximate(x⃗ -> 0.2)
    h          = myapproximate(x⃗ -> 0.5 * exp(-100.0 * norm(x⃗ - 0.5)^2))
    U⃗          = myapproximate(x⃗ -> zero(x⃗))
    Δh         = myapproximate(x⃗ -> zero(eltype(x⃗)))
    ΔU⃗         = myapproximate(x⃗ -> zero(x⃗))
    dX⃗         = ∇(X⃗[1])(zero(Î))
    J          = inv(det(dX⃗))
    gravity    = 10.0
    #sJ         = det(∇(X⃗⁻¹[1][face]))

    elem₁ = first(elems(mesh))
    faces₁ = faces(elem₁, mesh)
    Jfaces = SVector{length(faces₁)}([norm(∇(X⃗[elem₁][face])(zero(Î))) for face in faces₁])

    M = ∫Ψ(approximate(x⃗ -> J, Ψ))

    params = (mesh, h, bathymetry, U⃗, Δh, ΔU⃗, J, gravity, X⃗, dX⃗, Î, Ψ, Jfaces, M)

    # todo adapt to backend and reconstruct mesh
    return params
end

function compute(tend, mesh, h, bathymetry, U⃗, Δh, ΔU⃗, J, gravity, X⃗, dX⃗, Î, Ψ, Jfaces, M)
    ## Probably T instead of Float64?
    RKA = (Float64(0),
           Float64(-567301805773)  / Float64(1357537059087),
           Float64(-2404267990393) / Float64(2016746695238),
           Float64(-3550918686646) / Float64(2091501179385),
           Float64(-1275806237668) / Float64(842570457699 ))

    RKB = (Float64(1432997174477) / Float64(9575080441755 ),
           Float64(5161836677717) / Float64(13612068292357),
           Float64(1720146321549) / Float64(2090206949498 ),
           Float64(3134564353537) / Float64(4481467310338 ),
           Float64(2277821191437) / Float64(14882151754819))

    RKC = (Float64(0),
           Float64(1432997174477) / Float64(9575080441755),
           Float64(2526269341429) / Float64(6820363962896),
           Float64(2006345519317) / Float64(3224310063776),
           Float64(2802321613138) / Float64(2924317926251))

    dt = 0.001
    nsteps = ceil(Int64, tend / dt)
    dt = tend / nsteps

    sync_storage!(mesh, h)
    sync_storage!(mesh, bathymetry)
    sync_storage!(mesh, U⃗)

    for step in 1:nsteps
        for s in 1:length(RKA)

            # Volume integral
            overelems(mesh, h, bathymetry, U⃗, Δh, ΔU⃗) do elem, mesh, h, bathymetry, U⃗, Δh, ΔU⃗
                #function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
                ht         = h[elem] + bathymetry[elem]
                u⃗          = U⃗[elem] / ht
                fluxh      = U⃗[elem]
                Δh[elem]  += ∫∇Ψ(dX⃗ * fluxh * J)
                fluxU⃗      = (u⃗ * u⃗' * ht) + I * gravity * (0.5 * h[elem]^2 + h[elem] * bathymetry[elem])
                ΔU⃗[elem]  += ∫∇Ψ(dX⃗ * fluxU⃗ * J)
            end

            # Flux integral
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

            # Update steps
            overelems(mesh, h, bathymetry, U⃗, Δh, ΔU⃗) do elem, mesh, h, bathymetry, U⃗, Δh, ΔU⃗
                ## Assuming advection == false
                h[elem] += RKB[s] * dt * Δh[elem] / M
                U⃗[elem] += RKB[s] * dt * ΔU⃗[elem] / M
                Δh[elem] *= RKA[s]
                ΔU⃗[elem] *= RKA[s]
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
