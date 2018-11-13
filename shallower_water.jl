using SimpleMeshing
using .Meshing
using .Partitions
using TotallyNotApproxFun
using StaticArrays
using Base.Iterators
using LinearAlgebra
using Test
using MPI
include("constants.jl")


const dim     = parse(Int, get(ENV, "SHALLOW_WATER_DIM", "2"))
const simsize = parse(Int, get(ENV, "SHALLOW_WATER_SIZE", "10"))
const tend    = parse(Float64, get(ENV, "SHALLOW_WATER_TEND", "0.32"))
const base_dt = parse(Float64, get(ENV, "SHALLOW_WATER_DT", "0.001"))
const order   = 3


function simulate(tend, mesh, h, bathymetry, U⃗, Δh, ΔU⃗, J, g, X⃗, dX⃗, Î, Ψ, face_Js, M)
    nsteps = ceil(Int64, tend / base_dt)
    dt = tend / nsteps

    for step in 1:nsteps
        for s in 1:length(RKA)

            async_send!(mesh)
            async_recv!(mesh)
            wait_send(mesh) # iter=1 this is a noop

            # Volume integral
            overelems(mesh, h, bathymetry, U⃗, Δh, ΔU⃗) do elem, mesh, h, bathymetry, U⃗, Δh, ΔU⃗
            @inbounds begin
                hbₑ        = bathymetry[elem]
                htₑ        = h[elem] + hbₑ
                U⃗ₑ         = U⃗[elem]

                Δh[elem]  += ∫∇Ψ(dX⃗ * U⃗ₑ * J)
                ΔU⃗[elem]  += ∫∇Ψ(dX⃗ * (U⃗ₑ * U⃗ₑ' / htₑ + g * (htₑ^2 - hbₑ^2)/2 * I) * J)
            end
            end

            wait_recv(mesh) # fill in data from previous iteration

            # Flux integral
            overelems(mesh, h, bathymetry, U⃗, Δh, ΔU⃗) do elem, mesh, h, bathymetry, U⃗, Δh, ΔU⃗
            @inbounds begin
                Δhₑ = ComboFun(Δh[elem].basis, MArray(Δh[elem].coeffs))
                ΔU⃗ₑ = ComboFun(ΔU⃗[elem].basis, MArray(ΔU⃗[elem].coeffs))
                for (face, face_J) in zip(faces(elem, mesh), face_Js)
                    other_elem =  neighbor(elem, face, mesh)
                    other_face =  opposite(face, other_elem, mesh)

                    hₑ         =  h[elem][face]
                    hbₑ        =  bathymetry[elem][face]
                    htₑ        =  hₑ + hbₑ
                    U⃗ₑ         =  U⃗[elem][face]

                    other_hₑ   =  h[other_elem][other_face]
                    other_hbₑ  =  bathymetry[other_elem][other_face]
                    other_htₑ  =  hₑ + hbₑ
                    other_U⃗ₑ   =  U⃗[other_elem][other_face]


                    λ          =  max( abs( normal(face)' *       U⃗ₑ/      htₑ) + sqrt(g *       htₑ),
                                       abs(-normal(face)' * other_U⃗ₑ/other_htₑ) + sqrt(g * other_htₑ))

                    Δhₑ[face]  -= ∫Ψ(((U⃗ₑ + other_U⃗ₑ)' * normal(face) - λ * (other_hₑ - hₑ)) / 2 * face_J)

                    flux       =  (      U⃗ₑ *       U⃗ₑ' /       htₑ + g * (      htₑ^2 -       hbₑ^2)/2 * I)
                    other_flux =  (other_U⃗ₑ * other_U⃗ₑ' / other_htₑ + g * (other_htₑ^2 - other_hbₑ^2)/2 * I)
                    ΔU⃗ₑ[face]  -= ∫Ψ(((flux + other_flux)' * normal(face) - λ * (other_U⃗ₑ - U⃗ₑ)) / 2 * face_J)
                end
                Δh[elem] = ComboFun(Δhₑ.basis, SArray(Δhₑ.coeffs))
                ΔU⃗[elem] = ComboFun(ΔU⃗ₑ.basis, SArray(ΔU⃗ₑ.coeffs))
            end
            end

            # Update steps
            rka = RKA[s % length(RKA) + 1]
            rkb = RKB[s]
            overelems(mesh, h, U⃗, Δh, ΔU⃗, M, rka, rkb, dt) do elem, mesh, h, U⃗, Δh, ΔU⃗, M, rka, rkb, dt
            @inbounds begin
                ## Assuming advection == false
                h[elem] += rkb * dt * Δh[elem] / M
                U⃗[elem] += rkb * dt * ΔU⃗[elem] / M
                Δh[elem] *= rka
                ΔU⃗[elem] *= rka
            end
            end
        end
    end
    return h
end

if Base.find_package("GPUMeshing") !== nothing
    using GPUMeshing
    backend = GPU()
    GPUMeshing.CuArrays.allowscalar(false)
    adapt(x) = GPUMeshing._adapt(x)
else
    backend = CPU()
    adapt(x) = x
end

MPI.Initialized() || MPI.Init()
MPI.finalize_atexit()

const mpicomm = MPI.COMM_WORLD

function main(tend=tend, backend=backend)
    params = setup(backend)
    simulate(tend, params...)
end

function setup(backend)
    # Create a CPU mesh
    println("Starting shallower sim. dim=$dim; simsize=$simsize; tend=$tend; base_dt=$base_dt")
    globalMesh = PeriodicCartesianMesh(ntuple(i-> 1:simsize, dim); backend=backend)
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
    dX⃗         = ∇(X⃗[I⃗₀])(zero(Î))
    J          = inv(det(dX⃗))
    g    = 10.0
    #sJ         = det(∇(X⃗⁻¹[1][face]))

    # Keep these 3 arrays in sync across workers
    sync_ghost!(mesh, h)
    sync_ghost!(mesh, bathymetry)
    sync_ghost!(mesh, U⃗)

    elem₁ = first(elems(mesh))
    faces₁ = faces(elem₁, mesh)
    face_Js = SVector{length(faces₁)}([norm(∇(X⃗⁻¹[elem₁][face])(zero(Î))) for face in faces₁])

    M = ∫Ψ(approximate(x⃗ -> J, Ψ))

    params = (mesh, h, bathymetry, U⃗, Δh, ΔU⃗, J, g, X⃗, dX⃗, Î, Ψ, face_Js, M)

    return map(adapt, params)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
