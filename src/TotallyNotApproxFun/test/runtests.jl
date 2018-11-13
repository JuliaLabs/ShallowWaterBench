using TotallyNotApproxFun
using Canary
using Test
using StaticArrays

using InteractiveUtils
@testset "mathtest" begin
    N = 5
    dim = 2
    ξ, w = lglpoints(Float64,N-1)
    Ψ = ProductBasis(repeat([LagrangeBasis(LobattoPoints(N-1))], dim)...)
    F = LinearCombinationFun(Ψ, [TotallyNotApproxFun.SVector{dim}(randn(dim)) for i in 1:N, j in 1:N])
    D = TotallyNotApproxFun.spectralderivative(ξ)
    F1 = getindex.(F.coeffs,1)
    F2 = getindex.(F.coeffs,2)
    math∫∇ΨF = [w[j]*F1[:,j]'*(D[:,i].*w)+w[i]*F2[i,:]'*(D[:,j].*w) for i in 1:N, j in 1:N]
    println("Test ∫∇Ψ")
    @test math∫∇ΨF ≈ ∫∇Ψ(F).coeffs

    f = LinearCombinationFun(Ψ, randn(N,N))
    math∇f = [TotallyNotApproxFun.SVector(f.coeffs[:,j]'*D[i,:], f.coeffs[i,:]'*D[j,:]) for i in 1:N, j in 1:N]
    println("Test ∇")
    @test math∇f == ∇(f).coeffs
end

@testset "utils" begin
    @test TotallyNotApproxFun.dimsmapslices(1, x -> x .+ 1, @SArray [1  2; 3 4]) === @SArray [2 3; 4 5]
    @test TotallyNotApproxFun.dimsmapslices(2, x -> x .+ 1, @SArray [1  2; 3 4]) === @SArray [2 3; 4 5]
    f(s) = TotallyNotApproxFun.dimsmapslices(2, x -> x .+ 1, s)
    @inferred f(@SArray [1  2; 3 4])

    @test TotallyNotApproxFun.dimscat(1, (@SArray [1  2]), (@SArray [3  4])) === @SArray [1 2; 3 4]
    @test TotallyNotApproxFun.dimscat(2, (@SArray [1; 3]), (@SArray [2; 4])) === @SArray [1 2; 3 4]
end

@testset "funs" begin

    X⃗₀ = SVector(0.0, 0.0)
    X⃗₁ = SVector(1.0, 1.0)
    I⃗₀ = CartesianIndex(1,1)
    I⃗₁ = CartesianIndex(10,10)
    Î = one(I⃗₀)

    # I⃗⁻¹      is a function which maps indices to coordinates
    # I⃗        is a function which maps coordinates to indicies
    # X⃗⁻¹[i]   is a function which maps element coordinates (-1.0 to 1.0) to coordinates
    # X⃗[i] is a function which maps coordinates to element coordinates (-1.0 to 1.0)

    I⃗⁻¹      = MultilinearFun(I⃗₀, I⃗₁ + Î, X⃗₀, X⃗₁)
    I⃗(x⃗)    = CartesianIndex(floor.(Int, MultilinearFun(X⃗₀, X⃗₁, I⃗₀, I⃗₁ + Î)(x⃗))...)
    # test all(I == I⃗⁻¹(I⃗(I)) for I in elems(mesh))
    # Due to floating-point imprecisions that does not hold exactly everywhere
    X⃗⁻¹    = map(i -> MultilinearFun((-1.0, -1.0), (1.0, 1.0), I⃗⁻¹(i), I⃗⁻¹(i + Î)), CartesianIndices((1:10, 1:10)))
    X⃗      = map(i -> MultilinearFun(I⃗⁻¹(i), I⃗⁻¹(i + Î), (-1.0, -1.0), (1.0, 1.0)), CartesianIndices((1:10, 1:10)))

    @test I⃗⁻¹(1.5, 2.5) ≈ @SVector [0.05, 0.15]
    @test I⃗(@SVector [0.05, 0.15]) == CartesianIndex(1, 2)
    @test X⃗[2, 3](0.15, 0.25) ≈ @SVector [0.0, 0.0]
    @test X⃗[2, 3](0.2, 0.3) ≈ @SVector [1.0, 1.0]
    @test X⃗[2, 3](0.1, 0.2) ≈ @SVector [-1.0, -1.0]
    @test X⃗⁻¹[2, 3](0.0, 0.0) ≈ @SVector [0.15, 0.25]
    @test X⃗⁻¹[2, 3](1.0, 1.0) ≈ @SVector [0.2, 0.3]
    @test X⃗⁻¹[2, 3](-1.0, -1.0) ≈ @SVector [0.1, 0.2]
end

