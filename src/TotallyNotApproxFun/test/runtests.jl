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
    F = ComboFun(Ψ, [TotallyNotApproxFun.SVector{dim}(randn(dim)) for i in 1:N, j in 1:N])
    D = TotallyNotApproxFun.spectralderivative(ξ)
    F1 = getindex.(F.coeffs,1)
    F2 = getindex.(F.coeffs,2)
    math∫∇ΨF = [w[j]*F1[:,j]'*(D[:,i].*w)+w[i]*F2[i,:]'*(D[:,j].*w) for i in 1:N, j in 1:N]
    println("Test ∫∇Ψ")
    @test math∫∇ΨF ≈ ∫∇Ψ(F).coeffs

    f = ComboFun(Ψ, randn(N,N))
    math∇f = [TotallyNotApproxFun.SVector(f.coeffs[:,j]'*D[i,:], f.coeffs[i,:]'*D[j,:]) for i in 1:N, j in 1:N]
    println("Test ∇")
    @test math∇f == ∇(f).coeffs

    @test TotallyNotApproxFun.dimsmapslices(1, x -> x .+ 1, @SArray [1  2; 3 4]) === @SArray [2 3; 4 5]
    @test TotallyNotApproxFun.dimsmapslices(2, x -> x .+ 1, @SArray [1  2; 3 4]) === @SArray [2 3; 4 5]
    f(s) = TotallyNotApproxFun.dimsmapslices(2, x -> x .+ 1, s)
    display(@code_typed f(@SArray [1  2; 3 4]))

    @test TotallyNotApproxFun.dimscat(1, (@SArray [1  2]), (@SArray [3  4])) === @SArray [1 2; 3 4]
    @test TotallyNotApproxFun.dimscat(2, (@SArray [1; 3]), (@SArray [2; 4])) === @SArray [1 2; 3 4]
end
