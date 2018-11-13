using Pkg
using Test

subpkgs = ["SimpleMeshing", ]# "StructsOfArrays", "TotallyNotApproxFun"]
const hasGPU = Base.find_package("GPUMeshing") !== nothing

hasGPU && push!(subpkgs, "GPUMeshing")

for subpkg in subpkgs
    Pkg.test(subpkg)
end

examples = [joinpath(@__DIR__, "..", "shallower_water.jl")]

for example in examples
    @info "Testing example" example
    cmd = `$(Base.julia_cmd()) --project=$(Base.current_project()) $example`
    @test success(pipeline(cmd, stderr=stderr))
end


include("reference.jl")
@testset "Correctness check" begin
    include("../shallower_water.jl")
    h_test = map(t->Array(t.coeffs), main()[1:10, 1:10])
    h_ref = Reference.reference()
    @test h_test ≈ h_ref atol=1e-3
end
