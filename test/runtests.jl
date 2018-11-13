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


@testset "Correctness check" begin
    let
        include("../src/shallow_water.jl")
        EtoC = [Int.((mesh.elemtocoord.*brickN[1])[:,4,i]) for i in 1:prod(brickN)]
        global h1 = [Q.h[:,:,EtoC[i,j]] for i in 1:brickN[1], j in 1:brickN[2]]
    end
    let
        include("../shallower_water.jl")
        x = h_from_us
        global h2 = map(t->Array(t.coeffs), x)
    end
    @test h1 == h2
end
