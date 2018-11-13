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
        EtoC = [round.(Int, (mesh.elemtocoord.*10)[:,4,i]) for i = 1:(10 * 10)]
        RefC = zeros(Int64, brickN[1], brickN[2])
        for t in 1:prod(brickN)
           RefC[EtoC[t][1], EtoC[t][2]] = t
        end
        global h1 = [h[:,:,RefC[i,j]] for i in 1:10, j in 1:10]
    end
    let
        include("../shallower_water.jl")
        global h2 = map(t->Array(t.coeffs), main()[1:10, 1:10])
    end
    @test h1 ≈ h2 atol=1e-3
end
