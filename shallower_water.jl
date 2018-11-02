using SimpleMeshing
using .Meshing
using .Partitions
using TotallyNotApproxFun
using StaticArrays
using Base.Iterators
using LinearAlgebra

const dim = 2
const order = 3
const hasGPU = Base.find_package("GPUMeshing") !== nothing

if hasGPU
    using GPUMeshing
    backend = GPU()
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

# I⃗      is a function which maps indices to coordinates
# I⃗⁻¹    is a function which maps coordinates to indicies
# X⃗[i]   is a function which maps element coordinates (-1.0 to 1.0) to coordinates
# X⃗⁻¹[i] is a function which maps coordinates to element coordinates (-1.0 to 1.0)

I⃗      = MultilinearFun(I⃗₀, I⃗₁, X⃗₀, X⃗₁)
I⃗⁻¹(x⃗) = ceil(MultilinearFun(X⃗₀, X⃗₁, I⃗₀, I⃗₁))
X⃗      = map(i -> MultilinearFun(-1.0, 1.0, I⃗(i), I⃗(i) + 1), mesh)
X⃗⁻¹    = map(i -> MultilinearFun(I⃗(i), I⃗(i) + 1, -1.0, 1.0), mesh)

# Here is where we construct our basis. In our case, we've chosen an order 3 Lagrange basis over 3 + 1 Lobatto points

ψ = ProductBasis(repeat([LagrangeBasis(LobattoPoints(order))], dim)...)

# Set initial conditions

ψX⃗ = ApproxFun.(X⃗, Ref(ψ))

r = norm.(ψX⃗ .- 0.5)
bathymetry = zero.(r).+0.2
h = 0.5 .* exp.(-100.0 .* r)
U⃗ = zero.(ψX⃗)

using InteractiveUtils
@code_warntype(I((1, 1)))

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
