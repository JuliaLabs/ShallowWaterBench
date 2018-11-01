using SimpleMeshing
using .Meshing
using .Partitions
using TotallyNotApproxFun
using StaticArrays
using Base.Iterators

dim = 2

mesh = PeriodicCartesianMesh(CartesianIndices(ntuple(i-> 1:10, dim)))

# the whole mesh will go from X⃗₀ to X⃗₁
# (to add a vector arrow to a quantity like `v⃗`, type `v\vec` and then press tab.)
# (subscripts or superscripts like `v₀` can be added with `v\_1` followed by tab.)

X⃗₀ = SVector(2.0, 2.0)
X⃗₁ = SVector(123.0, 100.0)

# I⃗      is a function which maps indices to coordinates
# I⃗⁻¹    is a function which maps coordinates to indicies
# X⃗⁻¹[i] is a function which maps our local coordinates (-1.0 to 1.0) to global coordinates
# X⃗[i]   is a function which maps our global coordinates to local coordinates (-1.0 to 1.0)

I⃗      = MultilinearFun(first(elems(mesh)), last(elems(mesh)), X⃗₀, X⃗₁)
I⃗⁻¹(x⃗) = ceil(MultilinearFun(X⃗₀, X⃗₁, first(elems(mesh)), last(elems(mesh)))(x⃗))
X⃗⁻¹    = map(i -> MultilinearFun(-1.0, 1.0, I⃗(i), I⃗(i) + 1), mesh)
X⃗      = map(i -> MultilinearFun(-1.0, 1.0, I⃗(i), I⃗(i) + 1), mesh)

# Here is where we construct our basis. In our case, we've chosen an order 3 Lagrange basis over 3 + 1 Lobatto points

ψ = ProductBasis(ntuple(_->LagrangeBasis(LobattoPoints(3)), dim)...)

# Setting some initial conditions

h = map(Y⃗⁻¹ -> ApproxFun(x⃗ -> (y⃗ = Y⃗⁻¹(x⃗); (y⃗+1).*(y⃗-1)), ψ), X⃗⁻¹)

h = h .* (h .+ 1) * h

#plot h now
