using SimpleMeshing
using .Meshing
using .Partitions
using TotallyNotApproxFun

dim = 2

mesh = PeriodicCartesianMesh(CartesianIndices(ntuple(i-> 1:10, dim)))
inds = elemindices(mesh)

# the whole mesh will go from X⃗₀ to X⃗₁
# (to add a vector arrow to a quantity like `v⃗`, type `v\vec` and then press tab.)
# (subscripts like `v₀` can be added with `v\_1` followed by tab.)

X⃗₀ = SVector(2.0, 2.0)
X⃗₁ = SVector(100.0, 100.0)

# I⃗ is a function which maps indices to coordinates
I⃗ = RepositionFun(first(inds), last(inds), X⃗₀, X⃗₁)

# I⃗⁻¹ is a function which maps coordinates to indicies
I⃗⁻¹(x⃗) = ceil(RepositionFun(X⃗₀, X⃗₁, first(inds), last(inds))(x⃗))

# X⃗⁻¹ is a function which maps our local coordinates (-1.0 to 1.0) to global coordinates
X⃗⁻¹  = map(i -> RepositionFun(-1.0, 1.0, I⃗(i), I⃗(i) + 1), inds)

# X⃗ is a function which maps our global coordinates to local coordinates (-1.0 to 1.0)
X⃗ = map(i -> RepositionFun(-1.0, 1.0, I⃗(i), I⃗(i) + 1), inds)

h = map(Y⃗⁻¹ -> ApproxFun(x⃗ -> (y⃗ = Y⃗⁻¹(x⃗); (y⃗+1).*(y⃗-1)), LobattoBasis), X⃗⁻¹)

h = h .* (h .+ 1) * h

#plot h now
