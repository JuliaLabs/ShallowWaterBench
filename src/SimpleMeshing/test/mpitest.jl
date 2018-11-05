using SimpleMeshing
using .Meshing
using .Partitions

dim = 2

ranks = convert(Array, reshape(0:(2^dim-1), ntuple(_->2, dim)))

globalInds = CartesianIndices(ntuple(i-> 1:1024, dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

myrank = 2

P = CartesianPartition(globalInds, ranks)
inds = rankindices(P, myrank)

mesh = GhostCartesianMesh(CPU(), inds)

z = mpistorage(Complex{Int8}, mesh)
fill!(z, 0+0im)
