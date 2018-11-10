using .Meshing
using .Partitions

"""
    localpart(globalMesh, mpicomm[, ranks=[1:MPI.Comm_size(mpicomm);]'])

Create a localpart of a global mesh

# Arguments

- `globalMesh`: usually a `PeriodicCartesianMesh`
- `mpicomm`: MPI.MPIComm object
- `ranks`: an array showing rank layout. Defaults to [1 2 ... N] where N is the number of workers in mpicomm's pool.

# Returns

a `LocalCartesianMesh` wrapping a `GhostCartesianMesh` with indices offset to represent local part
in the global mesh.
"""
function localpart(globalMesh, mpicomm, ranks=convert(Array, (1:MPI.Comm_rank(mpicomm))'))
    P = CartesianPartition(globalInds, ranks)
    inds = rankindices(P, MPI.Comm_rank(mpicomm))
    mesh = GhostCartesianMesh(CPU(), inds) # TODO: GPU??!
    nbs = map(ghostboundaries(mesh), boundaries(mesh)) do ghostelems, elems
        locate(P, translate(globalMesh, ghostelems))
    end
    LocalCartesianMesh(mesh, nbs, [])
end
