using MPI

MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
MPI.finalize_atexit()

using SimpleMeshing
using .Meshing
using .Partitions
using Test


const mpicomm = MPI.COMM_WORLD
const myrank = MPI.Comm_rank(mpicomm)
const nprocesses = MPI.Comm_size(mpicomm)

dim = 2

ranks = convert(Array, 0:nprocesses-1)

if isinteger(sqrt(nprocesses))
    a = Int(sqrt(nprocesses))
    ranks = convert(Array, reshape(ranks, (a,a)))
end

if myrank==0
    @show ranks
end

globalInds = CartesianIndices(ntuple(i-> 1:13, dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

mesh = localpart(globalMesh, mpicomm, ranks)

A = storage(Tuple{Int, CartesianIndex{dim}}, mesh)
idxs = CartesianIndices(A)

println("RANK $myrank owns $(first(idxs)) to $(last(idxs))")

map!(x->(myrank, x), A, idxs) # fill elements with their own global index

sync_ghost!(mesh, A)

# It should still be the same
@test A == tuple.(myrank, idxs)

async_send!(mesh)
async_recv!(mesh)
wait_send(mesh)
wait_recv(mesh)

P = CartesianPartition(elems(globalMesh), ranks)
map(boundaries(mesh), ghostboundaries(mesh)) do b, gb
    global_b = translate(globalMesh, gb)
    proc = locate(P, global_b)
    try
        @test A[gb] == tuple.(proc, global_b)
    catch err
        myrank == 0 && rethrow(err)
        exit(1)
    end
end

# Check Idempotency

B = copy(A)

async_send!(mesh)
async_recv!(mesh)
wait_send(mesh)
wait_recv(mesh)

@test A == B

# Reverse send
map(boundaries(mesh), ghostboundaries(mesh)) do b, gb
    A[b] .= A[gb] # prepare to send back what we got above
end

async_send!(mesh)
async_recv!(mesh)
wait_send(mesh)
wait_recv(mesh)

map(ghostboundaries(mesh), boundaries(mesh)) do gb, b
    @test A[b] != tuple.(myrank, b)
    @test A[gb] == tuple.(myrank, b)
end
