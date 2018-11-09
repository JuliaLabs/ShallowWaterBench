using MPI

MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
MPI.finalize_atexit()

using SimpleMeshing
using .Meshing
using .Partitions
using Test


dim = 2

ranks = convert(Array, reshape(0:(2^dim-1), ntuple(_->2, dim)))

globalInds = CartesianIndices(ntuple(i-> 1:1024, dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

const mpicomm = MPI.COMM_WORLD
myrank = MPI.Comm_rank(mpicomm)

P = CartesianPartition(globalInds, ranks)
inds = rankindices(P, myrank)

mesh = GhostCartesianMesh(CPU(), inds)

let z = mpistorage(Complex{Int8}, mesh)
    fill!(z, 42+42im)

    for (bidx, buf) in zip(ghostboundaries(mesh), z.recv_buffers)
        @test z[bidx] != buf
    end

    fill_sendbufs!(z, mesh)

    for (bidx, buf) in zip(ghostboundaries(mesh), z.recv_buffers)
        @test z[bidx] == buf
    end

    for buf in z.recv_buffers
        fill!(buf, 1+1im)
    end

    flush_recvbufs!(z, mesh)
    for (bidx, buf) in ghostboundaries(mesh)
        @test all(z[bidx] .== 1+1im)
    end
end


@show myrank
s = mpistorage(Float64, mesh)
fill!(s, myrank)
P = CartesianPartition(globalInds, ranks)

nbs = map(ghostboundaries(mesh), boundaries(mesh)) do ghostelems, elems
    locate(P, translate(globalMesh, ghostelems))
end

@time for iter = 1:30
    async_recv!(s, mesh, nbs)
    wait_send(s) # iter=1 this is a noop
    overelems(mesh,iter, s) do elem, m, iter, s
        ns = neighbors(elem, m)
        s[elem] = sum(map(j->s[elem] * (iter-1), ns)) + s[elem]
    end
    async_send!(s, mesh, nbs)
    wait_recv(s)
    overelems(mesh,iter, s) do elem,m, iter, s
        s[elem] = s[elem] / iter
    end
end
