using SimpleMeshing
using .Meshing
using .Partitions
using Test

dim = 2

ranks = convert(Array, reshape(0:(2^dim-1), ntuple(_->2, dim)))

globalInds = CartesianIndices(ntuple(i-> 1:1024, dim))
globalMesh = PeriodicCartesianMesh(CartesianIndices(globalInds))

myrank = 2

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
