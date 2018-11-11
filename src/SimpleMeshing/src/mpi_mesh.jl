using MPI
using ..Partitions

export BufferedArray, localpart, neighbor_ranks, sync_storage!,
       fill_sendbufs!, flush_recvbufs!, async_send!, async_recv!,
       wait_send, wait_recv

# TODO: get rid of this type and store all buffers in the LocalCartesianMesh?

"""
    LocalCartesianMesh(G, neighbors)
"""
struct LocalCartesianMesh{N, B, G<:CartesianMesh{N, B}} <: CartesianMesh{N, B}
    mesh::G
    neighbor_ranks::Tuple{Vararg{Int}}
    synced_storage::Vector{Any}
end

struct BufferedArray{T, N, A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    arr::A
    recv_buffers::Tuple
    send_buffers::Tuple
    recv_reqs::Vector{MPI.Request}
    send_reqs::Vector{MPI.Request}
end

Base.size(a::BufferedArray) = size(a.arr)
Base.getindex(a::BufferedArray, idx...) = a.arr[idx...]
Base.setindex!(a::BufferedArray, val, idx...) = a.arr[idx...] = val
Base.axes(a::BufferedArray) = axes(a.arr) # now includes offsets
Base.IndexStyle(a::BufferedArray) = IndexStyle(a.arr)

function flush_recvbufs!(a::BufferedArray, mesh::LocalCartesianMesh)
    for (bidx, buf) in zip(ghostboundaries(mesh), a.recv_buffers)
        a[bidx] .= buf
    end
end

function fill_sendbufs!(a::BufferedArray, mesh::LocalCartesianMesh)
    for (bidx, buf) in zip(boundaries(mesh), a.send_buffers)
        buf .= a[bidx]
    end
end

const mpicomm = MPI.COMM_WORLD

function async_send!(a::BufferedArray, mesh::LocalCartesianMesh, tag)

    fill_sendbufs!(a, mesh)

    for i in 1:length(mesh.neighbor_ranks)
        a.send_reqs[i] = MPI.Isend(a.send_buffers[i], mesh.neighbor_ranks[i], tag, mpicomm)
    end
end

wait_send(a::BufferedArray) = MPI.Waitall!(a.send_reqs)

function async_recv!(a::BufferedArray, mesh::LocalCartesianMesh, tag)
    for i in 1:length(mesh.neighbor_ranks)
        a.recv_reqs[i] = MPI.Irecv!(a.recv_buffers[i], mesh.neighbor_ranks[i], tag, mpicomm)
    end
end

function wait_recv(a::BufferedArray, mesh)
    MPI.Waitall!(a.recv_reqs)
    flush_recvbufs!(a, mesh)
end

elems(mesh::LocalCartesianMesh) = elems(mesh.mesh)
neighbor(elem, face, mesh::LocalCartesianMesh) = neighbor(elem, face, mesh.mesh)
overelems(f, mesh::LocalCartesianMesh, args...) = overelems(f, mesh.mesh, args...)
ghostboundaries(mesh::LocalCartesianMesh) = ghostboundaries(mesh.mesh)

function storage(::Type{T}, mesh::LocalCartesianMesh{N, CPU}) where {T, N}
    bs = ghostboundaries(mesh)
    recv_bufs = map(idxs -> Array{T}(undef, size(idxs)), bs)
    send_bufs = map(idxs -> Array{T}(undef, size(idxs)), bs)

    recv_reqs = fill(MPI.REQUEST_NULL, length(bs))
    send_reqs = fill(MPI.REQUEST_NULL, length(bs))

    BufferedArray(storage(T, mesh.mesh), recv_bufs, send_bufs, recv_reqs, send_reqs)
end

"""
    sync_storage!(mesh::LocalCartesianMesh, st::BufferedArray)

Forward MPI communication `async_send!`, `async_recv!`, `wait_send`, `wait_recv` on `mesh`
to `st` BufferedArray.
"""
sync_storage!(mesh::LocalCartesianMesh, st::BufferedArray) = push!(mesh.synced_storage, st)

function async_send!(m::LocalCartesianMesh)
    for (i, s) in enumerate(m.synced_storage)
        async_send!(s, m, 666+i)
    end
end

function async_recv!(m::LocalCartesianMesh)
    for (i, s) in enumerate(m.synced_storage)
        async_recv!(s, m, 666+i)
    end
end

function wait_recv(m::LocalCartesianMesh)
    for b in m.synced_storage
        wait_recv(b, m)
    end
end
wait_send(m::LocalCartesianMesh) = foreach(wait_send, m.synced_storage)

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
function localpart(globalMesh, mpicomm, ranks=convert(Array, (0:MPI.Comm_size(mpicomm)-1)'))
    P = CartesianPartition(elems(globalMesh), ranks)
    inds = rankindices(P, MPI.Comm_rank(mpicomm))
    mesh = GhostCartesianMesh(CPU(), inds) # TODO: GPU??!
    nbs = map(ghostboundaries(mesh), boundaries(mesh)) do ghostelems, elems
        locate(P, translate(globalMesh, ghostelems))
    end
    LocalCartesianMesh(mesh, nbs, [])
end
