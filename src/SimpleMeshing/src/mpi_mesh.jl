using MPI
using ..Partitions

export localpart, neighbor_ranks, sync_ghost!,
       fill_sendbufs!, flush_recvbufs!, async_send!, async_recv!,
       wait_send, wait_recv

export LocalCartesianMesh

"""
    LocalCartesianMesh(G, neighbors)
"""
struct LocalCartesianMesh{N, B, G<:CartesianMesh{N, B}} <: CartesianMesh{N, B}
    mesh::G
    neighbor_ranks::Vector{Int}
    synced_storage::Vector{Any}
end

const mpicomm = MPI.COMM_WORLD

elems(mesh::LocalCartesianMesh) = elems(mesh.mesh)
neighbor(elem, face, mesh::LocalCartesianMesh) = neighbor(elem, face, mesh.mesh)
overelems(f, mesh::LocalCartesianMesh, args...) = overelems(f, mesh.mesh, args...)
boundaries(mesh::LocalCartesianMesh) = boundaries(mesh.mesh)
ghostboundaries(mesh::LocalCartesianMesh) = ghostboundaries(mesh.mesh)

storage(T::Type, m::LocalCartesianMesh{<:Any, CPU}) = storage(T, m.mesh)

"""
    sync_ghost!(mesh::LocalCartesianMesh, st::OffsetArray)

Track the ghost cells of this array, communicate them in `async_send!` and `async_recv!`
"""
function sync_ghost!(mesh::LocalCartesianMesh, x::OffsetArray)
    bs = boundaries(mesh)

    if haskey(ENV, "MPI_CUDA_SUPPORT")
        recv_bufs = map(idxs -> similar(parent(x), eltype(x), size(idxs)), bs)
        send_bufs = map(idxs -> similar(parent(x), eltype(x), size(idxs)), bs)
    else
        recv_bufs = map(idxs -> Array{eltype(x)}(undef, size(idxs)), bs)
        send_bufs = map(idxs -> Array{eltype(x)}(undef, size(idxs)), bs)
    end

    recv_reqs = fill(MPI.REQUEST_NULL, length(bs))
    send_reqs = fill(MPI.REQUEST_NULL, length(bs))

    st = (storage=x, recv_buffers=recv_bufs, send_buffers=send_bufs,
          recv_reqs=recv_reqs, send_reqs=send_reqs)

    push!(mesh.synced_storage, st)
    nothing
end
function sync_ghost!(mesh, x) end

function maketag(fs, face)
    t = findfirst(isequal(face), fs)
    if t === nothing
        error("Failed to figure out where to send this")
    end
    return t
end

@noinline function start_send_bufs(s, tag1, bs, fs, fs′, to)
    for j in 1:length(bs)
        b = bs[j]
        buf = s.send_buffers[j]
        copyto!(buf, CartesianIndices(buf), s.storage, b)
        t = tag1 + maketag(fs, fs′[j])
        #println("$(MPI.Comm_rank(mpicomm)) is sending $t to $to")
        s.send_reqs[j] = MPI.Isend(buf, to[j], t, mpicomm)
    end
end

function async_send!(m::LocalCartesianMesh)
    el = first(elems(m))
    fs = collect(faces(el, m))
    fs′ = opposite.(fs, (el,), (m,))
    bs = boundaries(m)
    for (i, s) in enumerate(m.synced_storage)
        start_send_bufs(s, 1000*i, bs, fs, fs′, m.neighbor_ranks)
    end
end
function async_send!(m) end

function start_recv_buf(s, tag1, fs, from, bs)
    for j in 1:length(bs)
        t = tag1 + maketag(fs, fs[j])
        #println("$(MPI.Comm_rank(mpicomm)) is listening for $t from $from")
        s.recv_reqs[j] = MPI.Irecv!(s.recv_buffers[j], from[j],t,mpicomm)
    end
end

function async_recv!(m::LocalCartesianMesh)
    el = first(elems(m))
    fs = collect(faces(el, m))
    fs′ = opposite.(fs, (el,), (m,))
    bs = ghostboundaries(m)
    from = m.neighbor_ranks
    for (i, s) in enumerate(m.synced_storage)
        start_recv_buf(s, 1000*i, fs, from, bs)
    end
end
function async_recv!(m) end


@noinline function flush_recvbufs!(x, boundaries, bufs)
    for j in 1:length(boundaries)
        buf = bufs[j]
        copyto!(x, boundaries[j], buf, CartesianIndices(buf))
    end
end
function wait_recv(m::LocalCartesianMesh)
    MPI.Waitall!(reduce(vcat, map(x->x.recv_reqs, m.synced_storage)))
    bs = ghostboundaries(m)
    for (i, s) in enumerate(m.synced_storage)
        flush_recvbufs!(s.storage, bs, s.recv_buffers)
    end
end
function wait_recv(m) end

function wait_send(m::LocalCartesianMesh)
    foreach(x->MPI.Waitall!(x.send_reqs), m.synced_storage)
end
function wait_send(m) end

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

Returns `globalMesh` if `length(ranks) == 1`.
"""
function localpart(globalMesh, mpicomm, ranks=convert(Array, (0:MPI.Comm_size(mpicomm)-1)'))
    if length(ranks) == 1 && first(ranks) == 0
        return globalMesh
    end
    P = CartesianPartition(elems(globalMesh), ranks)
    inds = rankindices(P, MPI.Comm_rank(mpicomm))
    mesh = GhostCartesianMesh(CPU(), inds) # TODO: GPU??!
    nbs = map(ghostboundaries(mesh), boundaries(mesh)) do ghostelems, elems
        locate(P, translate(globalMesh, ghostelems))
    end
    LocalCartesianMesh(mesh, [nbs...], [])
end
