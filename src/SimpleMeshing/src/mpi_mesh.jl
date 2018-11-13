using MPI
using ..Partitions

export localpart, neighbor_ranks, sync_ghost!,
       fill_sendbufs!, flush_recvbufs!, async_send!, async_recv!,
       wait_send, wait_recv

"""
    LocalCartesianMesh(G, neighbors)
"""
struct LocalCartesianMesh{N, B, G<:CartesianMesh{N, B}} <: CartesianMesh{N, B}
    mesh::G
    neighbor_ranks::Tuple{Vararg{Int}}
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

    recv_bufs = map(idxs -> Array{eltype(x)}(undef, size(idxs)), bs)
    send_bufs = map(idxs -> Array{eltype(x)}(undef, size(idxs)), bs)

    recv_reqs = fill(MPI.REQUEST_NULL, length(bs))
    send_reqs = fill(MPI.REQUEST_NULL, length(bs))

    st = (storage=x, recv_buffers=recv_bufs, send_buffers=send_bufs,
          recv_reqs=recv_reqs, send_reqs=send_reqs)

    push!(mesh.synced_storage, st)
end

function maketag(fs, face)
    findfirst(isequal(face), fs)
end

function async_send!(m::LocalCartesianMesh)
    el = first(elems(m))
    fs = collect(faces(el, m))
    fs′ = opposite.(fs, (el,), (m,)) # Abstraction leak, what if #boundaries != #faces?
    bs = boundaries(m)
    for (i, s) in enumerate(m.synced_storage)
        for (j, b) in enumerate(bs)
            s.send_buffers[j] .= @view s.storage[b]
            to = m.neighbor_ranks[j]
            t = 1000 * i + maketag(fs, fs′[j])
            #println("$(MPI.Comm_rank(mpicomm)) is sending $t to $to")
            s.send_reqs[j] = MPI.Isend(s.send_buffers[j], to, t, mpicomm)
        end
    end
end

function async_recv!(m::LocalCartesianMesh)
    el = first(elems(m))
    fs = collect(faces(el, m))
    fs′ = opposite.(fs, (el,), (m,))
    bs = ghostboundaries(m)
    for (i, s) in enumerate(m.synced_storage)
        for j in 1:length(bs)
            from = m.neighbor_ranks[j]
            t = 1000 * i + maketag(fs, fs[j])
            #println("$(MPI.Comm_rank(mpicomm)) is listening for $t from $from")
            s.recv_reqs[j] = MPI.Irecv!(s.recv_buffers[j], from, t,mpicomm)
        end
    end
end

function wait_recv(m::LocalCartesianMesh)
    MPI.Waitall!(reduce(vcat, map(x->x.recv_reqs, m.synced_storage)))
    bs = ghostboundaries(m)
    for (i, s) in enumerate(m.synced_storage)
        for j in 1:length(bs)
            copyto!(s.storage, bs[j], s.recv_buffers[j], CartesianIndices(s.recv_buffers[j]))
        end
    end
end
function wait_send(m::LocalCartesianMesh)
    MPI.Waitall!(reduce(vcat, map(x->x.send_reqs, m.synced_storage)))
end

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
