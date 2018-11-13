using MPI
using Canary
import Printf: @sprintf

##
# Configuration
##
const DFloat    = Float64       # Number Type
const N         = 3             # polynominal order
const brickN    = (10, 10)      # 2D brickmesh
const tend      = DFloat(0.005) # Final Time
const δnl       = 1.0           # switch to turn on/off nonlinear equations
const gravity   = 10.0          # gravity
const advection = false         # Boolean to turn on/off advection or swe

# ### The grid that we create determines the number of spatial dimensions that we are going to use.
const dim = length(brickN)

MPI.Initialized() || MPI.Init() # only initialize MPI if not initialized
MPI.finalize_atexit()
const mpicomm = MPI.COMM_WORLD
const mpirank = MPI.Comm_rank(mpicomm)
const mpisize = MPI.Comm_size(mpicomm)

if mpirank == 0
    @show mpisize
    @show DFloat
    @show N
    @show brickN
    @show dim
    @show tend
    @show δnl
    @show gravity
    @show advection
end

include(joinpath(@__DIR__, "..", "original", "vtk.jl"))
include("flux.jl")

##
# Main computation
##
function main(mesh, Q, Δ, bathymetry, coord, metric, D, ω, vmapM, vmapP)
    dt, nsteps = timestep()

    # ### Compute how many MPI neighbors we have
    # "mesh.nabrtorank" stands for "Neighbors to rank"
    numnabr = length(mesh.nabrtorank)

    # ### Create send/recv request arrays
    # "sendreq" is the array that we use to send the communication request. It needs to be of the same length as the number of neighboring ranks. Similarly, "recvreq" is the array that we use to receive the neighboring rank information.
    sendreq = fill(MPI.REQUEST_NULL, numnabr)
    recvreq = fill(MPI.REQUEST_NULL, numnabr)

    # ### Create send/recv buffer
    # The dimensions of these arrays are 
    # 1. degrees of freedom within an element
    # 2. number of solution vectors
    # 3. the number of "send elements" and "ghost elements"
    # respectively.
    sendQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.sendelems))
    recvQ = Array{DFloat, 3}(undef, (N+1)^dim, length(Q), length(mesh.ghostelems))

    # Build CartesianIndex map for moving between Cartesian and linear storage of
    # dofs
    index = CartesianIndices(ntuple(j->1:N+1, dim))
    nrealelem = length(mesh.realelems)

    # ### Dump the initial condition
    # Dump out the initial conditin to VTK prior to entering the time-step loop.
    temp=Q.h + bathymetry
    writemesh(@sprintf("SWE%dD_rank_%04d_step_%05d", dim, mpirank, 0),
            coord...; fields=(("hs+hb", temp),), realelems=mesh.realelems)

    
    rhs = state(dim, coord)

    # ### Begin Time-step loop
    # Go through nsteps time-steps and for each time-step, loop through the s-stages of the explicit RK method.
    for step = 1:nsteps
        mpirank == 0 && @show step
        for s = 1:length(RKA) # n-th order runge-kutta
            # #### Post MPI receives
            # We assume that an MPI_Isend has been posted (non-blocking send) and are waiting to receive any message that has
            # been posted for receiving.  We are looping through the : (1) number of neighbors, (2) neighbor ranks,
            # and (3) neighbor elements.
            for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                                mesh.nabrtorecv)
                recvreq[nnabr] = MPI.Irecv!((@view recvQ[:, :, nabrelem]), nabrrank, 777,
                                            mpicomm)
            end

            # #### Wait on (prior) MPI sends
            # WE assume that non-blocking sends have been sent and wait for this to happen. FXG: Why do we need to wait?
            MPI.Waitall!(sendreq)

            # #### Pack data to send buffer
            # For all faces "nf" and all elements "ne" we pack the send data.
            for (ne, e) ∈ enumerate(mesh.sendelems)
                for (nf, f) ∈ enumerate(Q)
                    sendQ[:, nf, ne] = f[index[:], e]
                end
            end

            # #### Post MPI sends
            # For all: (1) number of neighbors, (2) neighbor ranks, and (3) neighbor elements we perform a non-blocking send.
            for (nnabr, nabrrank, nabrelem) ∈ zip(1:numnabr, mesh.nabrtorank,
                                                mesh.nabrtosend)
                sendreq[nnabr] = MPI.Isend((@view sendQ[:, :, nabrelem]), nabrrank, 777,
                                        mpicomm)
            end

            # #### Compute RHS Volume Integral
            # Note that it is not necessary to have received all the MPI messages. Here we are interleaving computation
            # with communication in order to curtail latency.  Here we perform the RHS volume integrals.
            # call volumerhs
            volumerhs!(rhs, Q, bathymetry, metric, D, ω, mesh.realelems, gravity, δnl)

            # #### Wait on MPI receives
            # We need to wait to receive the messages before we move on to t=e flux integrals.
            MPI.Waitall!(recvreq)

            # #### Unpack data from receive buffer
            # The inverse of the Pack data to send buffer. We now unpack the receive buffer in order to use it in the RHS
            # flux integral.
            for elems ∈ mesh.nabrtorecv
                for (nf, f) ∈ enumerate(Q)
                    f[index[:], nrealelem .+ elems] = recvQ[:, nf, elems]
                end
            end

            # #### Compute RHS Flux Integral
            # We compute the flux integral on all "realelems" which are the elements owned by the current mpirank.
            # call fluxrhs
            fluxrhs!(rhs, Q, bathymetry, metric, ω, mesh.realelems, vmapM, vmapP, gravity, N, δnl)

            # #### Update solution and scale RHS
            # We need to update/evolve the solution in time and multiply by the inverse mass matrix.
            #call updatesolution
            updatesolution!(rhs, Q, bathymetry, metric, ω, mesh.realelems, RKA[s%length(RKA)+1], RKB[s], dt, advection)
        end #s-stages

        # #### Write VTK Output
        # After each time-step, we dump out VTK data for Paraview/VisIt.
        temp=Q.h + bathymetry
        writemesh(@sprintf("SWE%dD_rank_%04d_step_%05d", dim, mpirank, step),
                coord...; fields=(("hs+hb", temp),), realelems=mesh.realelems)
    end #step

    # ### Compute L2 Error Norms
    # Since we stored the initial condition, we can now compute the L2 error norms for both the solution and energy.

    #extract velocity fields
    if dim == 1
        Q.U .= Q.U ./ (Q.h+bathymetry)
        Δ.U .= Δ.U ./ (Δ.h+bathymetry)
        Q.h .= Q.h
        Δ.h .= Δ.h
    elseif dim == 2
        Q.U .= Q.U ./ (Q.h+bathymetry)
        Δ.U .= Δ.U ./ (Δ.h+bathymetry)
        Q.V .= Q.V ./ (Q.h+bathymetry)
        Δ.V .= Δ.V ./ (Δ.h+bathymetry)
        Q.h .= Q.h
        Δ.h .= Δ.h
    elseif dim == 3
        Q.U .= Q.U ./ Q.h
        Δ.U .= Δ.U ./ Δ.h
        Q.V .= Q.V ./ Q.h
        Δ.V .= Δ.V ./ Δ.h
        Q.W .= Q.W ./ Q.h
        Δ.W .= Δ.W ./ Δ.h
        Q.h .= Q.h
        Δ.h .= Δ.h
    end

    #Compute Norms
    for (δ, q) ∈ zip(Δ, Q)
        δ .-= q
    end
    eng = L2energy(Q, metric, ω, mesh.realelems)
    eng = MPI.Allreduce(eng, MPI.SUM, mpicomm)
    mpirank == 0 && @show sqrt(eng)

    err = L2energy(Δ, metric, ω, mesh.realelems)
    err = MPI.Allreduce(err, MPI.SUM, mpicomm)
    mpirank == 0 && @show sqrt(err)
    return (mesh, Q.h)

end

include("setup.jl")
(mesh, h) = main(mesh, Q, Δ, bathymetry, coord, metric, D, ω, vmapM, vmapP)
