# ### Generate a local view of a fully periodic Cartesian mesh.
if dim == 1
    (Nx, ) = brickN
    local x = range(DFloat(0); length=Nx+1, stop=1)
    mesh = brickmesh((x, ), (true, ); part=mpirank+1, numparts=mpisize)
  elseif dim == 2
    (Nx, Ny) = brickN
    local x = range(DFloat(0); length=Nx+1, stop=1)
    local y = range(DFloat(0); length=Ny+1, stop=1)
    mesh = brickmesh((x, y), (true, true); part=mpirank+1, numparts=mpisize)
  else
    (Nx, Ny, Nz) = brickN
    local x = range(DFloat(0); length=Nx+1, stop=1)
    local y = range(DFloat(0); length=Ny+1, stop=1)
    local z = range(DFloat(0); length=Nz+1, stop=1)
    mesh = brickmesh((x, y, z), (true, true, true); part=mpirank+1, numparts=mpisize)
end

# ### Partition the mesh using a Hilbert curve based partitioning
mesh = partition(mpicomm, mesh...)

# ### Connect the mesh in parallel
mesh = connectmesh(mpicomm, mesh...)

# ### Get the degrees of freedom along the faces of each element.
# vmap(:,f,e) gives the list of local (mpirank) points for the face "f" of element "e".  vmapP points to the outward (or neighbor) element and vmapM for the current element. P=+ or right and M=- or left.
(vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface, mesh.elemtoordr)

# ### Create 1-D operators
# $\xi$ and $\omega$ are the 1D Lobatto points and weights and $D$ is the derivative of the basis function.
(ξ, ω) = lglpoints(DFloat, N)
D = spectralderivative(ξ)

# ### Compute metric terms
# nface and nelem refers to the total number of faces and elements for this MPI rank. Also, coord contains the dim-tuple coordinates in the mesh.
(nface, nelem) = size(mesh.elemtoelem)
coord = creategrid(Val(dim), mesh.elemtocoord, ξ)
if dim == 1
  x = coord.x
  for j = 1:length(x)
    x[j] = x[j]
 end
elseif dim == 2
  (x, y) = (coord.x, coord.y)
  for j = 1:length(x)
#=    (x[j], y[j]) = (x[j] .+ sin.(π * x[j]) .* sin.(2 * π * y[j]) / 10,
                    y[j] .+ sin.(2 * π * x[j]) .* sin.(π * y[j]) / 10)
=#
  end
elseif dim == 3
  (x, y, z) = (coord.x, coord.y, coord.z)
  for j = 1:length(x)
    (x[j], y[j], z[j]) = (x[j] + (sin(π * x[j]) * sin(2 * π * y[j]) *
                                  cos(2 * π * z[j])) / 10,
                          y[j] + (sin(π * y[j]) * sin(2 * π * x[j]) *
                                  cos(2 * π * z[j])) / 10,
                          z[j] + (sin(π * z[j]) * sin(2 * π * x[j]) *
                                  cos(2 * π * y[j])) / 10)
  end
end

# ### First VTK Call
# This first VTK call dumps the mesh out for all mpiranks.
writemesh(@sprintf("SWE%dD_rank_%04d_mesh", dim, mpirank), coord...;
          realelems=mesh.realelems)

# ### Compute the metric terms
# This call computes the metric terms of the grid such as $\xi_\mathbf{x}$, $\eta_\mathbf{x}$, $\zeta_\mathbf{x}$ for all spatial dimensions $\mathbf{x}$ depending on the dimension of $dim$.
metric = computemetric(coord..., D)


"""
    state(dim)

Create the state vector, we need as many velocity vectors as there are dimensions
"""
function state(dim, coord)
    if dim == 1
      statesyms = (:h, :U)
    elseif dim == 2
      statesyms = (:h, :U, :V)
    elseif dim == 3
      statesyms = (:h, :U, :V, :W)
    end
    state = NamedTuple{statesyms}(ntuple(j->zero(coord.x), Val(dim+1)))
end

# ### Create storage for state vector and right-hand side
# Q holds the solution vector and rhs the rhs-vector which are dim+1 tuples
# In addition, here we generate the initial conditions
Q = state(dim, coord)
if dim == 1
    bathymetry = zero(coord.x)
    for i=1:length(coord.x)
        bathymetry[i]=0.1
    end
    r=(x .- 0.5).^2
    Q.h .= 0.5 .* exp.(-32.0 .* r)
    Q.U .= 0
    if (advection)
        δnl=1.0
        gravity=0.0
        Q.U .= (Q.h+bathymetry) .* (1.0)
    end
    #=
  for i=1:length(coord.x)
     bathymetry[i]=2.0
  end
  Q.h .= sin.(2 * π * x) .+ 0.0
  Q.U .= (Q.h+bathymetry) .* (1.0)
=#
elseif dim == 2
    bathymetry = zero(coord.x)
    for i=1:length(coord.x)
        bathymetry[i]=0.2
    end
    r=(x .- 0.5).^2 + (y .- 0.5).^2
    Q.h .= 0.5 .* exp.(-100.0 .* r)
    Q.U .= 0
    Q.V .= 0
    if (advection)
        δnl=1.0
        gravity=0.0
        Q.U .= (Q.h+bathymetry) .* (1.0)
        Q.V .= (Q.h+bathymetry) .* (0.0)
    end
#=
    for i=1:length(coord.x)
     bathymetry[i]=2.0
  end
  r=(x .- 0.5).^2 + (y .- 0.5).^2
  Q.h .= sin.(2 * π * x) .* sin.(2 *  π * y)
  #Q.h .= 0.5 .* exp.(-8.0 .* r)
  Q.U .= (Q.h+bathymetry) .* (1.0)
  Q.V .= (Q.h+bathymetry) .* (1.0)
=#
elseif dim == 3
  Q.h .= sin.(2 * π * x) .* sin.(2 *  π * y) .* sin.(2 * π * z) .+ 2.0
  Q.U .= Q.h .* (1.0)
  Q.V .= Q.h .* (1.0)
  Q.W .= Q.h .* (1.0)
end

function timestep()
    # ### Compute the time-step size and number of time-steps
    # Compute a $\Delta t$ such that the Courant number is $1$.
    # This is done for each mpirank and then we do an MPI_Allreduce to find the global minimum.
    # dt = floatmax(DFloat) 
    # if dim == 1
    #     (ξx) = (metric.ξx)
    #     (h,U) = (Q.h+bathymetry,Q.U)
    #     for n = 1:length(U)
    #         loc_dt = (2h[n])  ./ (abs.(U[n] * ξx[n]))
    #         dt = min(dt, loc_dt)
    #     end
    # elseif dim == 2
    #     (ξx, ξy) = (metric.ξx, metric.ξy)
    #     (ηx, ηy) = (metric.ηx, metric.ηy)
    #     (h,U,V) = (Q.h+bathymetry,Q.U,Q.V)
    #     for n = 1:length(U)
    #         loc_dt = (2h[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n]),
    #                         abs.(U[n] * ηx[n] + V[n] * ηy[n]))
    #         dt = min(dt, loc_dt)
    #     end
    # elseif dim == 3
    #     (ξx, ξy, ξz) = (metric.ξx, metric.ξy, metric.ξz)
    #     (ηx, ηy, ηz) = (metric.ηx, metric.ηy, metric.ηz)
    #     (ζx, ζy, ζz) = (metric.ζx, metric.ζy, metric.ζz)
    #     (h,U,V,W) = (Q.h,Q.U,Q.V,Q.W)
    #     for n = 1:length(U)
    #         loc_dt = (2h[n]) ./ max(abs.(U[n] * ξx[n] + V[n] * ξy[n] + W[n] * ξz[n]),
    #                         abs.(U[n] * ηx[n] + V[n] * ηy[n] + W[n] * ηz[n]),
    #                         abs.(U[n] * ζx[n] + V[n] * ζy[n] + W[n] * ζz[n]))
    #         dt = min(dt, loc_dt)
    #     end
    # end
    # dt = MPI.Allreduce(dt, MPI.MIN, mpicomm)
    # dt = DFloat(dt / N^sqrt(2))

    dt = 0.0025
    nsteps = ceil(Int64, tend / dt)
    dt = tend / nsteps
    return dt, nsteps
end

# ### Compute the exact solution at the final time.
# Later Δ will be used to store the difference between the exact and computed solutions.
Δ = state(dim, coord)
if dim == 1
  Δ.h .= Q.h
  Δ.U .= Q.U
elseif dim == 2
  Δ.h .= Q.h
  Δ.U .= Q.U
  Δ.V .= Q.V
elseif dim == 3
  u = Q.U ./ Q.h
  v = Q.V ./ Q.h
  w = Q.W ./ Q.h
  Δ.h .= sin.(2 * π * (x - tend * u)) .* sin.(2 *  π * (y - tend * v)) .*
         sin.(2 * π * (z - tend * w)) .+ 2
  Δ.U .=  Q.U
  Δ.V .=  Q.V
  Δ.W .=  Q.W
end