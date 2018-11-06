# ### Store Explicit RK Time-stepping Coefficients
# We use the fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy (1994)
# ((5,4) 2N-Storage RK scheme.
#
# Ref:
# @TECHREPORT{CarpenterKennedy1994,
#   author = {M.~H. Carpenter and C.~A. Kennedy},
#   title = {Fourth-order {2N-storage} {Runge-Kutta} schemes},
#   institution = {National Aeronautics and Space Administration},
#   year = {1994},
#   number = {NASA TM-109112},
#   address = {Langley Research Center, Hampton, VA},
# }
const RKA = (DFloat(0),
       DFloat(-567301805773)  / DFloat(1357537059087),
       DFloat(-2404267990393) / DFloat(2016746695238),
       DFloat(-3550918686646) / DFloat(2091501179385),
       DFloat(-1275806237668) / DFloat(842570457699 ))

const RKB = (DFloat(1432997174477) / DFloat(9575080441755 ),
       DFloat(5161836677717) / DFloat(13612068292357),
       DFloat(1720146321549) / DFloat(2090206949498 ),
       DFloat(3134564353537) / DFloat(4481467310338 ),
       DFloat(2277821191437) / DFloat(14882151754819))

const RKC = (DFloat(0),
       DFloat(1432997174477) / DFloat(9575080441755),
       DFloat(2526269341429) / DFloat(6820363962896),
       DFloat(2006345519317) / DFloat(3224310063776),
       DFloat(2802321613138) / DFloat(2924317926251))

#-------------------------------------------------------------------------------#
#-----Begin Volume, Flux, Update, and Error Functions for Multiple Dispatch-----#
#-------------------------------------------------------------------------------#
# ### Volume RHS Routines
# These functions solve the volume term $\int_{\Omega_e} \nabla \psi \cdot \left( \rho \mathbf{u} \right)^{(e)}_N$ for:
# Volume RHS for 1D
function volumerhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
  (rhsh, rhsU) = (rhs.h, rhs.U)
  (h, U) = (Q.h, Q.U)
  Nq = size(h, 1)
  J = metric.J
  ξx = metric.ξx
  for e ∈ elems
      #Get primitive variables and fluxes
      hb    = bathymetry[:,e]
      hs    = h[:,e]
      ht    = hs + hb
      u     = U[:,e] ./ ht
      fluxh = U[:,e]
      fluxU = (ht .* u .* u + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb

      # loop of ξ-grid lines
      rhsh[:,e] += D' * (ω .* J[:,e] .* (ξx[:,e] .* fluxh[:]))
      rhsU[:,e] += D' * (ω .* J[:,e] .* (ξx[:,e] .* fluxU[:])) #assuming dhb/dx=0: need to include it
  end #e ∈ elems
end #function volumerhs-1d

# Volume RHS for 2D
function volumerhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
    (rhsh, rhsU, rhsV) = (rhs.h, rhs.U, rhs.V)
    (h, U, V) = (Q.h, Q.U, Q.V)
    Nq = size(h, 1)
    J = metric.J
    dim=2
    (ξx, ξy) = (metric.ξx, metric.ξy)
    (ηx, ηy) = (metric.ηx, metric.ηy)
    fluxh=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxU=Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxV=Array{DFloat,3}(undef,dim,Nq,Nq)
    for e ∈ elems
        #Get primitive variables and fluxes
        hb=bathymetry[:,:,e]
        hs=h[:,:,e]
        ht=hs + hb
        u=U[:,:,e] ./ ht
        v=V[:,:,e] ./ ht
        fluxh[1,:,:]=U[:,:,e]
        fluxh[2,:,:]=V[:,:,e]
        fluxU[1,:,:]=(ht .* u .* u + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb
        fluxU[2,:,:]=(ht .* u .* v) .* δnl
        fluxV[1,:,:]=(ht .* v .* u) .* δnl
        fluxV[2,:,:]=(ht .* v .* v + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb

        # loop of ξ-grid lines
        for j = 1:Nq
            rhsh[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxh[1,:,j] + ξy[:,j,e] .* fluxh[2,:,j]))
            rhsU[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxU[1,:,j] + ξy[:,j,e] .* fluxU[2,:,j]))
            rhsV[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxV[1,:,j] + ξy[:,j,e] .* fluxV[2,:,j]))
        end #j
        # loop of η-grid lines
        for i = 1:Nq
            rhsh[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxh[1,i,:] + ηy[i,:,e] .* fluxh[2,i,:]))
            rhsU[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxU[1,i,:] + ηy[i,:,e] .* fluxU[2,i,:]))
            rhsV[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxV[1,i,:] + ηy[i,:,e] .* fluxV[2,i,:]))
        end #i
    end #e ∈ elems
end #function volumerhs-2d

# ### Flux RHS Routines
# These functions solve the flux integral term $\int_{\Gamma_e} \psi \mathbf{n} \cdot \left( \rho \mathbf{u} \right)^{(*,e)}_N$ for:
# Flux RHS for 1D
function fluxrhs!(rhs, Q::NamedTuple{S, NTuple{2, T}}, bathymetry, metric, ω, elems, vmapM, vmapP, gravity, N, δnl) where {S, T}

    (rhsh, rhsU) = (rhs.h, rhs.U)
    (h, U) = (Q.h, Q.U)
    nface = 2
    (nx, sJ) = (metric.nx, metric.sJ)
    nx = reshape(nx, size(vmapM))
    sJ = reshape(sJ, size(vmapM))

    for e ∈ elems
        for f ∈ 1:nface
            #Compute fluxes on M/Left/- side
            hsM = h[vmapM[1, f, e]]
            hbM=bathymetry[vmapM[1, f, e]]
            hM=hsM + hbM
            UM = U[vmapM[1, f, e]]
            uM = UM ./ hM
            fluxhM = UM
            fluxUM = ( hM .* uM .* uM + 0.5 .* gravity .* hsM .^2) .* δnl + gravity .* hsM .* hbM

            #Compute fluxes on P/Right/+ side
            hsP = h[vmapP[1, f, e]]
            hbP=bathymetry[vmapP[1, f, e]]
            hP=hsP + hbP
            UP = U[vmapP[1, f, e]]
            uP = UP ./ hP
            fluxhP = UP
            fluxUP = (hP .* uP .* uP + 0.5 .* gravity .* hsP .^2) .* δnl + gravity .* hsP .* hbP

            #Compute wave speed
            nxM = nx[1, f, e]
            λM=( abs.(nxM .* uM) + sqrt(gravity*hM) ) .* δnl + ( sqrt(gravity*hbM) ) .* (1.0-δnl)
            λP=( abs.(nxM .* uP) + sqrt(gravity*hP) ) .* δnl + ( sqrt(gravity*hbP) ) .* (1.0-δnl)
            λ = max.( λM, λP )

            #Compute Numerical Flux and Update
            fluxh_star = (nxM .* (fluxhM + fluxhP) - λ .* (hsP - hsM)) / 2
            fluxU_star = (nxM .* (fluxUM + fluxUP) - λ .* (UP - UM)) / 2
            rhsh[vmapM[1, f, e]] -= sJ[1, f, e] .* fluxh_star
            rhsU[vmapM[1, f, e]] -= sJ[1, f, e] .* fluxU_star
        end #for f ∈ 1:nface
    end #e ∈ elems
end #function fluxrhs-1d

# Flux RHS for 2D
function fluxrhs!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, ω, elems, vmapM, vmapP, gravity, N, δnl) where {S, T}
    (rhsh, rhsU, rhsV) = (rhs.h, rhs.U, rhs.V)
    (h, U, V) = (Q.h, Q.U, Q.V)
    nface = 4
    Nq=N+1
    dim=2
    (nx, ny, sJ) = (metric.nx, metric.ny, metric.sJ)
    fluxhM=Array{DFloat,2}(undef,dim,Nq)
    fluxUM=Array{DFloat,2}(undef,dim,Nq)
    fluxVM=Array{DFloat,2}(undef,dim,Nq)
    fluxhP=Array{DFloat,2}(undef,dim,Nq)
    fluxUP=Array{DFloat,2}(undef,dim,Nq)
    fluxVP=Array{DFloat,2}(undef,dim,Nq)
    for e ∈ elems
        for f ∈ 1:nface
            #Compute fluxes on M/Left/- side
            hsM = h[vmapM[:, f, e]]
            hbM=bathymetry[vmapM[:, f, e]]
            hM=hsM + hbM
            UM = U[vmapM[:, f, e]]
            uM = UM ./ hM
            VM = V[vmapM[:, f, e]]
            vM = VM ./ hM
            fluxhM[1,:] = UM
            fluxhM[2,:] = VM
            fluxUM[1,:] = ( hM .* uM .* uM + 0.5 .* gravity .* hsM .^2) .* δnl + gravity .* hsM .* hbM
            fluxUM[2,:] = ( hM .* uM .* vM ) .* δnl
            fluxVM[1,:] = ( hM .* vM .* uM ) .* δnl
            fluxVM[2,:] = ( hM .* vM .* vM + 0.5 .* gravity .* hsM .^2) .* δnl + gravity .* hsM .* hbM

            #Compute fluxes on P/right/+ side
            hsP = h[vmapP[:, f, e]]
            hbP=bathymetry[vmapP[:, f, e]]
            hP=hsP + hbP
            UP = U[vmapP[:, f, e]]
            uP = UP ./ hP
            VP = V[vmapP[:, f, e]]
            vP = VP ./ hP
            fluxhP[1,:] = UP
            fluxhP[2,:] = VP
            fluxUP[1,:] = ( hP .* uP .* uP + 0.5 .* gravity .* hsP .^2) .* δnl + gravity .* hsP .* hbP
            fluxUP[2,:] = ( hP .* uP .* vP ) .* δnl
            fluxVP[1,:] = ( hP .* vP .* uP ) .* δnl
            fluxVP[2,:] = ( hP .* vP .* vP + 0.5 .* gravity .* hsP .^2) .* δnl + gravity .* hsP .* hbP

            #Compute wave speed
            nxM = nx[:, f, e]
            nyM = ny[:, f, e]
            λM=( abs.(nxM .* uM + nyM .* vM) + sqrt.(gravity*hM) ) .* δnl + ( sqrt.(gravity*hbM) ) .* (1.0-δnl)
            λP=( abs.(nxM .* uP + nyM .* vP) + sqrt.(gravity*hP) ) .* δnl + ( sqrt.(gravity*hbP) ) .* (1.0-δnl)
            λ = max.( λM, λP )

            #Compute Numerical Flux and Update
            fluxh_star = (nxM .* (fluxhM[1,:] + fluxhP[1,:]) + nyM .* (fluxhM[2,:] + fluxhP[2,:]) - λ .* (hsP - hsM)) / 2
            fluxU_star = (nxM .* (fluxUM[1,:] + fluxUP[1,:]) + nyM .* (fluxUM[2,:] + fluxUP[2,:]) - λ .* (UP - UM)) / 2
            fluxV_star = (nxM .* (fluxVM[1,:] + fluxVP[1,:]) + nyM .* (fluxVM[2,:] + fluxVP[2,:]) - λ .* (VP - VM)) / 2
            rhsh[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxh_star
            rhsU[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxU_star
            rhsV[vmapM[:, f, e]] -= ω .* sJ[:, f, e] .* fluxV_star
        end #f ∈ 1:nface
    end #e ∈ elems
end #function fluxrhs-2d

# ### Update the solution via RK Method for:
# Update 1D
function updatesolution!(rhs, Q::NamedTuple{S, NTuple{2, T}}, bathymetry, metric, ω, elems, rka, rkb, dt, advection) where {S, T}
    #Save original velocity
    if advection
        h = Q.h + bathymetry
        u = Q.U ./ h
    end

    J = metric.J
    M =  ω
    for (rhsq, q) ∈ zip(rhs, Q)
        for e ∈ elems
            q[:, e] += rkb * dt * rhsq[:, e] ./ ( M .* J[:, e])
            rhsq[:, e] *= rka
        end
    end
    #Reset velocity
    if advection
        Q.U .= (Q.h+bathymetry) .* u
    end
end #function update-1d

# Update 2D
function updatesolution!(rhs, Q::NamedTuple{S, NTuple{3, T}}, bathymetry, metric, ω, elems, rka, rkb, dt, advection) where {S, T}
    #Save original velocity
    if (advection)
        h = Q.h + bathymetry
        u = Q.U ./ h
        v = Q.V ./ h
    end

    J = metric.J
    M = reshape(kron(ω, ω), length(ω), length(ω))
    for (rhsq, q) ∈ zip(rhs, Q)
        for e ∈ elems
            q[:, :, e] += rkb * dt * rhsq[:, :, e] ./ (M .* J[:, :, e])
            rhsq[:, :, e] *= rka
        end
    end
    #Reset velocity
    if (advection)
        Q.U .= (Q.h+bathymetry) .* u
        Q.V .= (Q.h+bathymetry) .* v
    end
end #function update-2d

# ### Compute L2 Error Norm for:
# 1D Error
function L2energy(Q::NamedTuple{S, NTuple{2, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = ω
  index = CartesianIndices(ntuple(j->1:Nq, Val(1)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end #end function L2energy-1d

# 2D Error
function L2energy(Q::NamedTuple{S, NTuple{3, T}}, metric, ω, elems) where {S, T}
  J = metric.J
  Nq = length(ω)
  M = reshape(kron(ω, ω), Nq, Nq)
  index = CartesianIndices(ntuple(j->1:Nq, Val(2)))

  energy = [zero(J[1])]
  for q ∈ Q
    for e ∈ elems
      for ind ∈ index
        energy[1] += M[ind] * J[ind, e] * q[ind, e]^2
      end
    end
  end
  energy[1]
end #end function L2energy-2d

#-------------------------------------------------------------------------------#
#--------End Volume, Flux, Update, Error Functions for Multiple Dispatch--------#
#-------------------------------------------------------------------------------#
