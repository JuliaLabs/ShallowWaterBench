abstract type Funk{T, N} end

abstract type SquareFunk{T, N} <: Funk{T, N} end

abstract type Basis{T, N} end

abstract type SquareLobottoBasis{T, N, P} <: Basis{T, N} end
#defines reference (lobotto) points
#defines interpolation function
#Also has a few Funks in it?

struct InterpolatedFunk{T, N, A<:AbstractArray{<:Any, N}, Basis{<:Any, N}} <: SquareFunk{T,N}
    coeffs::A
end

function Base.-(a::InterpolatedFunk{<:Any, <:Any, <:Any, B})
  InterpolatedFunk(-.a.coeffs)
end

function Base.+(a::InterpolatedFunk{<:Any, <:Any, <:Any, B}, b::InterpolatedFunk{<:Any, <:Any, <:Any, B}) where {B}
  InterpolatedFunk(a.coeffs .+ b.coeffs)
end

function Base.-(a::InterpolatedFunk{<:Any, <:Any, <:Any, B}, b::InterpolatedFunk{<:Any, <:Any, <:Any, B}) where {B}
  InterpolatedFunk(a.coeffs .- b.coeffs)
end

function Base.*(a::InterpolatedFunk{<:Any, <:Any, <:Any, B}, b::InterpolatedFunk{<:Any, <:Any, <:Any, B}) where {B}
  InterpolatedFunk(a.coeffs .* b.coeffs)
end

function Base./(a::InterpolatedFunk{<:Any, <:Any, <:Any, B}, b::InterpolatedFunk{<:Any, <:Any, <:Any, B}) where {B}
  InterpolatedFunk(a.coeffs ./ b.coeffs)
end

# Base.getindex(InterpolatedFunk, inds) = ?? how to interpolate?


function Base.show(io::IO, e::RectFunk)
    print(io, "RectFunk containing ")
    show(io, MIME"text/plain"(), e.coeffs)
end

function face2idx(arr, f)
    @assert ndims(arr) == length(f)
    ntuple(ndims(arr)) do i
        dir = f[i]
        if dir == 0
            Colon()
        elseif dir == -1
            1
        elseif dir == 1
            size(arr, i)
        else
            error("Only 1, 0 and -1 are allowed in face index")
        end
    end
end

function face(e::RectFunk, f, m::Mesh)
    e.coeffs[face2idx(e.coeffs, f)...]
end

function face(e::RectFunk, f)
    e.coeffs[face2idx(e.coeffs, f)...]
end

# Volume RHS for 2D
function volumerhs!(h′, V⃗′, h, V⃗, bathymetry, metric, D, ω, elems, gravity, δnl) where {S, T}
    Nq = size(h, 1)
    J = metric.J
    dim=2
    ξ⃗ = metric.ξ⃗
    η⃗ = metric.η⃗
    fluxh = Array{DFloat,3}(undef,dim,Nq,Nq)
    fluxV⃗ = Array{DFloat,3}(undef,dim,Nq,Nq)
    for e ∈ elems
        hb = bathymetry[e]
        hs = h[e]
        ht = hb + hs
        v⃗ = V⃗[e] / ht
        fluxh = V⃗[e]

        fluxV⃗ = ht * v⃗ * 

        ξ⃗ = dΩ

        u=U[:,:,e] ./ H
        v=V[:,:,e] ./ H
        fluxh[1,:,:]=U[:,:,e]
        fluxh[2,:,:]=V[:,:,e]
        fluxU[1,:,:]=(H .* u .* u + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb
        fluxU[2,:,:]=(H .* u .* v) .* δnl
        fluxV[1,:,:]=(H .* v .* u) .* δnl
        fluxV[2,:,:]=(H .* v .* v + 0.5 .* gravity .* hs .^2) .* δnl + gravity .* hs .* hb

        # loop of ξ-grid lines
        for j = 1:Nq
            h′[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxh[1,:,j] + ξy[:,j,e] .* fluxh[2,:,j]))
            rhsU[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxU[1,:,j] + ξy[:,j,e] .* fluxU[2,:,j]))
            V′[:,j,e] += D' * (ω[j] * ω .* J[:,j,e].* (ξx[:,j,e] .* fluxV[1,:,j] + ξy[:,j,e] .* fluxV[2,:,j]))
        end #j
        # loop of η-grid lines
        for i = 1:Nq
            h′[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxh[1,i,:] + ηy[i,:,e] .* fluxh[2,i,:]))
            rhsU[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxU[1,i,:] + ηy[i,:,e] .* fluxU[2,i,:]))
            V′[i,:,e] += D' * (ω[i] * ω .* J[i,:,e].* (ηx[i,:,e] .* fluxV[1,i,:] + ηy[i,:,e] .* fluxV[2,i,:]))
        end #i
    end #e ∈ elems
end #function volumerhs-2d
