abstract type Element end

struct RectElem{S<:AbstractArray} <: Element
    coeffs::S
end

function Base.show(io::IO, e::RectElem)
    print(io, "RectElem containing ")
    show(io, MIME"text/plain"(), e.coeffs)
end

Base.@pure function face2idx(arr, f)
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

function face(e::RectElem, f, m::Mesh)
    e.coeffs[face2idx(e.coeffs, f)...]
end

function face(e::RectElem, f)
    e.coeffs[face2idx(e.coeffs, f)...]
end
