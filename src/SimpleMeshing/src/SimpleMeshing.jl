module SimpleMeshing

export Meshing, Partitions

include("mesh.jl")
include("partitions.jl")

###
# Hacks
###

# storage of a CartesianMesh is an OffsetArray and the elems are CartesianIndex
# This function needs to be upstreamed
using OffsetArrays

Base.@propagate_inbounds function Base.getindex(A::OffsetArray{T,N}, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(A, I)
    @inbounds ret = parent(A)[OffsetArrays.offset(A.offsets, I.I)...]
    ret
end

end
