using Adapt
using OffsetArrays

Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(adapt(to, parent(x)), x.offsets)