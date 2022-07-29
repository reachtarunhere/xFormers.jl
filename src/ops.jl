module Ops

using ..ShapeUtils: @matrixop
using NNlib

export nested_batched_mul, ⊛

function nested_batched_mul(A, B)
    @matrixop batched_mul(A, B)
end

⊛ = nested_batched_mul


end
