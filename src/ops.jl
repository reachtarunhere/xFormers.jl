module Ops

using ..ShapeUtils: @matrixop
using NNlib

export nested_batched_mul, ⊛, nested_matrix_transpose

function nested_batched_mul(A, B)
    @matrixop batched_mul(A, B)
end

⊛ = nested_batched_mul


end




