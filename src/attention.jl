module Attention

using ..ShapeUtils: transpose
using ..Ops
using NNlib: softmax, batched_transpose

export dot_attention

# no scaling by default assuming single headed case
function dot_attention(Q, K, V; dₖ=1)
    A = softmax(transpose(K, 1, 2) ⊛ Q / √dₖ)
    return V ⊛ A , A
end

end
