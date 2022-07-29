module ShapeUtils

using Base: Fix1, Fix2
using NNlib

export @matrixop, @vecop

function toNdBatch(A, n = 2)
    extradims = size(A)[n+1:end]
    reshape(A, size(A)[1:n]..., :), extradims
end

to2dBatch = Fix2(toNdBatch, 2)
to1dBatch = Fix2(toNdBatch, 1)

restore_last_dims(A, extradims) = reshape(A, size(A)[1:end-1]..., extradims...)

function reshape_op_restore(reshape_fn, op, args...)
    reshaped = map(reshape_fn, args)
    reshaped_args = [x[1] for x in reshaped]
    extradims = [x[2] for x in reshaped]
    processed = op(reshaped_args...)
    processed = isa(processed, Tuple) ? processed : (processed,)
    restored = map(restore_last_dims, processed, extradims)
    length(restored) == 1 ? restored[1] : tuple(restored...)
end

macro vecop(exp)
    exp.args = [esc(a) for a in exp.args]
    exp.args = vcat([reshape_op_restore, to1dBatch], exp.args)
    exp
end


macro matrixop(exp)
    exp.args = [esc(a) for a in exp.args]
    exp.args = vcat([reshape_op_restore, to2dBatch], exp.args)
    exp
end

function split_heads(A; heads=1)
    length, bs = size(A)[end-1:end]
    permutedims(reshape(A, :, heads, length, bs), (1, 3, 2, 4))
end

function join_heads(A)
    A = permutedims(A, (1, 3, 2, 4))
    reshape(A, prod(size(A)[1:2]), size(A)[3:4]...)
end

function transpose(A, dim1::Int, dim2::Int)
    dims = collect(1:length(size(A)))
    dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
    permutedims(A, dims)
end


end
