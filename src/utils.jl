module ShapeUtils

using Base: Fix1, Fix2

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


end
