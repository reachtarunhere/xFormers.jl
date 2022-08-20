using xFormers
using Test

@testset "xFormers.jl" begin
    # Write your tests here.
    using xFormers.ShapeUtils: toNdBatch, reshape_op_restore, to1dBatch, @vecop, transpose

    @test size(transpose(rand(2,3,4,5), 1, 2)) == (3, 2, 4, 5)

    A = rand(2,2,3,4)
    A_t, extradims = toNdBatch(A)
    @test size(A_t) == (2, 2, 12)
    @test extradims == (3, 4)
    @test reshape(A_t, size(A_t)[1:end-1]..., extradims...) == A
    
    my_prod(A) = prod(A, dims=1)
    A_prod = reshape_op_restore(to1dBatch, my_prod, A)
    @test size(A_prod) == (1, 2, 3, 4)
    @test A_prod == prod(A, dims=1)

    using xFormers.Ops
    @test size(rand(3, 2, 4, 5) ⊛ rand(2, 3, 4, 5) ) == (3, 3, 4, 5)

end

@testset "Attention" begin
    K = hcat([10, 0, 0],
             [0, 10, 0],
             [0, 0, 10],
             [0, 0, 10])


    V = hcat([1, 0, 0],
             [10, 0, 21],
             [100, 5, 0],
             [1000, 6, 0])

    Q1 = hcat([0, 10, 0],)

    out1, attention_weights = xFormers.Attention.dot_attention(Q1, K, V)

    # Since Q is same as second K the outcome should be the second V
    @test out1 ≈ V[:, 2]

    Q2 = hcat([0, 0, 10])
    out2, attention_weights = xFormers.Attention.dot_attention(Q2, K, V)

    # Since we have two matches in the last 2 keys the outcome should be average of last 2 values

    @test out2 ≈ (V[:, 3] + V[:,4]) / 2

    # Now we test for multiple queries

    Q3 = hcat(Q1, Q2)

    out3, attention_weights = xFormers.Attention.dot_attention(Q3, K, V)

    @test out3 ≈ hcat(out1, out2)

    # Batched multi-headed test case
    # Inputs are tensors of dₖ=3, seq_len=3 (2 for Q), heads=2, bs=1

    K = reshape(hcat(K, K), 3, 4, 2, 1)
    V = reshape(hcat(V, V), 3, 4, 2, 1)
    Q = reshape(hcat(Q3, Q3), 3, 2, 2, 1) # 2 queries per head per batch

    out4, attention_weights = xFormers.Attention.dot_attention(Q, K, V)

    # dims => d * seq_len * heads * batch_size
    @test out4[:, 1, 1, 1] ≈ out4[:, 1, 2, 1] ≈ out1
    @test out4[:, 2, 1, 1] ≈ out4[:, 2, 2, 1] ≈ out2

end
