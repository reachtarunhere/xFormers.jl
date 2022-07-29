using xFormers
using Test

@testset "xFormers.jl" begin
    # Write your tests here.
    using xFormers.ShapeUtils: toNdBatch, reshape_op_restore, to1dBatch, @vecop
    

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
