using Test
using Llama2Inference

@testset "Transformer Tests" begin
    @testset "Single Function Tests" begin
        @testset "rmsnorm! Tests" begin
            # Test rmsnorm!(out, x, weight)
            @testset "Throw Error Tests" begin
                # Test for DimensionMismatch of either out, x, or weight
                undef_5 = Array{AbstractFloat}(undef, 5)
                @test_throws DimensionMismatch rmsnorm!(undef_5, undef_5[1:4], undef_5)
                @test_throws DimensionMismatch rmsnorm!(undef_5, undef_5, undef_5[1:4])
            end
            @testset "Functionality Tests" begin
                # Test when rmsnorm is computed over full length of x
                x = [0.1, 0.3, 2.0, -0.05, -2.0]
                weight = [0.2, 0.4, 0.1, 0.2, 0.1]
                expected = [0.0157111,  0.0942664,  0.1571106, -0.0078555, -0.1571106] 
                rmsnorm!(x, x, weight)
                @test isapprox(x, expected, atol=1e-5)
                
                # Test when rmsnorm is computed over a subset of x, rest of x should remain unchanged
                x = [0.1, 0.3, 2.0, -0.05, -2.0]
                weight = [0.2, 0.4, 0.5]
                expected = [0.0171080, 0.1026479, 0.8553989] 
                rmsnorm!(@view(x[begin:3]), @view(x[begin:3]), weight)
                @test (isapprox( @view(x[begin:3]), expected, atol=1e-5) && isapprox(@view(x[4:end]), [-0.05, -2.0]))
            end
        end
        @testset "softmax! Tests" begin
            @testset "Functionality Tests" begin
                # Test when softmax is computed over full length of x
                x = [0.1, 0.3, 2.0, -0.05, -2.0]
                expected = [0.10110752, 0.123493, 0.67599418, 0.08702405, 0.01238127]
                softmax!(x)
                @test isapprox(x, expected)
                
                # Test when softmax is computed over a subset of x, rest of x should remain unchanged
                x = [0.1, 0.3, 2.0, -0.05, -2.0]
                x_v = @view(x[begin:3])
                expected = [0.1122675,  0.13712384, 0.75060866]
                softmax!(x_v)
                @test (isapprox(x[begin:3], expected) && isapprox(x[4:end], [-0.05, -2.0]))
                
                # Test when softmax is computed over zero vector
                x = zeros(5)
                expected = [1//5, 1//5, 1//5, 1//5, 1//5]
                softmax!(x)
                @test isapprox(x, expected)
            end
        end
    end
    # Add more testsets for other functions in transformer.jl
end