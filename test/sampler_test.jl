using Test
using Llama2Inference

@testset "Sampler Tests" begin
    @testset "Sampler constructor" begin
        sampler = Sampler(10, 0.5f0, 0.1f0)
        @test sampler.vocab_size == 10
        @test sampler.temperature == 0.5f0
        @test sampler.topp == 0.1f0
        @test length(sampler.probindex) == 10
    end
    @testset "Throw ArgumentError if length(logits) != sampler.vocab_size" begin
        sampler = Sampler(10, 0.5f0, 0.1f0)
        logits = Vector{Float32}(undef, 0)
        @test_throws ArgumentError sample(sampler, logits)    
    end

    @testset "sample function" begin
        @testset "temperature = 0 => simple argmax case" begin
            sampler = Sampler(6, 0.0f0, 0.0f0)
            logits = [0.1f0, 0.3f0, 0.2f0, 0.15f0, 0.15f0, 0.1f0]
            @test sample(sampler, logits) == 2
        end
        @testset "temperatur != 0 and sampling from all logits" begin
            sampler = Sampler(6, 0.5f0, 0.0f0)
            logits = [0.001f0, 0.995f0, 0.001f0, 0.001f0, 0.001f0, 0.001f0]
            @test sample(sampler, logits) in 1:6
        end
        @testset "top-p sampling with low diversity" begin
            sampler = Sampler(6, 0.5f0, 0.1f0)
            logits = [0.001f0, 0.995f0, 0.001f0, 0.001f0, 0.001f0, 0.001f0]
            @test sample(sampler, logits) == 2
        end
        @testset "top-p sampling with high diversity" begin
            sampler = Sampler(6, 0.5f0, 0.9f0)
            logits = [0.001f0, 0.995f0, 0.001f0, 0.001f0, 0.001f0, 0.001f0]
            @test sample(sampler, logits) in 1:6
        end
    end
end