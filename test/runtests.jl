using Llama2Inference
using Test
using DataStructures

@testset "Llama2Inference.jl" begin
    # Write your tests here.
    include("tokenizer_tests.jl")
    include("config_tests.jl")
    include("transformer_weights_tests.jl")
end
