using Test
using Llama2Inference
using DelimitedFiles

@testset "transformer.jl Tests" begin
    @testset "Math Function Tests" begin
        @testset "rmsnorm! Tests" begin
            # Test rmsnorm!(out, x, weight)
            @testset "Throw Error Tests" begin
                # Test for DimensionMismatch of either out, x, or weight
                undef_5 = Array{Float32}(undef, 5)
                @test_throws DimensionMismatch rmsnorm!(undef_5, undef_5[1:4], undef_5)
                @test_throws DimensionMismatch rmsnorm!(undef_5, undef_5, undef_5[1:4])
            end
            @testset "Functionality Tests" begin
                # Test when rmsnorm is computed over full length of x
                x = Float32[0.1, 0.3, 2.0, -0.05, -2.0]
                weight = Float32[0.2, 0.4, 0.1, 0.2, 0.1]
                expected = Float32[0.0157111,  0.0942664,  0.1571106, -0.0078555, -0.1571106] 
                rmsnorm!(x, x, weight)
                @test isapprox(x, expected, atol=1e-5)
                
                # Test when rmsnorm is computed over a subset of x, rest of x should remain unchanged
                x = Float32[0.1, 0.3, 2.0, -0.05, -2.0]
                weight = Float32[0.2, 0.4, 0.5]
                expected = Float32[0.0171080, 0.1026479, 0.8553989] 
                rmsnorm!(@view(x[begin:3]), @view(x[begin:3]), weight)
                @test (isapprox( @view(x[begin:3]), expected, atol=1e-5) && isapprox(@view(x[4:end]), [-0.05, -2.0]))
            end
        end
        @testset "softmax! Tests" begin
            @testset "Functionality Tests" begin
                # Test when softmax is computed over full length of x
                x = Float32[0.1, 0.3, 2.0, -0.05, -2.0]
                expected = Float32[0.10110752, 0.123493, 0.67599418, 0.08702405, 0.01238127]
                softmax!(x)
                @test isapprox(x, expected)
                
                # Test when softmax is computed over a subset of x, rest of x should remain unchanged
                x = Float32[0.1, 0.3, 2.0, -0.05, -2.0]
                x_v = @view(x[begin:3])
                expected = Float32[0.1122675,  0.13712384, 0.75060866]
                softmax!(x_v)
                @test (isapprox(x[begin:3], expected) && isapprox(x[4:end], [-0.05, -2.0]))
                
                # Test when softmax is computed over zero vector
                x = zeros(Float32, 5)
                expected = Float32[1//5, 1//5, 1//5, 1//5, 1//5]
                softmax!(x)
                @test isapprox(x, expected)
            end
        end
    end
    @testset "Transformer Tests" begin
        # transformer and tokenizer are the same for the forward & generate tests
        config, weights = read_checkpoint("../stories15M.bin")
        transformer = Transformer(config, weights)
        @testset "forward test" begin
            @testset "forward with token=2 (empty string token) and pos=1" begin
                # This test uses the run.c file from https://github.com/karpathy/llama2.c/blob/master/run.c
                # The command "./run stories15M.bin" was used to execute the run.c file
                # The execution was stopped after the first logits are returned from the forward function
                # The corresponding logits are of dimension (vocab_size,) = (32000,) and are stored in empty_prompt_logits_first_iteration.csv
                # The output of the forward function of the transformer.jl are compared with the values stored in empty_prompt_logits_first_iteration.csv
                # To match the setup of the run.c file, the token is set to 1 and the pos is set to 1 (in the run.c file the token is 1 and the pos is 0)
                # The difference in the pos is due to the 1-indexed based system in Julia
                state = RunState(config)
                token = 2
                pos = 1
                forward!(transformer, state, token, pos)
                expected_logits = readdlm("../empty_prompt_logits_first_iteration.csv", Float32)
                @test maximum(abs.(state.logits - expected_logits)) < 1e-4
            end
        end
        
        @testset "generate test" begin
            @testset "generate with prompt \"The universe\"" begin
                # This test uses the run.c file from https://github.com/karpathy/llama2.c/blob/master/run.c
                # The command
                # ./run stories15M.bin -n 22 -t 0.0 -i "The universe"
                # was used for execution of the run.c file, -t 0.0 enables argmax sampling which is deterministic and therefore can be used for comparison
                # capture the output of the generate function and compare it with the expected output coming from the run.c file execution
                tokenizer = build_tokenizer("../tokenizer.bin", Int(config.vocab_size))
                sampler = Sampler(Int(config.vocab_size), 0.0f0, 0.9f0)
                expected_output = "The universe was bright and full of stars. Every night, the stars would twinkle and shine.\n"
                
                # capturing output of genereate funtion
                original_stdout = stdout
                (rd, wr) = redirect_stdout()
                generate(transformer, tokenizer, sampler, 23; prompt="The universe", performance=false)
                redirect_stdout(original_stdout)
                close(wr)
                output = read(rd, String)
                
                @test output == expected_output
            end
        end
    end
end