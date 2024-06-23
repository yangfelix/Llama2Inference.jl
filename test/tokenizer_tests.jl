using Test
using Llama2Inference

## Andrejs tests that work for run.c
@testset "Encoder Tests" begin
    vocab_size = 32000
    BOS::Int = 2
    EOS::Int = 0
    tokenizer_path = "../tokenizer.bin"
    tokenizer = build_tokenizer(tokenizer_path,vocab_size)
    sv = sort_vocab!(tokenizer)
    
    prompt1 = "I believe the meaning of life is"
    expected_tokens1 = [1,306, 4658, 278, 6593, 310, 2834, 338]
    expected_tokens1 .+= 1 # in julia start index is 1
    @test encode(tokenizer, prompt1, BOS, EOS) == expected_tokens1

    # Test Case 2
    prompt2 = "Simply put, the theory of relativity states that "
    expected_tokens2 = [1,3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871]
    expected_tokens2 .+= 1
    @test encode(tokenizer, prompt2, BOS, EOS) == expected_tokens2

end
