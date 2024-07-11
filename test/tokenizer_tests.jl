using Test
using Llama2Inference

## Andrejs tests that work for run.c
@testset "Encoder Tests" begin
    vocab_size = 32000
    BOS::Bool = true
    EOS::Bool = false
    tokenizer_path = "../bin/tokenizer.bin"
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

    # Test Case 3
    prompt3 = "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just "
    expected_tokens3 = [1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871]
    expected_tokens3 .+= 1 
  
    @test encode(tokenizer, prompt3, BOS, EOS) == expected_tokens3

    # Test Case 4
    prompt4 = "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>"
    expected_tokens4 = [1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149]
    expected_tokens4 .+= 1 

    @test encode(tokenizer, prompt4, BOS, EOS) == expected_tokens4
end
