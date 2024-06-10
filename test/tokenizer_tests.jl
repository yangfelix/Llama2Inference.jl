using Test
using Llama2Inference

#### currently very basic tests just to ensure that the functions are working correctly

# Test Tokenizer
@testset "Tokenizer Tests" begin
    text = "this is a test text with some repeated text"
    tokenizer = Tokenizer(text)
    @test length(tokenizer.vocab_ids) == length(tokenizer.vocab_bytes)
end

# Test count_consecutive
@testset "Count Consecutive Pairs Tests" begin
    vocab_ids = [1, 2, 3, 1, 2, 1, 2, 3, 4]
    pair_dict = count_consecutive(vocab_ids)
    
    @test pair_dict[(1, 2)] == 3
    @test pair_dict[(2, 3)] == 2
    @test pair_dict[(3, 1)] == 1
    @test pair_dict[(2, 1)] == 1
    @test pair_dict[(3, 4)] == 1
end

#Test get_most_common_pair
@testset "Most Common Pair Tests" begin
    vocab_ids = [1, 2, 3, 1, 2, 1, 2, 3, 4]
    top_pair, count = get_most_common_pair(vocab_ids)
    
    @test top_pair == (1, 2)
    @test count == 3
end

# Test replace_top_pair!
@testset "Replace Top Pair Tests" begin
    vocab_ids = [2, 3, 67, 90, 2, 3, 4]

    vocab = OrderedDict(idx => UInt8[idx] for idx in 0:255)  

    Llama2Inference.replace_top_pair!(vocab_ids,vocab)
    
    @test vocab_ids == [256, 67, 90, 256, 4]
end

@testset "Decoding" begin
    # Define the vocabulary
    vocab = OrderedDict(idx => UInt8[idx] for idx in 0:255)

    # Define the ids (example from the description)
    ids = [104,105]

    # Expected output
    expected_output = "hi"

    # Call the function with the provided example
    actual_output = decoding(ids, vocab)

    # Check if the actual output matches the expected output
    @test actual_output == expected_output

end


@testset "Test Encoding and all other functions" begin
    text = "this is a this test text with some repeated text  with some with some text"
    tokenizer = Tokenizer(text)
    vocab = tokenizer.vocab
    vocab_ids = tokenizer.vocab_ids
    Llama2Inference.replace_top_pair!(vocab_ids,vocab)

    str_text = "this"   # 116,104,105,115; after merge => 257,259

    ids_actual = Llama2Inference.encoding(str_text,vocab)
    ids_expected = [257,259]
  
    text_outp = Llama2Inference.decoding(ids_actual,vocab)

    @test Llama2Inference.decoding(Llama2Inference.encoding("hello",vocab),vocab) == "hello"  # test for text that tokenizer has not seen
    @test text_outp == str_text
    @test ids_actual == ids_expected

end


