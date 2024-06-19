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

    text = "hello"
    tokenizer = Tokenizer(text)
    ids = [104, 105]
    decoded_text = decoding(ids, tokenizer.vocab)
    @test decoded_text == expected_output

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


# Test Tokenizer_SentencePiece.jl
@testset "Tokenizer_SentencePiece Tests" begin
    text = "Hallo Hallo, Ich bin Ph1 L0ng Hallo Hallo #mit einem 😊 und 中文"
    
    # Initialisierung des Tokenizers
    tokenizer = Tokenizer_SentencePiece1(text)
    
    # Testen, ob die Länge der vocab_ids korrekt ist
    @test length(tokenizer.vocab_ids) == length(text)
    
    # Testen, ob das Vokabular korrekt erstellt wurde
    @test length(tokenizer.vocab) == length(Set(text))
    
    # Vorheriger Zustand von vocab_ids und vocab
    initial_vocab_ids = copy(tokenizer.vocab_ids)
    initial_vocab_length = length(tokenizer.vocab)
    
    # Anwenden von replace_top_pair!
    replace_top_pair!(tokenizer.vocab_ids, tokenizer.vocab)
    
    # Testen, ob die Länge von vocab_ids korrekt aktualisiert wurde
    @test length(tokenizer.vocab_ids) <= length(initial_vocab_ids)
    
    # Testen, ob das Vokabular erweitert wurde
    @test length(tokenizer.vocab) > initial_vocab_length
    
    # Testen der Encoding-Funktion
    encoded_ids = encoding(text, tokenizer.vocab)
    @test typeof(encoded_ids) == Vector{Int}
    
    # Testen der Decoding-Funktion
    decoded_text = decoding(encoded_ids, tokenizer.vocab)
    @test decoded_text == text
end
