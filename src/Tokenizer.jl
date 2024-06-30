using DataStructures
"""
    struct TokenIndex
A data structure representing a token and its corresponding index.

# Fields
- `str::String`: The token as a string.
- `id::Int`: The ID of the token.
"""
struct TokenIndex
    str::String
    id::Int
end

"""
    struct Tokenizer
A data structure representing a tokenizer with its vocabulary and related properties.

# Fields
- `vocab_size::Int`: The size of the vocabulary.
- `vocab::Vector{String}`: A vector containing the vocabulary tokens.
- `vocab_scores::Vector{Float32}`: A vector containing scores associated with each token in the vocabulary.
- `max_token_length::Int`: The maximum length of a token.
- `sorted_vocab::Union{Nothing, Vector{TokenIndex}}`: Can be either `nothing` or a vector of `TokenIndex` structs representing the tokens and their respective indices.
"""
struct Tokenizer
    vocab_size::Int
    vocab::Vector{String}
    vocab_scores::Vector{Float32}
    max_token_length::Int
    sorted_vocab::Union{Nothing, Vector{TokenIndex}}   # can be either nothing or vector of tokens and their indices
    
end

"""
    build_tokenizer(filepath::String, vocab_size::Int)

Constructs a `Tokenizer` from a file.

# Arguments
- `filepath::String`: Path to the file containing vocabulary data.
- `vocab_size::Int`: Size of the vocabulary to read from the file.

# Returns
- `Tokenizer`: A `Tokenizer` instance with the necessary data.

# Summary Steps
1. Initialization
2. Reading file
3. Handling errors
4. Closing file
5. Returning `Tokenizer` with necessary data
"""

function build_tokenizer(filepath::String, vocab_size::Int)
    vocab = Vector{String}(undef, vocab_size)
    vocab_scores = Vector{Float32}(undef, vocab_size)
    max_token_length = 0
    
    file = open(filepath)
    try
        
        max_token_length = read(file, Int32)
        for i in 1:vocab_size
            vocab_scores[i] = read(file, Float32)
            len = read(file, Int32)
            vocab[i] = String(read(file, len))
        end
    catch e
        close(file)
        rethrow(e)
    end
    close(file)

    sorted_vocab = Vector{Nothing}()
    return Tokenizer(vocab_size, vocab, vocab_scores, max_token_length,sorted_vocab)
end

"""
    sort_vocab!(tokenizer::Tokenizer)

Sorts the vocabulary of the given `Tokenizer`, storing unique tokens and sorting them.

# Argument
- `tokenizer::Tokenizer`: The `Tokenizer` whose vocabulary is to be sorted.

# Summary Steps
1. Checking if `sorted_vocab` is `nothing` or empty.
2. Initialization of data structures.
3. Identification of unique tokens.
4. Sorting the vocabulary.
"""

function sort_vocab!(tokenizer::Tokenizer)
   
    if isnothing(tokenizer.sorted_vocab) || isempty(tokenizer.sorted_vocab)
        #tokenizer.sorted_vocab = Vector{TokenIndex}()
        seen_strings = Set{String}()

        for i in 1:tokenizer.vocab_size
            str = tokenizer.vocab[i]
            if str in seen_strings
                continue  # skip if we've already seen this string
            end

            push!(seen_strings, str)
            push!(tokenizer.sorted_vocab, TokenIndex(str, i))
        end

        sort!(tokenizer.sorted_vocab, by = x -> x.str)
    end
end

"""
    find_token_id(tokenizer::Tokenizer, token_str::String) 

Finds and returns the ID of a given token string in the `sorted_vocab`.

# Arguments
- `tokenizer::Tokenizer`: The `Tokenizer` to search within.
- `token_str::String`: The token string to search for its ID.

# Return
- `Int`: The ID of the token if found, otherwise returns `-1`.

# Description
Iterates over the `sorted_vocab` to find the token string and return its ID.
"""

function find_token_id(tokenizer::Tokenizer, token_str::String)
    for token_index in tokenizer.sorted_vocab
        if token_index.str == token_str
            return token_index.id
        end
    end
    return -1  
end

"""
   function find_token_str(tokenizer::Tokenizer, token_id::Int)
Finds and returns the token string coressponding to a given token ID from `sort_vocab`

# Arguments
- `tokenizer::Tokenizer`: `Tokenizer` to search within
- `token_id::Int`: ID of the given token to find its coressponding string

# Return
- `token_index.str`: String of the token if found, otherwise returns `nothing`

# Description
Iterates over the `sorted_vocab` to find the token ID and return its corresponding string.
"""

function find_token_str(tokenizer::Tokenizer, token_id::Int)
    for token_index in tokenizer.sorted_vocab
        if token_index.id == token_id
            return token_index.str
        end
    end
    return nothing
end

"""
    decode(tokenizer::Tokenizer, prev_token::Int, token::Int) -> Union{String, UInt8}

Decodes a token ID into its corresponding token string representation or byte value using the `Tokenizer`.

# Arguments
- `tokenizer::Tokenizer`: The `Tokenizer` containing vocabulary and token mapping.
- `prev_token::Int`: Token ID of the previous token in the sequence.
- `token::Int`: Token ID to decode into its corresponding token string representation.

# Returns
- `byte_val` or `token_str`: Returns a decoded string or a byte value, depending on the type of the given token.

# Description
1. Finds the token string using `find_token_str`.
2. Removes leading whitespace if the previous token is `BOS`.
3. Checks for raw byte tokens and parses them if applicable.
4. Returns the token string or its byte value representation.
"""


function decode(tokenizer::Tokenizer, prev_token::Int,token::Int)
    BOS = 2
    token_str = find_token_str(tokenizer,token)

    if token_str == "<0x0A>"
        token_str = "\n"
    end
   
    # following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if prev_token == BOS && token_str[1] == ' '
        token_str = token_str[2:end]   # example for me "text" -> "ext"
    end
    
    # check for raw bytes tokens
    if startswith(token_str, "<") && endswith(token_str, ">")
        # remove '<' and '>' to gethexadecimal format 
        hex_str = token_str[3:end-1]  # example for me "<0x01>" will be "x01"
        
        # Parse the hexadecimal string into a UInt8 byte
        byte_val = parse(UInt8, hex_str, base=16)
        
        return byte_val
    else
        return token_str
    end

end

"""
    encode(tokenizer::Tokenizer, text::String, use_bos::Bool, use_eos::Bool) -> Vector{Int}

Encodes the input text into a sequence of token IDs using the provided `Tokenizer`.

# Arguments
- `tokenizer::Tokenizer`: The `Tokenizer` containing vocabulary and token mapping.
- `text::String`: Input text to encode into tokens.
- `use_bos::Bool`: Indicates if a token ID representing the beginning of the sequence should be included.
- `use_eos::Bool`: Indicates if a token ID representing the end of the sequence should be included.

# Returns
- `Vector{Int}`: Returns a vector of token IDs representing the encoded input text.

# Description
1. Ensures the tokenizer's vocabulary is sorted with `sort_vocab!`.
2. Encodes text into bytes.
3. Initializes the token ID vector.
4. Optionally adds the BOS token.
5. Handles leading whitespace.
6. Looks up and stores token IDs with `find_token_id`.
7. Performs merges (BPE) based on scores (`vocab_scores`).
8. Optionally adds the EOS token.
9. Returns the vector `tokens_indices` representing the encoded input text.
"""

function encode(tokenizer::Tokenizer, text::String, use_bos::Bool, use_eos::Bool)
    
    sort_vocab!(tokenizer)   # Ensure tokenizer's vocabulary is sorted
    text_bytes = StringEncodings.encode(text, "utf-8")  # Convert text to bytes

    tokens_indices = Vector{Int}()

    if use_bos
        push!(tokens_indices, 2)
    end
    # Handle whitespace token if text is non-empty
    if text != ""
        token_str = " "
        token_id = find_token_id(tokenizer, token_str)
        push!(tokens_indices, token_id)
    end

    # Lookup each byte token in the tokenizer's vocabulary and store its ID
    for token in text_bytes
        token_str = String([token])
        token_id = find_token_id(tokenizer, token_str)
        if token_id != -1
            push!(tokens_indices, token_id)
        else
            # Handle unknown tokens
            push!(tokens_indices, -1)  # Use -1 for unknown tokens
        end
    end

    n_tokens = length(tokens_indices)

    # Perform merges (BPE) based on scores
    while true
        best_score = -1e10
        best_id = -1
        best_idx = -1
        merged_str = ""

        for i in 1:(n_tokens - 1)
            token1 = tokens_indices[i]
            token2 = tokens_indices[i+1]
            
            merged_str = tokenizer.vocab[token1] * tokenizer.vocab[token2]

            merged_token_id = find_token_id(tokenizer, merged_str)

            # Check if new merged token exists and its score is better than the current one
            if merged_token_id != -1 && tokenizer.vocab_scores[merged_token_id] > best_score
                best_score = tokenizer.vocab_scores[merged_token_id]
                best_id = merged_token_id
                best_idx = i
            end
        end

        # No more merges possible
        if best_idx == -1
            break
        end

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens_indices[best_idx] = best_id
        deleteat!(tokens_indices, best_idx + 1)

        n_tokens -= 1  # Update the number of tokens after merge
    end

    # Optionally add EOS token
    if use_eos
        push!(tokens_indices, 3)
    end

    return tokens_indices
end
