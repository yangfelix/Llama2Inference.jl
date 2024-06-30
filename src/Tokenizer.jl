using DataStructures

struct TokenIndex
    str::String
    id::Int
end

struct Tokenizer
    vocab_size::Int
    vocab::Vector{String}
    vocab_scores::Vector{Float32}
    max_token_length::Int
    sorted_vocab::Union{Nothing, Vector{TokenIndex}}   # can be either nothing or vector of tokens and their indices
    
end

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

# find the id corresponding to a "token"
function find_token_id(tokenizer::Tokenizer, token_str::String)
    for token_index in tokenizer.sorted_vocab
        if token_index.str == token_str
            return token_index.id
        end
    end
    return -1  
end

function find_token_str(tokenizer::Tokenizer, token_id::Int)
    for token_index in tokenizer.sorted_vocab
        if token_index.id == token_id
            return token_index.str
        end
    end
    return nothing
end

#from id to str
#from id to str
function decode(tokenizer::Tokenizer, prev_token::Int, token::Int, BOS::Int )
    token_str = find_token_str(tokenizer,token)
   
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
# we split the encode function for "cleaner" code
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
