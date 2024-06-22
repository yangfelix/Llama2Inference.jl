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
    
    file = open(filepath, "r")
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

    return Tokenizer(vocab_size, vocab, vocab_scores, max_token_length)
end

function sort_vocab!(tokenizer::Tokenizer)
    if tokenizer.sorted_vocab === nothing
        tokenizer.sorted_vocab = Vector{TokenIndex}()
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


function decode(tokenizer::Tokenizer, prev_token::Int, token::Int)
    token_str = find_token_str(tokenizer,token)
    # following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if prev_token == BOS_TOKEN && token_str[1] == ' '
        token_str = token_str[2:end]   # example for me "text" -> "ext"
    end

    # check for raw bytes tokens
    if startswith(token, "<") && endswith(token, ">")
        # remove '<' and '>' to gethexadecimal format 
        hex_str = token[3:end-1]  # example for me "<0x01>" will be "x01"
        
        # Parse the hexadecimal string into a UInt8 byte
        byte_val = parse(UInt8, hex_str, base=16)
        
        return byte_val
    else
        error("Invalid token format: $token")
    end

end
# we split the encode function for "cleaner" code
function encode(tokenizer::Tokenizer, text::String, BOS::Int32, EOS::Int32)
    sort_vocab!(tokenizer)   # check if each token is already mapped to an index

    text_bytes = StringEncodings.encode(text, "utf-8")  # convert text to unicode  

    tokens_indices = Vector{TokenIndex}()

    # optionally add the BOS token
    if BOS != 0
        push!(tokens, BOS)
    end

    if text != ""
        token_str = " "
        token_id = find_token_id(tokenizer, token_str)
        push!(tokens_indices, TokenIndex(token_str, token_id))
    end

    # lookup each token in the tokenizer's vocabulary and store its ID
    for token in text_bytes
        token_str = String(token)
        token_id = find_token_id(tokenizer, token_str)
        if token_id != -1
            push!(tokens_indices, TokenIndex(token_str, token_id))
        else
            # handle unknown tokens 
            push!(tokens_indices, TokenIndex(nothing, -1)) 
        end
    end

    n_tokens = length(tokens_indices)

    # perform merges (bpe) based on scores
    while true
        best_score = -1e10;
        best_id = -1;
        best_idx = -1;
        merged_str;

        for i in 1:(n_tokens - 1)
            token1 = tokens_indices[i].str
            token2 = tokens_indices[i+1].str
            
            merged_str = token1*token2
            merged_token_id = find_token_id(tokenizer, merged_str)
            # check if new merged token exists and its score is better than the current one
            if(merged_token_id != -1 && tokenizer.vocab_scores[merged_token_id]  > best_score){
                best_score = tokenizer.vocab_scores[merged_token_id];
                best_id = merged_token_id;
                best_idx = i;
            }
        end
        # no more merges possible
        if best_idx == -1
            break
        end

        # merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens_indices[best_idx] = TokenIndex(merged_str,best_id) # replace the first token index with the merged one
        splice!(tokens_indices, best_idx + 1)  # delete the second token index 
        n_tokens = length(tokens_indices)

    end

    if EOS != 0
        push!(tokens, BOS)
    end

    # extract ids
    ids = [token_index.id for token_index in tokens_indices]
    return ids
end




