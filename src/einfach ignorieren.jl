using DataStructures

struct Tokenizer_SentencePiece3
    text::String  # our vocabulary
    #vocab_bytes::Vector{UInt8}
    vocab_ids::Vector{Int}
    vocab::OrderedDict{Int, String}
    vocab_bytes::Vector{UInt8}

    function Tokenizer_SentencePiece3(text::String)
        #vocab_bytes = Vector{UInt8}(text)
        vocab_ids = collect(Int,text)
        #vocab_ids = collect(Int, vocab_bytes)

        vocab = OrderedDict{Int, String}()
        for char in text
            #neu (eine Zeile)
            if !haskey(vocab,Int(char))
                vocab[Int(char)] = string(char)
            end
        #new(text, vocab_ids, vocab)
        end
        # for byte in vocab_bytes  
        #     if !haskey(vocab, Int(byte))
        #         vocab[Int(byte)] = string(Char(byte)) 
        #     end
        # end
        
    #NEU 
    # Hinzufügen von Sonderzeichen und Leerzeichen
        # if !haskey(vocab, Int(' '))
        #   vocab[Int(' ')] = " "
        # end
        # special_tokens = ["<UNK>", "<PAD>", "<EOS>", "<BOS>"]
        # for token in special_tokens
        #     vocab[maximum(keys(vocab)) + 1] = token
        # end

        new(text, vocab_ids, vocab)
    end
end


function count_consecutive(vocab_ids::Vector{Int})
    pair_dict = Dict{Tuple{Int, Int}, Int}()
    
    # iterate over the ids, excluding the last id and get number of occurrences of all consecutive pairs
    for i in 1:length(vocab_ids)-1
        id1 = vocab_ids[i]
        id2 = vocab_ids[i+1]
        pair = (id1,id2)
        pair_dict[pair] = get(pair_dict, pair, 0) +1 # neu
        # If the pair exists in the dictionary, increment its count - alt
        # if haskey(pair_dict, pair)
        #     pair_dict[pair] += 1
        # else
        #     pair_dict[pair] = 1
        # end
    end
    return pair_dict
end


function get_most_common_pair(vocab_ids::Vector{Int})
    pair_dict = count_consecutive(vocab_ids)
    top_pair = nothing
    max_count = 0
    
    for (pair, count) in pair_dict
        if count > max_count
            top_pair = pair
            max_count = count
        end
    end
    
    return top_pair,max_count
end


function replace_top_pair!(vocab_ids::Vector{Int}, vocab::OrderedDict{Int, String})
    desired_vocab_size = 160000
    num_merges = desired_vocab_size - length(vocab)
    new_id = maximum(keys(vocab)) + 1

    while num_merges > 0
        top_pair, max_count = get_most_common_pair(vocab_ids)
        
        if max_count > 1
            (id1, id2) = top_pair
            new_token = vocab[id1] * vocab[id2]
            vocab[new_id] = new_token
            i = 1

            while i <= length(vocab_ids)-1
                if  vocab_ids[i] == id1 && vocab_ids[i+1] == id2
                    vocab_ids[i] = new_id
                    deleteat!(vocab_ids, i+1)
                    #neu (1Zeile)
                    i = max(i - 1, 1)
                else
                    i += 1
                end
            end
            num_merges -= 1
            new_id += 1
        else
            break
        end 
    end

end

function decoding(ids::Vector{Int},vocab::OrderedDict{Int, String})
    tokens = Vector{String}()
    for id in ids
        if haskey(vocab, id)
            push!(tokens, vocab[id])
        else
            push!(tokens, "<ERROR>")
        end
    end
    return join(tokens)
end


# function repair_invalid_utf8(text::String)::String
#     repaired_text = IOBuffer()
#     i = 1
#     while i <= lastindex(text)
#         try
#             c = text[i]
#             String([c])  # Try to decode single byte
#             write(repaired_text, c)
#             i += 1
#         catch e
#             if isa(e, Base.InvalidCharError)
#                 write(repaired_text, UInt8(0xEF))  # � in UTF-8 is 0xEF 0xBF 0xBD
#                 write(repaired_text, UInt8(0xBF))
#                 write(repaired_text, UInt8(0xBD))
#                 i += 1
#             else
#                 rethrow(e)
#             end
#         end
#     end
#     return String(take!(repaired_text))
# end





function encoding(str_text::String, vocab::OrderedDict{Int, String})
    #convert input text into unicode
    text_ids =  collect(Int, str_text)
    #println("Text IDs: ", text_ids)
    #println("Vocab: ", vocab)
    #filter only merges from vocab
    merges = filter(kv -> length(kv[2]) > 1, vocab) # get only the merges e.g 258 => [7,0x17]
    #println("Merges: ", merges)
    ids = Int[] # ids after applying the merged ids
    i = 1

    while i <= length(text_ids)
        found_merge = false
        # check all merges, if they are in text
        for (id, merge) in merges
            merge_chars = collect(Int, merge)
            merge_length = length(merge_chars)
            #if yes then add id to it
            if i + merge_length - 1 <= length(text_ids) && text_ids[i:i+merge_length-1] == merge_chars
                push!(ids, id)
                i += merge_length
                found_merge = true
                break
            end
        end
        #no merges then take current id
        if !found_merge
            push!(ids, text_ids[i])
            i += 1
        end
    end

    return ids
end

#for test: do tokenizer, use replace_top_pair and encoded or encoded+decoded function
function usecase(text::String, action::Int)
    #repaired_text = repair_invalid_utf8(text)
    #tokenizer_test = Tokenizer_SentencePiece1(repaired_text)
    tokenizer_test = Tokenizer_SentencePiece3(text)
    replace_top_pair!(tokenizer_test.vocab_ids, tokenizer_test.vocab)
    if action == 1
        #println("Encoded: ", encoding(repaired_text, tokenizer_test.vocab))
        println("Encoded: ", encoding(text, tokenizer_test.vocab))
    elseif action == 2
        #println("Decoded: ", decoding(encoding(repaired_text, tokenizer_test.vocab), tokenizer_test.vocab))
        println("Decoded: ", decoding(encoding(text, tokenizer_test.vocab), tokenizer_test.vocab))
    else
        error("Invalid action, Use 1 for encoding and 2 for decoding")
    end
end

#test
text2 = "Hallo Hallo, Ich bin Ph1 L0ng Hallo Hallo #mit einem 😊 und 中文"
text = "\xC3\x28"
tokenizer_test = Tokenizer_SentencePiece3(text2)
replace_top_pair!(tokenizer_test.vocab_ids, tokenizer_test.vocab)
println("Encoded: ", encoding(text2, tokenizer_test.vocab))
println("Decoded: ", decoding(encoding(text2, tokenizer_test.vocab), tokenizer_test.vocab))
usecase(text,2)