using DataStructures

"""
    struct Tokenizer

Tokenizer struct that represents a text tokenizer.

# Fields
- `text::String`: The input text that will be used for training.
- `vocab_bytes::Vector{UInt8}`: Byte representation of the provided text.
- `vocab_ids::Vector{Int}`: IDs corresponding to the text bytes.
- `vocab::OrderedDict{Int, Vector{UInt8}}`: Ordered dictionary mapping the IDs from 0 to 255 to bytes.

# Constructor
`Tokenizer(text::String)`: Constructs a new `Tokenizer` object.

# Examples
```julia
text = "hello"
tokenizer = Tokenizer(text)

"""

struct Tokenizer
    text::String  # our vocabulary
    vocab_bytes::Vector{UInt8}
    vocab_ids::Vector{Int}
    vocab::OrderedDict{Int, Vector{UInt8}}
    
    function Tokenizer(text::String)

        # simple example for me:
        # text = "hi" , vocab_bytes = [0x68,0x69] , vocab_ids = [104,105] , UInt8(104) = 0x68, String(vocab_bytes) = "hi" 

        vocab_bytes = encode(text, "utf-8")  # convert text to unicode  
        vocab_ids = [Int(byte) for byte in vocab_bytes]

        # idx => [byte] , e.g (0 => [0x00], 1 => [0x01]...)
        vocab = OrderedDict(idx => UInt8[idx] for idx in 0:255)  
        new(text, vocab_bytes,vocab_ids, vocab)
    end

end


"""
    count_consecutive(vocab_ids::Vector{Int})

Count consecutive pairs of vocabulary IDs in the training data.

# Arguments
- `vocab_ids::Vector{Int}`: Vector of the trainings text's vocabulary IDs.

# Returns
- pair_dict::Dict: Dictionary containing number of occurrences of each consecutive pairs.

# Example 
text = "hello"
tokenizer = Tokenizer(text)
pair_dict = count_consecutive(tokenizer.vocab_ids)
"""

# function that counts most common consecutive pairs

function count_consecutive(vocab_ids::Vector{Int})
    pair_dict = Dict{Tuple{Int, Int}, Int}()
    
    # iterate over the ids, excluding the last id and get number of occurrences of all consecutive pairs
    for i in 1:length(vocab_ids)-1
        id1 = vocab_ids[i]
        id2 = vocab_ids[i+1]
        pair = (id1,id2)
        
        # If the pair exists in the dictionary, increment its count
        if haskey(pair_dict, pair)
            pair_dict[pair] += 1
        else
            pair_dict[pair] = 1
        end
    end
    return pair_dict
end

"""
    get_most_common_pair(vocab_ids::Vector{Int})

Get the most common consecutive pair of vocabulary IDs in the training data.

# Arguments
- `vocab_ids::Vector{Int}`: Vector of the trainings text's vocabulary IDs.

# Returns
- `top_pair::Tuple{Int, Int}, max_count::Int`: Tuple containing the most common pair and its count.

# Calls 
- `count_consecutive` : Gets the dictionary containing number of occurrences of each consecutive pairs.

# Examples

text = "hello"
tokenizer = Tokenizer(text)
top_pair, max_count = get_most_common_pair(tokenizer.vocab_ids)

"""
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

"""
    replace_top_pair!(vocab_ids::Vector{Int}, vocab::OrderedDict{Int, Vector{UInt8}})

Replaces the 20 most common consecutive pairs with a new vocabulary ID.

# Arguments
- `vocab_ids::Vector{Int}`: Vector of the trainings text's vocabulary IDs.
- `vocab::OrderedDict{Int, Vector{UInt8}}`: Ordered dictionary mapping IDs to vocabulary bytes.

# Calls 
- `get_most_common_pair` : Gets the most common pair.

# Examples
text = "hello"
tokenizer = Tokenizer(text)
replace_top_pair!(tokenizer.vocab_ids, tokenizer.vocab)

"""
function replace_top_pair!(vocab_ids::Vector{Int}, vocab::OrderedDict{Int, Vector{UInt8}})
    desired_vocab_size = 276
    num_merges = desired_vocab_size - 256
    new_id = 256

    while num_merges > 0
        top_pair, max_count = get_most_common_pair(vocab_ids)
        
        if max_count > 1
            (id1, id2) = top_pair
            vocab[new_id] = vcat(vocab[id1], vocab[id2])  # add new e.g 256 => [0x04,0x08] to the vocabulary
      
            i = 1
            j = 1

            while i <= length(vocab_ids)
                if i < length(vocab_ids) && vocab_ids[i] == id1 && vocab_ids[i+1] == id2
                    vocab_ids[j] = new_id
                    i += 2
                else
                    vocab_ids[j] = vocab_ids[i]
                    i += 1
                end
                j += 1
            end

            resize!(vocab_ids, j - 1)
            num_merges -= 1
            new_id += 1
        else
            break
        end 
    end

end

"""
    decoding(ids::Vector{Int}, vocab::OrderedDict{Int, Vector{UInt8}})

Decode a sequence of vocabulary IDs into text.

# Arguments
- `ids::Vector{Int}`: Vector of text IDs.
- `vocab::OrderedDict{Int, Vector{UInt8}}`: Ordered dictionary mapping IDs to vocabulary bytes.

# Returns
- `text::String`: Decoded text.

# Examples
text = "hello"
tokenizer = Tokenizer(text)
ids = [104, 105]
decoded_text = decoding(ids, tokenizer.vocab)
"""
# ids to text, vocab is the new vocab where the consecutive pairs are alreay merged
function decoding(ids::Vector{Int},vocab::OrderedDict{Int, Vector{UInt8}})
    # simple example for me:
    # vocab = (0 => [0x00], 1 => [0x01].., 258 => [7,0x17])
    # text = "hi" , vocab_bytes = [0x68,0x69] , vocab_ids = [104,105] , UInt8(104) = 0x68, String(vocab_bytes) = "hi" 
    
    text_bytes = Vector{UInt8}()

    for id in ids
        for i in vocab[id]
            push!(text_bytes, i)   # push the byte(s) corresponding to this id
        end
    end

    return String(text_bytes)
end

"""
    encoding(str_text::String, vocab::OrderedDict{Int, Vector{UInt8}})

Encode text into a sequence of vocabulary IDs.

# Arguments
- `str_text::String`: Input text.
- `vocab::OrderedDict{Int, Vector{UInt8}}`: Ordered dictionary mapping IDs to vocabulary bytes.

# Returns
- `ids::Vector{Int}`: Encoded vocabulary IDs.

# Examples
1.
text = "this is a this test text with some repeated text  with some with some text"
tokenizer = Tokenizer(text)
replace_top_pair!(tokenizer.vocab_ids,tokenizer.vocab)
str_text = "this"  
ids = encoding(str_text,tokenizer.vocab)

2.
decoding(encoding("hello",tokenizer.vocab),tokenizer.vocab) == "hello"
"""
# text to ids, vocab is the new vocab where the consecutive pairs are alreay merged
function encoding(str_text::String, vocab::OrderedDict{Int, Vector{UInt8}})
    text_bytes = encode(str_text, "utf-8")  # convert text to unicode  
    text_ids = [Int(byte) for byte in text_bytes] # get ids of text

    merges = OrderedDict(filter(kv -> length(kv[2]) > 1, vocab)) # get only the merges e.g 258 => [7,0x17]

    ids = Int[] # ids after applying the merged ids
    i = 1

    while i <= length(text_ids)
        found_merge = false

        for (id, merge) in merges
            merge_length = length(merge)
            if i + merge_length - 1 <= length(text_ids) && text_ids[i:i+merge_length-1] == merge
                push!(ids, id)
                i += merge_length
                found_merge = true
                break
            end
        end

        if !found_merge
            push!(ids, text_ids[i])
            i += 1
        end
    end

    return ids
end





