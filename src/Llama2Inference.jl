module Llama2Inference
using StringEncodings: encode
using DataStructures
export Tokenizer
using Base.Iterators: partition


export replace_top_pair!,Tokenizer,get_most_common_pair,count_consecutive,decoding,encoding
include("Tokenizer.jl")

struct Config
    dim::Int
    hidden_dim::Int
    n_layers::Int
    n_heads::Int
    n_kv_heads::Int
    vocab_size::Int
    seq_len::Int
end

struct TransformerWeights
    token_embedding_table::Matrix{Float64}
    rms_att_weight::Matrix{Float64}
    rms_ffn_weight::Matrix{Float64}
    wq::Array{Float64,3}
    wk::Array{Float64,3}
    wv::Array{Float64,3}
    wo::Array{Float64,3}
    w1::Array{Float64,3}
    w2::Array{Float64,3}
    w3::Array{Float64,3}
    rms_final_weight::Vector{Float64}
    freq_cis_real::Matrix{Float64}
    freq_cis_imag::Matrix{Float64}
end

struct RunState
    x::Vector{Float32}
    xb::Vector{Float32}
    xb2::Vector{Float32}
    hb::Vector{Float32}
    hb2::Vector{Float32}
    q::Vector{Float32}
    k::Vector{Float32}
    v::Vector{Float32}
    att::Vector{Float32}
    logits::Vector{Float32}
    key_cache::Array{Float32,3}
    value_cache::Array{Float32,3}
end

struct Transformer
    config::Config
    weights::TransformerWeights
    state::RunState
end

function forward(transformer::Transformer, token::Int64, pos::Int64)
    # some convenience variables
    config = transformer.config
    weights = transformer.weights
    state = transformer.state
    # line 72 overwrites x before using it
    # x = state.x
    dim = config.dim
    kv_dim = (config.dim * config.n_kv_heads) / config.n_heads
    kv_mul = config.n_heads / config.n_kv_heads # integer multiplier of the kv sharing in multiquery
    hidden_dim = config.hidden_dim
    head_size = dim / config.n_heads

    # copy token embedding into x
    x = weights.token_embedding_table + token * dim

    # forward all layer
    # line 249
    for nothing in nothing
        
    end
end

function generate(transformer::Transformer, tokenizer::Tokenizer, sampler, steps:Integer; prompt::String="")
    # start with the input text in prompt
    prompt_tokens = encoding(prompt, tokenizer.vocab) # return Vector{Int64} containing the ids (tokens?)
    num_prompt_tokens = length(prompt_tokens)
    if num_prompt_tokens < 1
        throw(error("length of prompt_tokens is $(num_prompt_tokens)!"))
    end

    # start the main loop
    next = nothing
    token = prompt_tokens[1]
    pos = 1 # Julia is 1 vs. C is 0

    while pos < steps
        # forward the transformer to get logits for the next token
        logits = forward(transformer, token, pos)

        # advance the state machine
        if (pos < num_prompt_tokens)
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1]
        else
            # otherwise sample the next token from the logits
            next = sample(sampler, logits)
        end
        pos += 1

        # data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1)
            break
        end

        # print the token as string, decode it with the Tokenizer object
        piece = decoding(token, tokenizer.vocab)
        print(piece) # same as printf("%s", piece), but skips "unsafe" bytes
        token = next
    end

    println("")
end

end
