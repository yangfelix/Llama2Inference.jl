struct TransformerWeights{T<:AbstractFloat}
    # token embedding table
    token_embedding_table::AbstractArray{T,2} # (vocab_size, dim)
    # weights for rmsnorms att
    rms_att_weight::AbstractArray{T,2} # (layer, dim) 
    # weights for matmuls.
    # Note:
    #   dim == n_heads * head_size
    #   kv_dim == n_kv_heads * head_size
    wq::AbstractArray{T,3} # (layer, dim, dim)
    # corresponding to matmul function call in line 146,
    # the dimensionality in the comment of run.c is wrong, the last 2 dimensions got swapped
    wk::AbstractArray{T,3} # (layer, kv_dim, dim) 
    wv::AbstractArray{T,3} # (layer, kv_dim, dim)
    wo::AbstractArray{T,3} # (layer, dim, dim)
    # weights for rmsnorms ffn
    rms_ffn_weight::AbstractArray{T,2} # (layer, dim)
    # weights for ffn
    w1::AbstractArray{T,3} # (layer, hidden_dim, dim)
    w2::AbstractArray{T,3} # (layer, dim, hidden_dim)
    w3::AbstractArray{T,3} # (layer, hidden_dim, dim)
    # final rmsnorm
    rms_final_weight::AbstractArray{T,1} # (dim,)
    # (optinoal) classifier weights for the logits, on the last layer
    wcls::AbstractArray{T,2} # (vocab_size, dim)
end

function memory_map_weights(
    config::Config,
    weights::Vector{Float32},
    shared_weights::Int32
)::TransformerWeights
    # making sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models   
    n_layers::Int64 = config.n_layers
    # precalculate some useful values
    head_size::Int = config.dim / config.n_heads
    dimlayers = n_layers * config.dim
    # starting/ending offsets for all the weights in binary file
    # the second to last offset (config.seq_len * head_size) is for compatibility reasons to run.c and is not relevant anymore
    offsets::Vector{UInt64} = cumsum([
        0,
        config.vocab_size * config.dim,
        dimlayers,
        dimlayers * (config.n_heads * head_size),
        dimlayers * (config.n_kv_heads * head_size),
        dimlayers * (config.n_kv_heads * head_size),
        dimlayers * (config.n_heads * head_size),
        dimlayers,
        dimlayers * config.hidden_dim,
        dimlayers * config.hidden_dim,
        dimlayers * config.hidden_dim,
        config.dim,
        config.seq_len * head_size,
        config.vocab_size * config.dim
    ])
    dims = [
        (config.vocab_size, config.dim), # token_embedding_table
        (n_layers, config.dim), # rms_att_weight
        (n_layers, config.dim, config.n_heads * head_size), # wq
        (n_layers, config.n_kv_heads * head_size, config.dim), # wk
        (n_layers, config.n_kv_heads * head_size, config.dim), # wv
        (n_layers, config.n_heads * head_size, config.dim), # wo
        (n_layers, config.dim), # rms_ffn_weight
        (n_layers, config.hidden_dim, config.dim), # w1
        (n_layers, config.dim, config.hidden_dim), # w2
        (n_layers, config.hidden_dim, config.dim), # w3
        (config.dim,), # rms_final_weight
        (config.seq_len, head_size), # skip
        (config.vocab_size, config.dim), # wcls
    ]
    # if there are no shared offsets then the last offset causes BoundsError so it has to be left out
    length_offsets = shared_weights == 0 ? length(offsets) - 1 : length(offsets) - 2
    # weights are in the same order in the binary file as in the struct definition
    split_weights::Vector{AbstractArray} = [get_weights(weights, offsets[i], offsets[i+1], dims[i]) for i in 1:length_offsets]
    token_embedding_table = split_weights[1]

    wcls = shared_weights == 0 ? split_weights[end] : split_weights[begin]
    return TransformerWeights(split_weights[1], split_weights[2], split_weights[3], split_weights[4], split_weights[5], split_weights[6], split_weights[7], split_weights[8], split_weights[9], split_weights[10], split_weights[11], wcls)
end

function get_weights(weights::Vector, offset_1, offset_2, dims)::AbstractArray
    w = @view weights[begin+offset_1:begin+offset_2-1]
    return reshape(w, dims)
end
