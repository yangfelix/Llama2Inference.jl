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
    # dimensions of the weights, but they need to be inverted to match the order in the binary file (C stores arrays row-wise, Julia column-wise)!
    dims = [
        (config.dim, config.vocab_size), # token_embedding_table
        (config.dim, n_layers), # rms_att_weight
        (config.dim, config.n_heads * head_size, n_layers), # wq
        (config.dim, config.n_kv_heads * head_size, n_layers), # wk
        (config.dim, config.n_kv_heads * head_size, n_layers), # wv
        (config.dim, config.n_heads * head_size, n_layers), # wo
        (config.dim, n_layers), # rms_ffn_weight
        (config.dim, config.hidden_dim, n_layers), # w1
        (config.hidden_dim, config.dim, n_layers), # w2
        (config.dim, config.hidden_dim, n_layers), # w3
        (config.dim,), # rms_final_weight
        (head_size, config.seq_len), # skip
        (config.dim, config.vocab_size), # wcls
    ]
    # if there are no shared offsets then the last offset causes BoundsError so it has to be left out
    length_offsets = shared_weights == 0 ? length(offsets) - 1 : length(offsets) - 2
    # weights are in the same order in the binary file as in the struct definition
    split_weights::Vector{AbstractArray} = [get_weights(weights, offsets[i], offsets[i+1], dims[i]) for i in 1:length_offsets]
    token_embedding_table = split_weights[1]

    wcls = shared_weights == 0 ? split_weights[end] : split_weights[begin]

    # permutation of dimensions is needed to match the original order of dimensions
    token_embedding_table = permutedims(token_embedding_table)
    rms_att_weight = permutedims(split_weights[2])
    wq = permutedims(split_weights[3], (3,2,1))
    wk = permutedims(split_weights[4], (3,2,1))
    wv = permutedims(split_weights[5], (3,2,1))
    wo = permutedims(split_weights[6], (3,2,1))
    rms_ffn_weight = permutedims(split_weights[7])
    w1 = permutedims(split_weights[8], (3,2,1))
    w2 = permutedims(split_weights[9], (3,2,1))
    w3 = permutedims(split_weights[10], (3,2,1))
    rms_final_weight = split_weights[11]
    wcls = permutedims(wcls)

    return TransformerWeights(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls)
end

function get_weights(weights::Vector, offset_1, offset_2, dims)::AbstractArray
    w = @view weights[begin+offset_1:begin+offset_2-1]
    return reshape(w, dims)
end
