"""

    TransformerWeights{T<:AbstractFloat}

Struct to hold the weights of the [`Transformer`](@ref) used for inference.

It is possible to construct a `TransformerWeights` using the constructor 

    TransformerWeights{T<:AbstractFloat}(
        token_embedding_table::AbstractArray{T,2}, 
        rms_att_weight::AbstractArray{T,2}, 
        wq::AbstractArray{T,3}, 
        wk::AbstractArray{T,3}, 
        wv::AbstractArray{T,3},
        wo::AbstractArray{T,3},
        rms_ffn_weight::AbstractArray{T,2},
        w1::AbstractArray{T,3},
        w2::AbstractArray{T,3},
        w3::AbstractArray{T,3},
        rms_final_weight::AbstractArray{T,1},
        wcls::AbstractArray{T,2}
        )

but it is recommended to use the data format defined by [Andrew Karpathy](https://github.com/karpathy/llama2.c)
to store both the configuration and weights of the transformer used and read the config and weights using 
[`read_checkpoint`](@ref).
"""
struct TransformerWeights{T<:AbstractFloat}
    # Note: In comparison to the dimensions in the original run.c from Karpathy,
    # the dimensions are transposed to increase performance by making use of the column-major order of Julia

    # token embedding table
    token_embedding_table::AbstractArray{T,2} # (dim, vocab_size)
    # weights for rmsnorms att
    rms_att_weight::AbstractArray{T,2} # (dim, layer) 
    # weights for matmuls.
    # Note:
    #   dim == n_heads * head_size
    #   kv_dim == n_kv_heads * head_size
    wq::AbstractArray{T,3} # (dim, dim, layer)
    wk::AbstractArray{T,3} # (dim, kv_dim, layer) 
    wv::AbstractArray{T,3} # (dim, kv_dim, layer)
    wo::AbstractArray{T,3} # (dim, dim, layer)
    # weights for rmsnorms ffn
    rms_ffn_weight::AbstractArray{T,2} # (dim, layer)
    # weights for ffn
    w1::AbstractArray{T,3} # (dim, hidden_dim, layer)
    w2::AbstractArray{T,3} # (hidden_dim, dim, layer)
    w3::AbstractArray{T,3} # (dim, hidden_dim, layer)
    # final rmsnorm
    rms_final_weight::AbstractArray{T,1} # (dim,)
    # (optinoal) classifier weights for the logits, on the last layer
    wcls::AbstractArray{T,2} # (dim, vocab_size)
    function TransformerWeights{T}(
        token_embedding_table::AbstractArray{T,2},
        rms_att_weight::AbstractArray{T,2},
        wq::AbstractArray{T,3},
        wk::AbstractArray{T,3},
        wv::AbstractArray{T,3},
        wo::AbstractArray{T,3},
        rms_ffn_weight::AbstractArray{T,2},
        w1::AbstractArray{T,3},
        w2::AbstractArray{T,3},
        w3::AbstractArray{T,3},
        rms_final_weight::AbstractArray{T,1},
        wcls::AbstractArray{T,2}
    ) where {T<:AbstractFloat}
        token_embedding_table_size = size(token_embedding_table)
        rms_att_weights_size = size(rms_att_weight)
        wq_size = size(wq)
        wk_size = size(wk)
        wv_size = size(wv)
        wo_size = size(wo)
        rms_ffn_weight_size = size(rms_ffn_weight)
        w1_size = size(w1)
        w2_size = size(w2)
        w3_size = size(w3)
        rms_final_weight_size = size(rms_final_weight)
        wcls_size = size(wcls)

        if !(token_embedding_table_size[1] == rms_att_weights_size[1]
             == wq_size[1] == wq_size[2] == wk_size[1] == wv_size[1]
             == wo_size[1] == rms_ffn_weight_size[1] == w1_size[1]
             == w2_size[2] == w3_size[1] == rms_final_weight_size[1]
             == wcls_size[1])
            throw(ArgumentError("dim does not match"))
        end
        if !(token_embedding_table_size[2] == wcls_size[2])
            throw(ArgumentError("vocab_size does not match"))
        end
        if !(rms_att_weights_size[2] == wq_size[3] == wk_size[3]
             == wv_size[3] == wo_size[3] == rms_ffn_weight_size[2]
             == w1_size[3] == w2_size[3] == w3_size[3])
            throw(ArgumentError("n_layers does not match"))
        end
        if !(w1_size[2] == w2_size[1] == w3_size[2])
            throw(ArgumentError("hidden_dim does not match"))
        end
        if !(wq_size[2] == wo_size[2])
            throw(ArgumentError("n_heads does not match"))
        end
        if !(wk_size[2] == wv_size[2])
            throw(ArgumentError("n_kv_heads does not match"))
        end
        return new(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls)
    end
end

"""
    memory_map_weights(config::Config, weights::Vector{T}, shared_weights::Int32)::TransformerWeights{T} where {T<:AbstractFloat}

Takes the values in `weights` and maps creates a new [`TransformerWeights`](@ref) struct using `config` and `shared_weights`.
"""
function memory_map_weights(
    config::Config,
    weights::Vector{T},
    shared_weights::Int32
)::TransformerWeights{T} where {T<:AbstractFloat}
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
    token_embedding_table = token_embedding_table
    rms_att_weight = split_weights[2]
    wq = split_weights[3]
    wk = split_weights[4]
    wv = split_weights[5]
    wo = split_weights[6]
    rms_ffn_weight = split_weights[7]
    w1 = split_weights[8]
    w2 = split_weights[9]
    w3 = split_weights[10]
    rms_final_weight = split_weights[11]
    wcls = wcls

    weights_type = eltype(weights)

    tfweights = TransformerWeights{weights_type}(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls)
    return tfweights
end

"""
    get_weights(weights::Vector, offset_1, offset_2, dims)::AbstractArray

Creates a view of `weights` from `offset_1` to `offset_2` with the dimension specified by `dims`.
"""
function get_weights(weights::Vector, offset_1, offset_2, dims)::AbstractArray
    w = @view weights[begin+offset_1:begin+offset_2-1]
    return reshape(w, dims)
end
