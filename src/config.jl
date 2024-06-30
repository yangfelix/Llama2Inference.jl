"""
    Config{T<:Integer}

Struct to hold the configuration of the [`Transformer`](@ref) used for inference. It contains the following fields:

- dim::T: transformer dimension
- hidden_dim::T: for ffn layers
- n_layers::T: number of layers
- n_heads::T: number of query heads
- n_kv_heads::T: number of key/value heads
- vocab_size::T: vocabulary size
- seq_len::T: max sequence length

It is possible to construct a `Config` using the constructor 

    Config{T}(dim::T, hidden_dim::T, n_layers::T, n_heads::T, n_kv_heads::T, vocab_size::T, seq_len::T)

but it is recommended to use the data format defined by [Andrew Karpathy](https://github.com/karpathy/llama2.c)
to store both the configuration and weights of the transformer used and read the config and weights using 
[`read_checkpoint`](@ref).
"""
struct Config{T<:Integer}
    dim::T          # transformer dimension
    hidden_dim::T   # for ffn layers
    n_layers::T     # number of layers
    n_heads::T      # number of query heads
    n_kv_heads::T   # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::T   # vocabulary size
    seq_len::T      # max sequence length
end

"""
    read_config(T::Type{<:Integer}, file::IO)::Config{T}

Reads from `file` to create a [`Config{T}`](@ref).
"""
function read_config(T::Type{<:Integer}, file::IO)::Config{T}
    num_bytes = sizeof(Config{T})
    buffer = Vector{UInt8}(undef, num_bytes)
    read!(file, buffer)
    config::Config{T} = reinterpret(Config{T}, buffer)[1]
    return config
end

"""
    set_config_vocab_size(config::Config{T}, vocab_size::T)

Creates a new [`Config{T}`](@ref) with the field `vocab_size` updated to `vocab_size`.
"""
function set_config_vocab_size(config::Config{T}, vocab_size::T) where {T<:Integer}
    return Config(
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        vocab_size,
        config.seq_len
    )
end
