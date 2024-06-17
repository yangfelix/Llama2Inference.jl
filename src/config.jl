struct Config{T<:Integer}
    dim::T          # transformer dimension (most likely 4096 like in og LlaMa2)
    hidden_dim::T   # for ffn layers
    n_layers::T     # number of layers
    n_heads::T      # number of query heads
    n_kv_heads::T   # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::T   # vocabulary size, usually 256 (byte-level)
    seq_len::T      # max sequence length
end

function read_config(T::Type{<:Integer}, file::IO)
    num_bytes = sizeof(Config{T})
    buffer = Vector{UInt8}(undef, num_bytes)
    read!(file, buffer)
    config::Config{T} = reinterpret(Config{T}, buffer)[1]
    return config
end

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
