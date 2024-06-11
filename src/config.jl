struct Config{T<:Integer}
    dim::T
    hidden_dim::T
    n_layers::T
    n_heads::T
    n_kv_heads::T
    vocab_size::T
    seq_len::T
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
