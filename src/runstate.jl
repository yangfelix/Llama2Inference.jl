mutable struct RunState{T<:AbstractFloat}
    x::Array{T} # activation at current time stamp (dim,)
    xb::Array{T} # same, but inside a residual branch (dim,)
    xb2::Array{T} # an additional buffer just for convenience (dim,)
    hb::Array{T} # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Array{T} # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Array{T} # query (dim,)
    k::Array{T} # key (dim,)
    v::Array{T} # value (dim,)
    att::Array{T,2} # buffer for scores/attention values (n_heads, seq_len)
    logits::Array{T} # output logits
    # kv cache
    # corresponding to line 86 in run.c the actual dimensionality of key-cache is (layer, seq, kv_dim)
    # s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    # in the calloc function kv_dim is used instead of dim like in the dimensionality describtion
    key_cache::Array{T,3} # (kv_dim, seq_len, layer)
    # same for value_cache in line 87
    value_cache::Array{T,3} # (kv_dim, seq_len, layer)
end

"""
    RunState(config::Config; T::Type{<:AbstractFloat} = Float32)

Create a RunState object with fields of type T (Float32 by default) from the given configuration.

This is used to store the state of the model during inference and is 0-initialized.
"""
function RunState(config::Config; T::Type{<:AbstractFloat} = Float32)
    kv_dim = (config.dim * config.n_kv_heads) ÷ config.n_heads
    return RunState(
        zeros(T, config.dim),
        zeros(T, config.dim),
        zeros(T, config.dim),
        zeros(T, config.hidden_dim),
        zeros(T, config.hidden_dim),
        zeros(T, config.dim),
        zeros(T, config.dim),
        zeros(T, config.dim),
        zeros(T, (config.n_heads, config.seq_len)),
        zeros(T, config.vocab_size),
        zeros(T, (kv_dim, config.seq_len, config.n_layers)),
        zeros(T, (kv_dim, config.seq_len, config.n_layers))
    )
end