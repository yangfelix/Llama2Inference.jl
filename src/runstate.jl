mutable struct RunState
    x::Array{Float32} # activation at current time stamp (dim,)
    xb::Array{Float32} # same, but inside a residual branch (dim,)
    xb2::Array{Float32} # an additional buffer just for convenience (dim,)
    hb::Array{Float32} # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Array{Float32} # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Array{Float32} # query (dim,)
    k::Array{Float32} # key (dim,)
    v::Array{Float32} # value (dim,)
    att::Array{Float32,2} # buffer for scores/attention values (n_heads, seq_len)
    logits::Array{Float32} # output logits
    # kv cache
    # corresponding to line 86 in run.c the actual dimensionality of key-cache is (layer, seq, kv_dim)
    # s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    # in the calloc function kv_dim is used instead of dim like in the dimensionality describtion
    key_cache::Array{Float32,3} # (layer, seq_len, kv_dim)
    # same for value_cache in line 87
    value_cache::Array{Float32,3} # (layer, seq_len, kv_dim)
end

function RunState(config::Config)
    kv_dim = (config.dim * config.n_kv_heads) ÷ config.n_heads
    return RunState(
        zeros(Float32, config.dim),
        zeros(Float32, config.dim),
        zeros(Float32, config.dim),
        zeros(Float32, config.hidden_dim),
        zeros(Float32, config.hidden_dim),
        zeros(Float32, config.dim),
        zeros(Float32, config.dim),
        zeros(Float32, config.dim),
        zeros(Float32, (config.n_heads, config.seq_len)),
        zeros(Float32, config.vocab_size),
        zeros(Float32, (config.n_layers, config.seq_len, kv_dim)),
        zeros(Float32, (config.n_layers, config.seq_len, kv_dim))
    )
end