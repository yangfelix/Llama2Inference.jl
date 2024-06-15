# 
mutable struct RunState
    x::Array{float} # activation at current time stamp (dim,)
    xb::Array{float} # same, but inside a residual branch (dim,)
    xb2::Array{float} # an additional buffer just for convenience (dim,)
    hb::Array{float} # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Array{float} # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Array{float} # query (dim,)
    k::Array{float} # key (dim,)
    v::Array{float} # value (dim,)
    att::Array{float,2} # buffer for scores/attention values (n_heads, seq_len)
    logits::Array{float} # output logits
    # kv cache
    # corresponding to line 86 in run.c the actual dimensionality of key-cache is (layer, seq, kv_dim)
    # s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    # in the calloc function kv_dim is used instead of dim like in the dimensionality describtion
    key_cache::Array{float,3} # (layer, seq_len, kv_dim)
    # same for value_cache in line 87
    value_cache::Array{float,3} # (layer, seq_len, kv_dim)
end