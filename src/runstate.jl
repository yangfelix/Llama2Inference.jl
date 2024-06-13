struct RunState
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
    key_cache::Array{float,3} # (layer, seq_len, dim)
    value_cache::Array{float,3} # (layer, seq_len, dim)
end