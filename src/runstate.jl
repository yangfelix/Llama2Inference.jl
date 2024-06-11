struct RunState
    x::Array{float}
    xb::Array{float}
    xb2::Array{float}
    hb::Array{float}
    hb2::Array{float}
    q::Array{float}
    k::Array{float}
    v::Array{float}
    att::Array{float,2}
    logits::Array{float}
    key_cache::Array{float,3}
    value_cache::Array{float,3}
end