struct Transformer
    config::Config
    weights::TransformerWeights
    state::RunState
    fd::Int
    data::Array{float}
    file_size::Int
end

function read_checkpoint(checkpoint::String)::Tuple{Config,TransformerWeights}#, config::Config, weights::TransformerWeights, fd::Int, data::Array{float}, file_size::Int)
    filesize = stat(checkpoint).size
    config, weights = open(checkpoint, "r") do file
        config = read_config(Int32, file)
        weights_size = div(filesize - sizeof(config), sizeof(Float32))
        weights = mmap(file, Vector{Float32}, weights_size)
        return config, weights
    end
    shared_weights::Int32 = config.vocab_size > 0 ? 1 : 0
    config = set_config_vocab_size(config, abs(config.vocab_size))
    transformer_weights = memory_map_weights(config, weights, shared_weights)
    return config, transformer_weights
end

"""
    rmsnorm!(out::Array{T, 1}, x::Array{T,1}, weight::Array{T,1}) where T<:AbstractFloat

normalize `out` in place by the root mean square of `x` and multiply by the learned weights `weight`.
"""
function rmsnorm!(out::AbstractArray{T, 1}, x::AbstractArray{T,1}, weight::AbstractArray{T,1}) where T<:AbstractFloat
    (size(out) == size(x) == size(weight)) || throw(DimensionMismatch("size(out) != size(x) != size(weight), $(size(out)) != $(size(x)) != $(size(weight))."))
    # calculate 1 / (the root mean square of the input)
    rms = 1.0f0 / sqrt( (sum(x.^2) / length(x)) + 1e-5) # add 1e-8 for numerical stability
    # multiply by the learned weight and normalize by 
    @. out = weight * x * rms
end

"""
    softmax!(x::T{T2,1}) where {T<:AbstractArray, T2<:AbstractFloat}

softmax the values in `x` in place, up to position `pos` inclusively.
"""
function softmax!(x::AbstractArray{T2,1}) where T2<:AbstractFloat
    # subtract the maximum value for numerical stability
    x .-= maximum(x)
    # exponentiate the values
    x .= exp.(x)
    # normalize the values
    x ./= sum(x)
end

# used for debugging
function test_forward(token::Int; pos::Int=1)
    # in run.c token = 1, pos = 0
    # to reach same logits as in run.c, we need to set pos = 1
    config, weights = read_checkpoint("stories15M.bin")
    state = RunState(config)
    logits = forward(config, weights, state, token, pos)
    return logits
end

"""
    forward(transformer::Transformer, token::Int, pos::Int)

forward the transformer model with the input `token` at position `pos`.

LlaMa2 was used as the architecure of the transformer model and it's modifications.
The forward pass looks like the following:
  1) forward through all layers
    a) RMSNorm
    b) linear project x into Query, Key, Value with Wq, Wk, Wv
    c) RoPE relative positional encoding
    d) multihead attention
    e) residual of x + RMSNorm
    f) MLP with SwiGLU non-linearity
  2) rmsnrom
  3) classify into logits
"""
#function forward(transformer::Transformer, token::Int, pos::Int)
    #= config = transformer.config
    weights = transformer.weights
    state = transformer.state =#
function forward(config::Config, weights::TransformerWeights, state::RunState, token::Int, pos::Int)
    # some convenience variables
    dim = config.dim
    # integer multiplier of the kv sharing in (multiquery ?) GQA was used in LlaMa2 ...
    # kv_mul = config.n_heads / config.n_kv_heads
    group_size = config.n_heads ÷ config.n_kv_heads
    hidden_dim = config.hidden_dim
    head_size = dim ÷ config.n_heads
    kv_dim =  head_size * config.n_kv_heads

    # copy token embedding into x, TODO probably it is at position token+1 because of Julia indexing, then the values are the same
    state.x = weights.token_embedding_table[token+1,:] # (dim,)

    # 1) forward through all layers
    for layer in 1:config.n_layers
        # a) attention RMSNorm
        state.xb = rmsnorm!(state.xb, state.x, weights.rms_att_weight[layer,:])

        # b) linear projection to Q,K,V
        state.q = @view(weights.wq[layer,:,:]) * state.xb # (dim, dim) * (dim,) = (dim,)
        state.key_cache[layer, pos, :] = @views weights.wk[layer,:,:] * state.xb # (kv_dim, dim) * (dim,) = (kv_dim,)
        state.value_cache[layer, pos, :] = @views weights.wv[layer,:,:] * state.xb # (kv_dim, dim) * (dim,) = (kv_dim,)

        # c) RoPE relative positional encoding: complex-valued rotate q and k in each head
        for i in range(1, dim, step=2)
            head_dim = (i-1) % head_size
            freq = 1.0f0 / (100000.0f0^(head_dim/head_size))
            val = (pos-1) * freq # in our code pos is 1-based because of Julia indexing, here we need to subtract 1 to have correct calculations
            fcr = cos(val)
            fci = sin(val)

            v0 = state.q[i]
            v1 = state.q[i+1]
            state.q[i] = v0 * fcr - v1 * fci
            state.q[i+1] = v0 * fci + v1 * fcr


            if i <= kv_dim
                v0 = state.key_cache[layer, pos, i]
                v1 = state.key_cache[layer, pos, i+1]
                state.key_cache[layer, pos, i] = v0 * fcr - v1 * fci
                state.key_cache[layer, pos, i+1] = v0 * fci + v1 * fcr
            end
        end

        # d) multihead attention       
        for h in 0:config.n_heads-1 # iterate over all heads

            # get part of the query vector for this head
            h_offset = h*head_size
            q = @view(state.q[h_offset+1 : h_offset + head_size]) # +1 for Julia indexing, (head_size,)
            
            # attention vector for this head
            att = @view(state.att[h+1, :]) # (seq_len,) +1 for Julia indexing

            # Integer division for assoiciated group number of head,
            # each head belongs to a certain kv-group, where they share the same wk, wv -> key/value (GQA)
            group_number = h ÷ group_size
            kv_offset = group_number * head_size

            # iterate over all timestamps including the current one
            for t in 1:pos
                # get the key vector for this head and at this timestep
                k = @view(state.key_cache[layer, t, kv_offset+1 : kv_offset + head_size]) # +1 for Julia indexing, (head_size,)
                # update attention in place to the calculated 'similarity' score
                att[t] = dot(q,k) / sqrt(Float32(head_size))
            end

            # softmax the scores to get attention weights, from 0..pos inclusively
            softmax!(@view(att[begin:pos]))

            # weighted sum of the values, store back into xb
            for t in 1:pos
                # get the value vector for this head and at this timestep
                v = @view(state.value_cache[layer, t, kv_offset+1 : kv_offset + head_size]) # (head_size,)
                @. state.xb[h_offset+1 : h_offset + head_size] = v * att[t]
            end
        end # end of head loop

        # matmul with wo and xb = attention = wo * (value * att_score)
        state.xb2 = weights.wo[layer,:,:] * state.xb

        # e) residual connection back into x + RMSNorm
        state.x += state.xb2
        rmsnorm!(state.xb, state.x, weights.rms_ffn_weight[layer,:])

        # f) MLP with SwiGLU non-linearity
        # self.w2(F.silu(self.w1(x)) * self.w3(x))
        state.hb = @view(weights.w1[layer,:,:]) * state.xb # (hidden_dim, dim) * (dim,) = (hidden_dim,)
        state.hb2 = @view(weights.w3[layer,:,:]) * state.xb # (hidden_dim, dim) * (dim,) = (hidden_dim,)

        # SwiGLU non-linearity
        # @. macro prepends . to all operations,
        @. state.hb *= (1.0f0 / (1.0f0 + exp(-state.hb))) * state.hb2

        # final matul to get output of MLP
        state.xb = @view(weights.w2[layer,:,:]) * state.hb # (dim, hidden_dim) * (hidden_dim,) = (dim,)

        # residual connection
        state.x += state.xb
    end # end of layer loop

    # 2) RMSNorm
    rmsnorm!(state.x, state.x, weights.rms_final_weight)

    # 3) classify into logits
    state.logits = weights.wcls * state.x # (vocab_size, dim) * (dim,) = (vocab_size,)
    return state.logits
end

function generate(transformer::Transformer, tokenizer::Tokenizer, sampler, steps::Int; prompt::String="")
    # start with the input text in prompt
    prompt_tokens = encoding(prompt, tokenizer.vocab) # return Vector{Int} containing the ids (tokens?)
    num_prompt_tokens = length(prompt_tokens)
    if num_prompt_tokens < 1
        throw(error("length of prompt_tokens is $(num_prompt_tokens)!"))
    end

    # start the main loop
    next = nothing
    token = prompt_tokens[1]
    pos = 1 # Julia is 1 vs. C is 0

    while pos < steps
        # forward the transformer to get logits for the next token
        logits = forward(transformer, token, pos)

        # advance the state machine
        if (pos < num_prompt_tokens)
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1]
        else
            # otherwise sample the next token from the logits
            next = sample(sampler, logits)
        end
        pos += 1

        # data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1)
            break
        end

        # print the token as string, decode it with the Tokenizer object
        piece = decoding(token, tokenizer.vocab)
        print(piece) # same as printf("%s", piece), but skips "unsafe" bytes
        token = next
    end

    println("")
end