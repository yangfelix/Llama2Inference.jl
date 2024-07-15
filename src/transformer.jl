"""
Transformer struct which contains the fields
- config::[`Config`](@ref)
- weights::[`TransformerWeights`](@ref)
"""
struct Transformer
    config::Config
    weights::TransformerWeights
end

"""
    read_checkpoint(checkpoint::String, T_weights::Type=Float32, T_config::Type=Int32)::Tuple{Config{T_config},TransformerWeights{T_weights}}

Reads the config and weights from the file at `checkpoint`.
"""
function read_checkpoint(checkpoint::String, T_weights::Type=Float32, T_config::Type=Int32)::Tuple{Config{T_config},TransformerWeights{T_weights}}
    filesize = stat(checkpoint).size
    config, weights = open(checkpoint, "r") do file
        config = read_config(T_config, file)
        weights_size = div(filesize - sizeof(config), sizeof(T_weights))
        weights = mmap(file, Vector{T_weights}, weights_size)
        return config, weights
    end
    shared_weights::Int32 = config.vocab_size > 0 ? 1 : 0
    config = set_config_vocab_size(config, abs(config.vocab_size))
    transformer_weights = memory_map_weights(config, weights, shared_weights)

    return config, transformer_weights
end

"""
    safe_print(piece::String)

Prints the string `piece` to the console, skipping any "unsafe" bytes.
"""
function safe_print(piece::String)
    if piece[1] == '\0'
        return
    end
    if length(piece) > 1
        byte_val::Char = piece[1]
        if !(isprint(byte_val) || isspace(byte_val))
            return
        end
    end
    print(piece)
end

"""
    rmsnorm!(out::AbstractArray{Float32, 1}, x::AbstractArray{Float32,1}, weight::AbstractArray{Float32,1})

Normalize `out` in place by the root mean square of `x` and multiply with `weight`.

```math
out_i = \\frac{x_i}{RMS(x)} * weight_i \\quad\\text{,where}\\quad RMS(x)= \\sqrt{ (\\frac{1}{n} * \\sum_{i=1}^{n} x_i^2) +1\\mathrm{e}{-5}} 
```
1e-5 is added for numerical stability in the square root part.
"""
function rmsnorm!(out::AbstractArray{T,1}, x::AbstractArray{T,1}, weight::AbstractArray{T,1}) where {T<:AbstractFloat}
    (size(out) == size(x) == size(weight)) || throw(DimensionMismatch("size(out) != size(x) != size(weight), $(size(out)) != $(size(x)) != $(size(weight))."))
    # calculate 1 / (the root mean square of the input)
    rms = one(T) / sqrt((dot(x, x) / length(x)) + T(1e-5)) # add 1e-5 for numerical stability
    # multiply by the learned weight and normalize by 
    @. out = weight * x * rms
end

"""
    softmax!(x::AbstractArray{Float32,1})

Softmax the values in `x` in place.

```math
x_i = \\frac{e^{x_i}}{\\sum_{j=1}^{n} e^{x_j}}
```
"""
function softmax!(x::AbstractArray{T,1}) where {T<:AbstractFloat}
    # subtract the maximum value for numerical stability
    @. x -= $maximum(x)
    # exponentiate the values
    @. x = exp(x)
    # normalize the values
    @. x /= $sum(x)
end

"""
    mat_T_vec!(out::AbstractArray{T,1}, x::AbstractArray{T,1}, w::AbstractArray{T,2}) where T<:AbstractFloat

Efficient transpose(matrix)-vector multiplication `out = w' * x`.
"""
function mat_T_vec!(out::AbstractArray{T,1}, x::AbstractArray{T,1}, w::AbstractArray{T,2}) where {T<:AbstractFloat}
    # out = w' * x
    # with w (n,d) x (n,) out (d,)
    @inbounds for (i, col) in enumerate(eachcol(w))
        out[i] = dot(col, x)
    end
end

"""
    forward!(transformer::Transformer, state::RunState, token::Int, pos::Int)

Forward the `token` at position `pos` through the transformer model.

The foward pass corresponds to the LlaMa2 decoder architecture and is based on the C implementation by [Andrej Karpathy](https://github.com/karpathy/llama2.c/blob/master/run.c).
The output is a logits vector of size `transformer.config.vocab_size`.

# Arguments
- `transformer::Transformer`: The transformer object containg config, weights and the token embedding table.
- `state::RunState`: The state object to store the intermediate results during the forward pass.
- `token::Int`: The token to forward through the transformer.
- `pos::Int`: The position of the token in the sequence. The position is 1-based, which means the first position in a sequence is 1.

# Example
```julia-repl
julia> config, weights = read_checkpoint("./bin/stories15M.bin")
julia> transformer = Transformer(config, weights)
julia> tokenizer = build_tokenizer("./bin/tokenizer.bin", Int(config.vocab_size))
julia> state = RunState(config)
julia> token = 2
julia> pos = 1
julia> forward!(transformer, state, token, pos)
julia> state.logits
32000-element Vector{Float32}:
 -6.7907834
  0.82811606
 -6.7904234
 -6.790472
  ⋮
 -6.79068
 -6.790696
 -6.7906737
 -6.7905493
```
"""
function forward!(transformer::Transformer, state::RunState, token::Int, pos::Int)
    # some convenience variables
    config = transformer.config
    weights = transformer.weights
    dim = config.dim
    # integer multiplier of the kv sharing in (multiquery ?) according to run.c by Andrej Karpathy
    group_size = config.n_heads ÷ config.n_kv_heads
    head_size = dim ÷ config.n_heads
    kv_dim = head_size * config.n_kv_heads

    # copy token embedding into x
    state.x = weights.token_embedding_table[:, token] # (dim,)

    # 1) forward through all layers
    @inbounds for layer in 1:config.n_layers
        # a) attention RMSNorm
        @views rmsnorm!(state.xb, state.x, weights.rms_att_weight[:, layer])

        # b) linear projection to Q,K,V
        @views mat_T_vec!(state.q, state.xb, weights.wq[:, :, layer])  # wq (dim, dim) xb (dim,) -> q = wq' * xb
        @views mat_T_vec!(state.key_cache[:, pos, layer], state.xb, weights.wk[:, :, layer]) # wk (dim, kv_dim) xb (dim,) -> key_cache (kv_dim, )
        @views mat_T_vec!(state.value_cache[:, pos, layer], state.xb, weights.wv[:, :, layer]) # wv (dim, kv_dim) xb (dim,) -> value_cache (kv_dim, )

        # c) RoPE relative positional encoding: complex-valued rotate q and k in each head
        @inbounds for i in range(1, dim, step=2)
            head_dim = (i - 1) % head_size
            freq = 1.0f0 / (10000.0f0^(head_dim / head_size))
            val = (pos - 1) * freq # in our code pos is 1-based because of Julia indexing, here we need to subtract 1 to have correct calculations
            fcr = cos(val)
            fci = sin(val)

            v0 = state.q[i]
            v1 = state.q[i+1]
            state.q[i] = v0 * fcr - v1 * fci
            state.q[i+1] = v0 * fci + v1 * fcr

            if i <= kv_dim
                v0 = state.key_cache[i, pos, layer]
                v1 = state.key_cache[i+1, pos, layer]
                state.key_cache[i, pos, layer] = v0 * fcr - v1 * fci
                state.key_cache[i+1, pos, layer] = v0 * fci + v1 * fcr
            end
        end

        # d) multihead attention
        @inbounds for h in 0:config.n_heads-1 # iterate over all heads
            # get part of the query vector for this head
            h_offset = h * head_size
            q = @view(state.q[h_offset+1:h_offset+head_size]) # +1 for Julia indexing, (head_size,)

            # attention vector for this head
            att = @view(state.att[h+1, :]) # (seq_len,) +1 for Julia indexing

            # Integer division for assoiciated group number of head,
            # each head belongs to a certain kv-group, where they share the same wk, wv -> key/value (GQA)
            group_number = h ÷ group_size
            kv_offset = group_number * head_size

            # iterate over all timestamps including the current one
            @inbounds for t in 1:pos
                # get the key vector for this head and at this timestep
                k = @view(state.key_cache[kv_offset+1:kv_offset+head_size, t, layer]) # +1 for Julia indexing, (head_size,)
                # update attention in place to the calculated 'similarity' score
                att[t] = dot(q, k) / sqrt(Float32(head_size))
            end

            # softmax the scores to get attention weights, from 0..pos inclusively
            softmax!(@view(att[begin:pos]))

            # weighted sum of the values, store back into xb
            state.xb[h_offset+1:h_offset+head_size] .= 0.0f0
            @inbounds for t in 1:pos
                # get the value vector for this head and at this timestep
                v = @view(state.value_cache[kv_offset+1:kv_offset+head_size, t, layer]) # (head_size,)
                @. state.xb[h_offset+1:h_offset+head_size] += v * att[t]
            end
        end # end of head loop

        # matmul with wo and xb = attention = wo * (value * att_score)
        @views mat_T_vec!(state.xb2, state.xb, weights.wo[:, :, layer]) # wo (dim, dim) xb (dim,) -> xb2 = wo' * xb

        # e) residual connection back into x + RMSNorm
        state.x += state.xb2
        @views rmsnorm!(state.xb, state.x, weights.rms_ffn_weight[:, layer])

        # f) MLP with SwiGLU non-linearity
        # self.w2(F.silu(self.w1(x)) * self.w3(x))
        @views mat_T_vec!(state.hb, state.xb, weights.w1[:, :, layer]) # w1 (dim, hidden_dim) xb (dim,) -> hb = w1' * xb
        @views mat_T_vec!(state.hb2, state.xb, weights.w3[:, :, layer]) # w3 (dim, hidden_dim) xb (dim,) -> hb2 = w3' * xb

        # SwiGLU non-linearity
        @. state.hb *= (1.0f0 / (1.0f0 + exp(-state.hb))) * state.hb2

        # final matul to get output of MLP
        @views mat_T_vec!(state.xb, state.hb, weights.w2[:, :, layer]) # w2 (hidden_dim, dim) hb (dim) -> xb = w2' * hb

        # residual connection
        state.x += state.xb
    end # end of layer loop

    # 2) RMSNorm
    rmsnorm!(state.x, state.x, weights.rms_final_weight)

    # 3) classify into logits
    mat_T_vec!(state.logits, state.x, weights.wcls) # wcls (dim, vocab_size) x (dim,) -> logits = wcls' * x
end

"""
    generate(transformer::Transformer, tokenizer::Tokenizer, sampler::Sampler, steps::Int; prompt::String="", performance=true)

Generate a sequence of tokens using the `transformer`.

# Arguments
- `transformer::Transformer`: The transformer object containg config and weights.
- `tokenizer::Tokenizer`: The tokenizer object to encode and decode tokens.
- `sampler::Sampler`: The sampler object to sample a token from the output logits.
- `steps::Int`: The number of maximum tokens to generate, upper bound by `transformer.config.seq_len`.
- `prompt::String`: The input text to start the generation. If none, the generation starts with an empty string.
- `performance::Bool`: If true, print the number of generated tokens and number of tokens generated per second.

# Example
```julia-repl
julia> config, weights = read_checkpoint("./bin/stories15M.bin")
julia> transformer = Transformer(config, weights)
julia> tokenizer = build_tokenizer("./bin/tokenizer.bin", Int(config.vocab_size))
julia> sampler = Sampler(config.vocab_size, 0.0f0, 0.9f0)
julia> generate(transformer, tokenizer, sampler, 23; prompt="The universe", performance=true)
The universe was bright and full of stars. Every night, the stars would twinkle and shine.
```
"""
function generate(transformer::Transformer, tokenizer::Tokenizer, sampler::Sampler, steps::Int; prompt::String="", performance=true)
    # check if the number of steps is valid
    if steps < 1 || steps > transformer.config.seq_len
        steps = transformer.config.seq_len
    end

    # start with the input text in prompt
    prompt_tokens = encode(tokenizer, prompt, true,false) # return Vector{Int} containing the ids (tokens?)
    num_prompt_tokens = length(prompt_tokens)
    if num_prompt_tokens < 1
        throw(error("length of prompt_tokens is $(num_prompt_tokens)!"))
    end

    # initiate the state
    state = RunState(transformer.config)

    # start the main loop
    start = 0.0
    next = nothing
    token = prompt_tokens[1]
    pos = 1 # Julia is 1 vs. C is 0

    while pos < steps
        # forward the transformer to get logits for the next token
        forward!(transformer, state, token, pos)

        # advance the state machine
        if (pos < num_prompt_tokens)
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos+1]
        else
            # otherwise sample the next token from the logits
            next = sample(sampler, state.logits)
        end
        pos += 1

        # data-dependent terminating condition: the BOS (=2) token delimits sequences
        if (next == 2)
            break
        end

        # print the token as string, decode it with the Tokenizer object
        piece = decode(tokenizer, token, next)
        #print("Type of Token is ", typeof(piece), " length of token = ", length(piece), " ")
        safe_print(piece) # same as printf("%s", piece), but skips "unsafe" bytes
        #print("\n")
        token = next

        if start == 0.0
            start = time()
        end
    end

    println("")

    if performance && pos > 2
        elapsed = time() - start
        println("Generated $(pos-1) tokens in $(elapsed) seconds, $(pos/elapsed) tokens per second.")
    end
end