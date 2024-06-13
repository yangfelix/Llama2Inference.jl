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

function forward(transformer::Transformer, token::Int64, pos::Int64)
    # some convenience variables
    config = transformer.config
    weights = transformer.weights
    state = transformer.state
    # line 72 overwrites x before using it
    # x = state.x
    dim = config.dim
    kv_dim = (config.dim * config.n_kv_heads) / config.n_heads
    kv_mul = config.n_heads / config.n_kv_heads # integer multiplier of the kv sharing in multiquery
    hidden_dim = config.hidden_dim
    head_size = dim / config.n_heads

    # copy token embedding into x
    x = weights.token_embedding_table + token * dim

    # forward all layer
    # line 249
    for nothing in nothing
        
    end
end

function generate(transformer::Transformer, tokenizer::Tokenizer, sampler, steps:Integer; prompt::String="")
    # start with the input text in prompt
    prompt_tokens = encoding(prompt, tokenizer.vocab) # return Vector{Int64} containing the ids (tokens?)
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