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