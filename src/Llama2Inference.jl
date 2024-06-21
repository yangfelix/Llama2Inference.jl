module Llama2Inference
using StringEncodings: encode
using DataStructures
using Mmap: mmap
using Base.Iterators: partition

include("Tokenizer.jl")
include("config.jl")
include("runstate.jl")
include("transformer_weights.jl")
include("transformer.jl")


# Write your package code here.


export Tokenizer
export read_checkpoint, Transformer, forward, rmsnorm!, softmax!, generate
export Config, set_config_vocab_size, read_config
export TransformerWeights, get_weights, memory_map_weights
export replace_top_pair!, Tokenizer, get_most_common_pair, count_consecutive, decoding, encoding
export RunState

end
