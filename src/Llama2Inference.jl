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
export read_checkpoint, TransformerWeights
export replace_top_pair!, Tokenizer, get_most_common_pair, count_consecutive, decoding, encoding

end
