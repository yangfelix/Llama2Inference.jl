module Llama2Inference
using StringEncodings
using DataStructures
using Mmap: mmap
using Base.Iterators: partition
using LinearAlgebra: dot

include("Tokenizer.jl")
include("config.jl")
include("runstate.jl")
include("transformer_weights.jl")
include("transformer.jl")


# Write your package code here.


export Tokenizer
export read_checkpoint, Transformer, forward, rmsnorm!, softmax!, generate, test_forward
export Config, set_config_vocab_size, read_config
export TransformerWeights, get_weights, memory_map_weights
export encode,decode,find_token_str,find_token_id,sort_vocab!,build_tokenizer,Tokenizer,TokenIndex
export RunState

end
