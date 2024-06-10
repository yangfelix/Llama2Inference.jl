module Llama2Inference
using StringEncodings: encode
using DataStructures
export Tokenizer
using Base.Iterators: partition


export replace_top_pair!,Tokenizer,get_most_common_pair,count_consecutive,decoding,encoding
include("Tokenizer.jl")

end
