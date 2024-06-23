#include("transformer.jl")
using Random: rand

struct ProbIndex
    prob::Float32
    index::Int
end

#= function ProbIndex(prob::Float32, index::Int)
    return ProbIndex(prob, index)
end =#

mutable struct Sampler
    vocab_size::Int
    temperature::Float32
    topp::Float32
    probindex::Vector{ProbIndex} # only used with topp sampling
end

function Sampler(vocab_size::Int, temperature::Float32, topp::Float32)
    probindex = Vector{ProbIndex}(undef, vocab_size)
    return Sampler(vocab_size, temperature, topp, probindex)
end

"""
    sample_topp(sampler::Sampler, logits::Vector{Float32}, coin::Float32)::Int

top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed probability topp.
"""
function sample_topp(sampler::Sampler, logits::Vector{Float32}, coin::Float32)::Int
    # cutoff probability for the top-p sampling
    cutoff = (1.f0 - sampler.topp) / (sampler.vocab_size - 1.f0)
    # TODO: not needed I guess
    # pre-allocate an array of ProbIndex structs to store the probabilities and indices
    # sampler.probindex = Vector{ProbIndex}(undef, sampler.vocab_size)
    idx_good = 1
    @inbounds for idx in eachindex(logits)
        if logits[idx] >= cutoff
            sampler.probindex[idx_good] = ProbIndex(logits[idx], idx)
            idx_good += 1
        end
    end
    # use only the part of the probindex that is filled with values >= cutoff
    probindex_view = @view(sampler.probindex[1:idx_good-1])
    # sort the view in descending order by their probabilities
    sort!(probindex_view, by=(x -> x.prob), rev=true)

    # truncate the view to the smallest set of tokens where their cumulative sum of probabilities exceeds topp
    cumprob = 0.f0
    # in case of rounding errors, set the index to the last idx of this view
    cumprobidx = idx_good - 1
    @inbounds for idx in eachindex(probindex_view)
        cumprob += probindex_view[idx].prob
        if cumprob > sampler.topp
            cumprobidx = idx
            break
        end
    end

    # sample from the truncated view
    cumprob_view = @view(sampler.probindex[1:cumprobidx])
    r = coin * cumprob
    cumprob = 0.f0
    @inbounds for idx in eachindex(cumprob_view)
        cumprob += cumprob_view[idx].prob
        if r < cumprob
            return cumprob_view[idx].index
        end
    end
    # in case of rounding errors
    return cumprob_view[end].index
end

function sample(sampler::Sampler, logits::Vector{Float32})::Int
    sampler.vocab_size == length(logits) || throw(ArgumentError("logits length does not match vocab_size"))
    if sampler.temperature == 0.0f0
        return argmax(logits)
    else
        # apply the temperature to the logits
        logits ./= sampler.temperature
        # apply softmax to the logits to get the probabilities for next token
        softmax!(logits)
        # flip a (float) coin (this is our source of entropy for sampling)
        coin = rand(Float32)
        if sampler.topp <= 0.f0 || sampler.topp >= 1.f0
            # sample from the full distribution
            cumsum = 0.f0
            for idx in 1:sampler.vocab_size
                cumsum += logits[idx]
                if cumsum >= coin
                    return idx
                end
            end
            # in case of rounding errors
            return sampler.vocab_size
        else
            # top-p (nucleus) sampling, clamping least likely tokens to 0
            return sample_topp(sampler, logits, coin)
        end
    end
end