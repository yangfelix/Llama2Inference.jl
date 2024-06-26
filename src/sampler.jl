using Random: rand

struct ProbIndex
    prob::Float32
    index::Int32
end

mutable struct Sampler
    vocab_size::Int32
    temperature::Float32
    topp::Float32
    probindex::Vector{ProbIndex} # only used with topp sampling

    function Sampler(vocab_size::Int32, temperature::Float32, topp::Float32)
        probindex = Vector{ProbIndex}(undef, vocab_size)
        new(vocab_size, temperature, topp, probindex)
    end
end

"""
    Sampler(vocab_size::Int, temperature::Float32, topp::Float32)

A struct to hold the parameters for sampling from a distribution.

# Arguments
- `vocab_size::Int`: The size of the vocabulary.
- `temperature::Float32`: The temperature to apply to the logits before sampling.
- `topp::Float32`: The probability mass to sample from the top-p distribution.
"""
function Sampler(vocab_size::Int, temperature::Float32, topp::Float32)
    vocab_size = Int32(vocab_size)
    return Sampler(vocab_size, temperature, topp)
end

"""
    sample_topp(sampler::Sampler, logits::Vector{Float32}, coin::Float32)::Int

top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed probability topp.
"""
function sample_topp(sampler::Sampler, logits::Vector{Float32}, coin::Float32)::Int
    # cutoff probability for the top-p sampling
    cutoff = (1.f0 - sampler.topp) / (sampler.vocab_size - 1.f0)
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

"""
    sample(sampler::Sampler, logits::Vector{Float32})

Sample from the `logits` using the `sampler`.

# Example
```julia-repl
julia> sampler = Sampler(6, 0.0f0, 0.0f0)
julia> logits = [0.1f0, 0.3f0, 0.2f0, 0.15f0, 0.15f0, 0.1f0]
julia> sample(sampler, logits)
2
```
"""
function sample(sampler::Sampler, logits::Vector{Float32})
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
            # top-p (nucleus) sampling
            return sample_topp(sampler, logits, coin)
        end
    end
end