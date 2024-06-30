# Getting Started
This page shows some examples on how to use the `Llama2Inference` module. 

Our first example generates a random story, while the second one uses a random factor, so the output may differ each time you run it.

## Prerequisites
Before running the examples, you need to download the necessary weights. In this example, we use the weights for a small 15M model trained by Andrej Karpathy. Follow these steps:

* Navigate to the directory where you want to download the weights.
* Run the following command:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```
* Additionally, you will need to download the `tokenizer.bin` file from the GitHub repository.

## Setup
* Navigate to the directory where downloaded the weights.
```
julia
using Pkg
Pkg.activate("Llama2Inference")
Pkg.add(url="https://github.com/yangfelix/Llama2Inference.jl")

```

## Generating a random Story
```@repl
using Llama2Inference
config, weights = read_checkpoint("./stories15M.bin");
transformer = Transformer(config, weights);
tokenizer = build_tokenizer("./tokenizer.bin", Int(config.vocab_size));
sampler = Sampler(config.vocab_size, 0.5f0, 0.9f0);
generate(transformer, tokenizer, sampler, 256; prompt="The universe")
```

## Generating a deterministic Story
```@repl
using Llama2Inference
config, weights = read_checkpoint("./stories15M.bin");
transformer = Transformer(config, weights);
tokenizer = build_tokenizer("./tokenizer.bin", Int(config.vocab_size));
sampler = Sampler(config.vocab_size, 0.0f0, 0.9f0);
generate(transformer, tokenizer, sampler, 256; prompt="The universe")
```

!!! note
    The `temperature` and `topp` argument of the [`Sampler`](@ref) control the random factor of the sampled tokens at each timestep and therefore directly control the diversity generated stories with the same setup.