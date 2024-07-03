# Getting Started
This page shows some examples on how to use the `Llama2Inference` module. 

Our first example generates a deterministic story based on the input (`prompt`), while the second one uses a random factor, so the output story may differ each time you run it, despite using the same `prompt`.

## Setup
Open a new Pluto notebook, script or REPL session

* Script or REPL in package mode
```
activate --temp
add https://github.com/yangfelix/Llama2Inference.jl
```

* Or by creating a new environment in a [Pluto](https://plutojl.org/) notebook.
```
begin
    using Pkg
    Pkg.activate("Llama2Inference")
    Pkg.add(url="https://github.com/yangfelix/Llama2Inference.jl")
end
```

## Generating a random Story
```@repl
using Llama2Inference
config, weights = read_checkpoint("./bin/stories15M.bin");
transformer = Transformer(config, weights);
tokenizer = build_tokenizer("./bin/tokenizer.bin", Int(config.vocab_size));
sampler = Sampler(config.vocab_size, 0.5f0, 0.9f0);
generate(transformer, tokenizer, sampler, 256; prompt="The universe")
```

## Generating a deterministic Story
```@repl
using Llama2Inference
config, weights = read_checkpoint("./bin/stories15M.bin");
transformer = Transformer(config, weights);
tokenizer = build_tokenizer("./bin/tokenizer.bin", Int(config.vocab_size));
sampler = Sampler(config.vocab_size, 0.0f0, 0.9f0);
generate(transformer, tokenizer, sampler, 256; prompt="The universe")
```

!!! note
    The `temperature` and `topp` argument of the [`Sampler`](@ref) control the random factor of the sampled tokens at each timestep and therefore directly control the diversity generated stories with the same setup.