# Getting Started
This page shows some examples on how to use the `Llama2Inference` module.

## Generating a random Story
```@repl
using Llama2Inference
config, weights = read_checkpoint("./stories15M.bin");
transformer = Transformer(config, weights);
tokenizer = build_tokenizer("./tokenizer.bin", Int(config.vocab_size));
sampler = Sampler(config.vocab_size, 0.5f0, 0.9f0);
generate(transformer, tokenizer, sampler, 1024; prompt="The universe")
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
    The `temperature` and `topp` argument of the [`Sampler`](@ref) control the random factor of the sampled tokens at each timestep and therefore directly control the diversity of the generated stories.