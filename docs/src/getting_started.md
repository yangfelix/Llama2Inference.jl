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

!!! note
    You can download other weights for more complex models pre-trained by Karpathy [here](https://huggingface.co/karpathy/tinyllamas/tree/main).
    These are not included in this repository to keep the required space lower.

    The above examples work by only changing the path in `read_checkpoint(path)` to the desired weights file.

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
    The `temperature` and `topp` argument of the [`Sampler`](@ref) control the random factor of the sampled tokens at each timestep and therefore directly control the diversity of the generated stories.

## Tokenizer only
```@repl
using Llama2Inference
config, _ = read_checkpoint("./bin/stories15M.bin");
tokenizer = build_tokenizer("./bin/tokenizer.bin", Int(config.vocab_size));
BOS::Bool = true;
EOS::Bool = false;
prompt1 = "Example prompt";
encoded_prompt = encode(tokenizer, prompt1, BOS, EOS);
println("The encoded prompt is: $encoded_prompt");
token = encoded_prompt[1];
for next in encoded_prompt[2:end]
    piece = decode(tokenizer, token, next)
    token = next
    safe_print(piece)
end
```

## Sampler only
```@repl
using Llama2Inference
temperature = 0.0;
topp = 0.0;
sampler = Sampler(6, temperature, topp);
logits = [0.1f0, 0.3f0, 0.2f0, 0.15f0, 0.15f0, 0.1f0];
index = sample(sampler, logits)
```
!!! note
    Check the influence of the `temperature` and `topp` parameters on the choice of sampling algorithm in the documentation of the [`sample`](@ref) function.