# Llama2Inference

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yangfelix.github.io/Llama2Inference.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yangfelix.github.io/Llama2Inference.jl/dev/)
[![Build Status](https://github.com/yangfelix/Llama2Inference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/yangfelix/Llama2Inference.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/yangfelix/Llama2Inference.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yangfelix/Llama2Inference.jl)

This repository holds a Julia module called `Llama2Inference` and was created as part of course work at [TU Berlin](https://www.tu.berlin/).

The module enables the user to generate text by inferencing the large language model [Llama2](https://llama.meta.com/llama2/). The implementation is based on a [C implementation](https://github.com/karpathy/llama2.c/blob/master/run.c) of the inference pass for Llama2 by [Andrej Karpathy](https://github.com/karpathy).

## Architecture of Llama2 Decoder-Block
<img src="./assets/llama_architecture.png" width="500">

## Getting Started
Please follow the instructions given in the [docs](https://yangfelix.github.io/Llama2Inference.jl/dev/getting_started/).