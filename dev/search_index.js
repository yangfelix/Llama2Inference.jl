var documenterSearchIndex = {"docs":
[{"location":"getting_started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"This page shows some examples on how to use the Llama2Inference module. ","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Our first example generates a random story, while the second one uses a random factor, so the output may differ each time you run it.","category":"page"},{"location":"getting_started/#Prerequisites","page":"Getting Started","title":"Prerequisites","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Before running the examples, you need to download the necessary weights. In this example, we use the weights for a small 15M model trained by Andrej Karpathy. Follow these steps:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Navigate to the directory where you want to download the weights.\nRun the following command:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Additionally, you will need to download the tokenizer.bin file from the GitHub repository.","category":"page"},{"location":"getting_started/#Setup","page":"Getting Started","title":"Setup","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Navigate to the directory where you downloaded the weights.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia\nusing Pkg\nPkg.activate(\"Llama2Inference\")\nPkg.add(url=\"https://github.com/yangfelix/Llama2Inference.jl\")\n","category":"page"},{"location":"getting_started/#Generating-a-random-Story","page":"Getting Started","title":"Generating a random Story","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"using Llama2Inference\nconfig, weights = read_checkpoint(\"./stories15M.bin\");\ntransformer = Transformer(config, weights);\ntokenizer = build_tokenizer(\"./tokenizer.bin\", Int(config.vocab_size));\nsampler = Sampler(config.vocab_size, 0.5f0, 0.9f0);\ngenerate(transformer, tokenizer, sampler, 256; prompt=\"The universe\")","category":"page"},{"location":"getting_started/#Generating-a-deterministic-Story","page":"Getting Started","title":"Generating a deterministic Story","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"using Llama2Inference\nconfig, weights = read_checkpoint(\"./stories15M.bin\");\ntransformer = Transformer(config, weights);\ntokenizer = build_tokenizer(\"./tokenizer.bin\", Int(config.vocab_size));\nsampler = Sampler(config.vocab_size, 0.0f0, 0.9f0);\ngenerate(transformer, tokenizer, sampler, 256; prompt=\"The universe\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"note: Note\nThe temperature and topp argument of the Sampler control the random factor of the sampled tokens at each timestep and therefore directly control the diversity generated stories with the same setup.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Llama2Inference","category":"page"},{"location":"#Llama2Inference","page":"Home","title":"Llama2Inference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Llama2Inference.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Llama2Inference]","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Llama2Inference]\nPages = [\"transformer.jl\", \"Tokenizer.jl\", \"sampler.jl\", \"config.jl\", \"transformer_weights.jl\", \"runstate.jl\"]","category":"page"},{"location":"#Llama2Inference.Transformer","page":"Home","title":"Llama2Inference.Transformer","text":"Transformer struct which contains the fields\n\nconfig::Config\nweights::TransformerWeights\n\n\n\n\n\n","category":"type"},{"location":"#Llama2Inference.forward!-Tuple{Transformer, RunState, Int64, Int64}","page":"Home","title":"Llama2Inference.forward!","text":"forward!(transformer::Transformer, state::RunState, token::Int, pos::Int)\n\nForward the token at position pos through the transformer model.\n\nThe foward pass corresponds to the LlaMa2 decoder architecture and is based on the C implementation by Andrej Karpathy. The output is a logits vector of size transformer.config.vocab_size.\n\nArguments\n\ntransformer::Transformer: The transformer object containg config, weights and the token embedding table.\nstate::RunState: The state object to store the intermediate results during the forward pass.\ntoken::Int: The token to forward through the transformer.\npos::Int: The position of the token in the sequence. The position is 1-based, which means the first position in a sequence is 1.\n\nExample\n\njulia> config, weights = read_checkpoint(\"./stories15M.bin\")\njulia> transformer = Transformer(config, weights)\njulia> tokenizer = build_tokenizer(\"./tokenizer.bin\", Int(config.vocab_size))\njulia> state = RunState(config)\njulia> token = 2\njulia> pos = 1\njulia> forward!(transformer, state, token, pos)\njulia> state.logits\n32000-element Vector{Float32}:\n -6.7907834\n  0.82811606\n -6.7904234\n -6.790472\n  ⋮\n -6.79068\n -6.790696\n -6.7906737\n -6.7905493\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.generate-Tuple{Transformer, Tokenizer, Sampler, Int64}","page":"Home","title":"Llama2Inference.generate","text":"generate(transformer::Transformer, tokenizer::Tokenizer, sampler::Sampler, steps::Int; prompt::String=\"\", performance=true)\n\nGenerate a sequence of tokens using the transformer.\n\nArguments\n\ntransformer::Transformer: The transformer object containg config and weights.\ntokenizer::Tokenizer: The tokenizer object to encode and decode tokens.\nsampler::Sampler: The sampler object to sample a token from the output logits.\nsteps::Int: The number of maximum tokens to generate, upper bound by transformer.config.seq_len.\nprompt::String: The input text to start the generation. If none, the generation starts with an empty string.\nperformance::Bool: If true, print the number of generated tokens and number of tokens generated per second.\n\nExample\n\njulia> config, weights = read_checkpoint(\"./stories15M.bin\")\njulia> transformer = Transformer(config, weights)\njulia> tokenizer = build_tokenizer(\"./tokenizer.bin\", Int(config.vocab_size))\njulia> sampler = Sampler(config.vocab_size, 0.0f0, 0.9f0)\njulia> generate(transformer, tokenizer, sampler, 23; prompt=\"The universe\", performance=true)\nThe universe was bright and full of stars. Every night, the stars would twinkle and shine.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.mat_T_vec!-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}, AbstractMatrix{T}}} where T<:AbstractFloat","page":"Home","title":"Llama2Inference.mat_T_vec!","text":"mat_T_vec!(out::AbstractArray{T,1}, x::AbstractArray{T,1}, w::AbstractArray{T,2}) where T<:AbstractFloat\n\nEfficient transpose(matrix)-vector multiplication out = w' * x.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.read_checkpoint","page":"Home","title":"Llama2Inference.read_checkpoint","text":"read_checkpoint(checkpoint::String, T_weights::Type=Float32, T_config::Type=Int32)::Tuple{Config{T_config},TransformerWeights{T_weights}}\n\nReads the config and weights from the file at checkpoint.\n\n\n\n\n\n","category":"function"},{"location":"#Llama2Inference.rmsnorm!-Union{Tuple{T}, Tuple{AbstractVector{T}, AbstractVector{T}, AbstractVector{T}}} where T<:AbstractFloat","page":"Home","title":"Llama2Inference.rmsnorm!","text":"rmsnorm!(out::AbstractArray{Float32, 1}, x::AbstractArray{Float32,1}, weight::AbstractArray{Float32,1})\n\nNormalize out in place by the root mean square of x and multiply with weight.\n\nout_i = fracx_iRMS(x) * weight_i quadtextwherequad RMS(x)= sqrt (frac1n * sum_i=1^n x_i^2) +1mathrme-5 \n\n1e-5 is added for numerical stability in the square root part.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.softmax!-Union{Tuple{AbstractVector{T}}, Tuple{T}} where T<:AbstractFloat","page":"Home","title":"Llama2Inference.softmax!","text":"softmax!(x::AbstractArray{Float32,1})\n\nSoftmax the values in x in place.\n\nx_i = frace^x_isum_j=1^n e^x_j\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.TokenIndex","page":"Home","title":"Llama2Inference.TokenIndex","text":"struct TokenIndex\n\nA data structure representing a token and its corresponding index.\n\nFields\n\nstr::String: The token as a string.\nid::Int: The ID of the token.\n\n\n\n\n\n","category":"type"},{"location":"#Llama2Inference.Tokenizer","page":"Home","title":"Llama2Inference.Tokenizer","text":"struct Tokenizer\n\nA data structure representing a tokenizer with its vocabulary and related properties.\n\nFields\n\nvocab_size::Int: The size of the vocabulary.\nvocab::Vector{String}: A vector containing the vocabulary tokens.\nvocab_scores::Vector{Float32}: A vector containing scores associated with each token in the vocabulary.\nmax_token_length::Int: The maximum length of a token.\nsorted_vocab::Union{Nothing, Vector{TokenIndex}}: Can be either nothing or a vector of TokenIndex structs representing the tokens and their respective indices.\n\n\n\n\n\n","category":"type"},{"location":"#Llama2Inference.build_tokenizer-Tuple{String, Int64}","page":"Home","title":"Llama2Inference.build_tokenizer","text":"build_tokenizer(filepath::String, vocab_size::Int)\n\nConstructs a Tokenizer from a file.\n\nArguments\n\nfilepath::String: Path to the file containing vocabulary data.\nvocab_size::Int: Size of the vocabulary to read from the file.\n\nReturns\n\nTokenizer: A Tokenizer instance with the necessary data.\n\nSummary Steps\n\nInitialization\nReading file\nHandling errors\nClosing file\nReturning Tokenizer with necessary data\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.decode-Tuple{Tokenizer, Int64, Int64}","page":"Home","title":"Llama2Inference.decode","text":"decode(tokenizer::Tokenizer, prev_token::Int, token::Int) -> Union{String, UInt8}\n\nDecodes a token ID into its corresponding token string representation or byte value using the Tokenizer.\n\nArguments\n\ntokenizer::Tokenizer: The Tokenizer containing vocabulary and token mapping.\nprev_token::Int: Token ID of the previous token in the sequence.\ntoken::Int: Token ID to decode into its corresponding token string representation.\n\nReturns\n\nbyte_val or token_str: Returns a decoded string or a byte value, depending on the type of the given token.\n\nDescription\n\nFinds the token string using find_token_str.\nRemoves leading whitespace if the previous token is BOS.\nChecks for raw byte tokens and parses them if applicable.\nReturns the token string or its byte value representation.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.encode-Tuple{Tokenizer, String, Bool, Bool}","page":"Home","title":"Llama2Inference.encode","text":"encode(tokenizer::Tokenizer, text::String, use_bos::Bool, use_eos::Bool) -> Vector{Int}\n\nEncodes the input text into a sequence of token IDs using the provided Tokenizer.\n\nArguments\n\ntokenizer::Tokenizer: The Tokenizer containing vocabulary and token mapping.\ntext::String: Input text to encode into tokens.\nuse_bos::Bool: Indicates if a token ID representing the beginning of the sequence should be included.\nuse_eos::Bool: Indicates if a token ID representing the end of the sequence should be included.\n\nReturns\n\nVector{Int}: Returns a vector of token IDs representing the encoded input text.\n\nDescription\n\nEnsures the tokenizer's vocabulary is sorted with sort_vocab!.\nEncodes text into bytes.\nInitializes the token ID vector.\nOptionally adds the BOS token.\nHandles leading whitespace.\nLooks up and stores token IDs with find_token_id.\nPerforms merges (BPE) based on scores (vocab_scores).\nOptionally adds the EOS token.\nReturns the vector tokens_indices representing the encoded input text.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.find_token_id-Tuple{Tokenizer, String}","page":"Home","title":"Llama2Inference.find_token_id","text":"find_token_id(tokenizer::Tokenizer, token_str::String)\n\nFinds and returns the ID of a given token string in the sorted_vocab.\n\nArguments\n\ntokenizer::Tokenizer: The Tokenizer to search within.\ntoken_str::String: The token string to search for its ID.\n\nReturn\n\nInt: The ID of the token if found, otherwise returns -1.\n\nDescription\n\nIterates over the sorted_vocab to find the token string and return its ID.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.find_token_str-Tuple{Tokenizer, Int64}","page":"Home","title":"Llama2Inference.find_token_str","text":"findtokenstr(tokenizer::Tokenizer, token_id::Int)\n\nFinds and returns the token string coressponding to a given token ID from sort_vocab\n\nArguments\n\ntokenizer::Tokenizer: Tokenizer to search within\ntoken_id::Int: ID of the given token to find its coressponding string\n\nReturn\n\ntoken_index.str: String of the token if found, otherwise returns nothing\n\nDescription\n\nIterates over the sorted_vocab to find the token ID and return its corresponding string.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.sort_vocab!-Tuple{Tokenizer}","page":"Home","title":"Llama2Inference.sort_vocab!","text":"sort_vocab!(tokenizer::Tokenizer)\n\nSorts the vocabulary of the given Tokenizer, storing unique tokens and sorting them.\n\nArgument\n\ntokenizer::Tokenizer: The Tokenizer whose vocabulary is to be sorted.\n\nSummary Steps\n\nChecking if sorted_vocab is nothing or empty.\nInitialization of data structures.\nIdentification of unique tokens.\nSorting the vocabulary.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.Sampler-Tuple{Int64, Float32, Float32}","page":"Home","title":"Llama2Inference.Sampler","text":"Sampler(vocab_size::Int, temperature::Float32, topp::Float32)\n\nCreate a sampler object with the given parameters, used for sampling from a distribution.\n\nArguments\n\nvocab_size::Int: The size of the vocabulary.\ntemperature::Float32: The temperature to apply to the logits before sampling.\ntopp::Float32: The probability mass to sample from the top-p distribution.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.sample-Tuple{Sampler, Vector{Float32}}","page":"Home","title":"Llama2Inference.sample","text":"sample(sampler::Sampler, logits::Vector{Float32})\n\nSample from the logits using the sampler.\n\nExample\n\njulia> sampler = Sampler(6, 0.0f0, 0.0f0)\njulia> logits = [0.1f0, 0.3f0, 0.2f0, 0.15f0, 0.15f0, 0.1f0]\njulia> sample(sampler, logits)\n2\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.sample_topp-Tuple{Sampler, Vector{Float32}, Float32}","page":"Home","title":"Llama2Inference.sample_topp","text":"sample_topp(sampler::Sampler, logits::Vector{Float32}, coin::Float32)::Int\n\ntop-p sampling (or \"nucleus sampling\") samples from the smallest set of tokens that exceed probability topp.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.Config","page":"Home","title":"Llama2Inference.Config","text":"Config{T<:Integer}\n\nStruct to hold the configuration of the Transformer used for inference. It contains the following fields:\n\ndim::T: transformer dimension\nhidden_dim::T: for ffn layers\nn_layers::T: number of layers\nn_heads::T: number of query heads\nnkvheads::T: number of key/value heads\nvocab_size::T: vocabulary size\nseq_len::T: max sequence length\n\nIt is possible to construct a Config using the constructor \n\nConfig{T}(dim::T, hidden_dim::T, n_layers::T, n_heads::T, n_kv_heads::T, vocab_size::T, seq_len::T)\n\nbut it is recommended to use the data format defined by Andrew Karpathy to store both the configuration and weights of the transformer used and read the config and weights using  read_checkpoint.\n\n\n\n\n\n","category":"type"},{"location":"#Llama2Inference.read_config-Tuple{Type{<:Integer}, IO}","page":"Home","title":"Llama2Inference.read_config","text":"read_config(T::Type{<:Integer}, file::IO)::Config{T}\n\nReads from file to create a Config{T}.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.set_config_vocab_size-Union{Tuple{T}, Tuple{Config{T}, T}} where T<:Integer","page":"Home","title":"Llama2Inference.set_config_vocab_size","text":"set_config_vocab_size(config::Config{T}, vocab_size::T)\n\nCreates a new Config{T} with the field vocab_size updated to vocab_size.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.TransformerWeights","page":"Home","title":"Llama2Inference.TransformerWeights","text":"TransformerWeights{T<:AbstractFloat}\n\nStruct to hold the weights of the Transformer used for inference.\n\nIt is possible to construct a TransformerWeights using the constructor \n\nTransformerWeights{T<:AbstractFloat}(\n    token_embedding_table::AbstractArray{T,2}, \n    rms_att_weight::AbstractArray{T,2}, \n    wq::AbstractArray{T,3}, \n    wk::AbstractArray{T,3}, \n    wv::AbstractArray{T,3},\n    wo::AbstractArray{T,3},\n    rms_ffn_weight::AbstractArray{T,2},\n    w1::AbstractArray{T,3},\n    w2::AbstractArray{T,3},\n    w3::AbstractArray{T,3},\n    rms_final_weight::AbstractArray{T,1},\n    wcls::AbstractArray{T,2}\n    )\n\nbut it is recommended to use the data format defined by Andrew Karpathy to store both the configuration and weights of the transformer used and read the config and weights using  read_checkpoint.\n\n\n\n\n\n","category":"type"},{"location":"#Llama2Inference.get_weights-Tuple{Vector, Any, Any, Any}","page":"Home","title":"Llama2Inference.get_weights","text":"get_weights(weights::Vector, offset_1, offset_2, dims)::AbstractArray\n\nCreates a view of weights from offset_1 to offset_2 with the dimension specified by dims.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.memory_map_weights-Union{Tuple{T}, Tuple{Config, Vector{T}, Int32}} where T<:AbstractFloat","page":"Home","title":"Llama2Inference.memory_map_weights","text":"memory_map_weights(config::Config, weights::Vector{T}, shared_weights::Int32)::TransformerWeights{T} where {T<:AbstractFloat}\n\nTakes the values in weights and maps creates a new TransformerWeights struct using config and shared_weights.\n\n\n\n\n\n","category":"method"},{"location":"#Llama2Inference.RunState-Tuple{Config}","page":"Home","title":"Llama2Inference.RunState","text":"RunState(config::Config)\n\nCreate a RunState object with the given configuration.\n\nThis is used to store the state of the model during inference and is 0-initialized.\n\n\n\n\n\n","category":"method"}]
}
