using Llama2Inference
using Test

@testset "transformer_weights Tests" begin
    @testset "get_weights" begin
        weights::Vector{Float32} = [i for i in 1:2000]
        offset_1 = 0
        offset_2 = 10
        offset_3 = 199
        offset_4 = 226
        dim_1 = (5, 2)
        dim_2 = (3, 3, 3)
        result_1 = get_weights(weights, offset_1, offset_2, dim_1)
        result_2 = get_weights(weights, offset_3, offset_4, dim_2)
        # test correct dimensions
        @test size(result_1) == dim_1
        @test size(result_2) == dim_2
        # test correct values
        @test result_1[:, 1] == weights[1:5]
        @test result_1[:, 2] == weights[6:10]
        @test reshape(result_2[:, :, 1], 9) == weights[200:208]
        @test reshape(result_2[:, :, 2], 9) == weights[209:217]
        @test reshape(result_2[:, :, 3], 9) == weights[218:226]
        # test that get_weights returns a view
        weights[1] = 0
        weights[226] = -1
        @test weights[1] == 0
        @test weights[226] == -1
        @test result_1[1] == weights[1]
        @test result_2[3, 3, 3] == weights[226]
    end

    @testset "memory_map_weights" begin
        dim = 16
        hidden_dim = 2
        n_layers = 4
        n_heads = 4
        n_kv_heads = 8
        vocab_size = 8
        head_size = div(dim, n_heads)
        seq_len = 0
        config = Config{Int32}(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        # number of necessary weights (shared_weights == 1, head_size = dim/n_heads) is 
        # vocab_size * dim 
        # + n_layers * dim 
        # + n_layers * dim * n_heads * head_size
        # + n_layers * dim * n_kv_heads * head_size
        # + n_layers * dim * n_kv_heads * head_size
        # + n_layers * dim * n_heads * head_size
        # + n_layers * dim 
        # + n_layers * dim * hidden_dim
        # + n_layers * dim * hidden_dim
        # + n_layers * dim * hidden_dim
        # + dim 
        # + seq_len * head_size
        # = 8 * 16 + 4 * 16 * (2 + 2 * 4 * 4 + 2 * 8 * 4 + 3 * 2) + 16 + 0
        # = 6800
        num_weights = vocab_size * dim + n_layers * dim + n_layers * dim * n_heads * head_size + n_layers * dim * n_kv_heads * head_size + n_layers * dim * n_kv_heads * head_size + n_layers * dim * n_heads * head_size + n_layers * dim + n_layers * dim * hidden_dim + n_layers * dim * hidden_dim + n_layers * dim * hidden_dim + dim + seq_len * head_size
        weights::Vector{Float32} = [i for i in 1:num_weights]
        shared_weights::Int32 = 1
        tfweights = memory_map_weights(config, weights, shared_weights)
        # test dimensions
        @test size(tfweights.token_embedding_table) == (dim, vocab_size)
        @test size(tfweights.rms_att_weight) == (dim, n_layers)
        @test size(tfweights.wq) == (dim, n_heads * head_size, n_layers)
        @test size(tfweights.wk) == (dim, n_kv_heads * head_size, n_layers)
        @test size(tfweights.wv) == (dim, n_kv_heads * head_size, n_layers)
        @test size(tfweights.wo) == (dim, n_heads * head_size, n_layers)
        @test size(tfweights.rms_ffn_weight) == (dim, n_layers)
        @test size(tfweights.w1) == (dim, hidden_dim, n_layers)
        @test size(tfweights.w2) == (hidden_dim, dim, n_layers)
        @test size(tfweights.w3) == (dim, hidden_dim, n_layers)
        @test size(tfweights.rms_final_weight) == (dim,)
        @test size(tfweights.wcls) == (dim, vocab_size)
        # test correct values

        @test reshape(tfweights.token_embedding_table, 128) == weights[1:128]
        @test reshape(tfweights.rms_att_weight, 64) == weights[129:192]
        @test reshape(tfweights.wq, 1024) == weights[193:1216]
        @test reshape(tfweights.wk, 2048) == weights[1217:3264]
        @test reshape(tfweights.wv, 2048) == weights[3265:5312]
        @test reshape(tfweights.wo, 1024) == weights[5313:6336]
        @test reshape(tfweights.rms_ffn_weight, 64) == weights[6337:6400]
        @test reshape(tfweights.w1, 128) == weights[6401:6528]
        @test reshape(tfweights.w2, 128) == weights[6529:6656]
        @test reshape(tfweights.w3, 128) == weights[6657:6784]
        @test reshape(tfweights.rms_final_weight, 16) == weights[6785:6800]
        @test reshape(tfweights.wcls, 128) == weights[1:128]

        # test that views are returned
        weights[begin] = 0
        weights[end] = -1
        @test weights[begin] == 0
        @test weights[end] == -1
        @test tfweights.token_embedding_table[begin] == weights[begin]
        @test tfweights.rms_final_weight[end] == weights[end]

    end

    @testset "transformer_weights_constructor_test" begin
        dim = 16
        hidden_dim = 2
        n_layers = 4
        n_heads = 4
        n_kv_heads = 8
        vocab_size = 8
        head_size = div(dim, n_heads) # 4
        seq_len = 0
        token_embedding_table = zeros(16, 8)
        rms_att_weight = zeros(16, 4)
        wq = zeros(16, 16, 4)
        wk = zeros(16, 32, 4)
        wv = zeros(16, 32, 4)
        wo = zeros(16, 16, 4)
        rms_ffn_weight = zeros(16, 4)
        w1 = zeros(16, 2, 4)
        w2 = zeros(2, 16, 4)
        w3 = zeros(16, 2, 4)
        rms_final_weight = zeros(16)
        wcls = zeros(16, 8)
        @test TransformerWeights{Float64}(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls) isa TransformerWeights{Float64}
        wq_wrong_dim = zeros(17, 16, 4)
        @test_throws ArgumentError TransformerWeights{Float64}(token_embedding_table, rms_att_weight, wq_wrong_dim, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls)
        rms_final_weight_wrong_dim = zeros(32)
        @test_throws ArgumentError TransformerWeights{Float64}(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight_wrong_dim, wcls)
        wcls_wrong_vocab_size = zeros(16, 7)
        @test_throws ArgumentError TransformerWeights{Float64}(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls_wrong_vocab_size)
        rms_ffn_weight_wrong_n_layers = zeros(16, 8)
        @test_throws ArgumentError TransformerWeights{Float64}(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight_wrong_n_layers, w1, w2, w3, rms_final_weight, wcls)
        w2_wrong_hidden_dim = zeros(16, 2, 4)
        @test_throws ArgumentError TransformerWeights{Float64}(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2_wrong_hidden_dim, w3, rms_final_weight, wcls)
        wv_wrong_n_kv_heads = zeros(16, 16, 4)
        @test_throws ArgumentError TransformerWeights{Float64}(token_embedding_table, rms_att_weight, wq, wk, wv_wrong_n_kv_heads, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls)
    end
end