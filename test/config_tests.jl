using Llama2Inference
using Test

@testset "set_config_vocab_size" begin
    config = Config(1, 2, 3, 4, 5, 6, 7)
    updated_config = set_config_vocab_size(config, 8)
    @test config.dim == updated_config.dim
    @test config.hidden_dim == updated_config.hidden_dim
    @test config.n_layers == updated_config.n_layers
    @test config.n_heads == updated_config.n_heads
    @test config.n_kv_heads == updated_config.n_kv_heads
    @test config.seq_len == updated_config.seq_len
    @test config.vocab_size == 6
    @test updated_config.vocab_size == 8
end

@testset "read_config" begin
    io = IOBuffer()
    data::Vector{Int32} = [i for i in 1:200]
    bw = write(io, data)
    seekstart(io)
    config = read_config(Int32, io)
    @test config.dim == 1
    @test config.hidden_dim == 2
    @test config.n_layers == 3
    @test config.n_heads == 4
    @test config.n_kv_heads == 5
    @test config.vocab_size == 6
    @test config.seq_len == 7
end
