import Pkg
Pkg.add("BenchmarkTools")
using BenchmarkTools

mutable struct Test_mutable
    arr::AbstractVector{Float64}
end

struct Test
    arr::AbstractVector{Float64}
end

function change_mutable(tmp::Test_mutable)
    tmp.arr[:1000] .+= π
end

function change(tmp::Test)
    v = @view tmp.arr[:100]
    v .+= π
end

function matmul(dest, x, w, n, d)
    for i in 1:d
        val = Float64(0)
        for j in 1:n
            val .+= w[i*n + j] * x[j]
        end
        dest[i] = val
    end
end    

r = rand(1000)
tmp_mutable = Test_mutable(r)
tmp = Test(r)

# @benchmark change_mutable(tmp_mutable)
# @benchmark change(tmp)
