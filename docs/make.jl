push!(LOAD_PATH,"../src/")
using Llama2Inference
using Documenter

DocMeta.setdocmeta!(Llama2Inference, :DocTestSetup, :(using Llama2Inference); recursive=true)

makedocs(;
    modules=[Llama2Inference],
    authors="Veronika Dimitrova <veronika.dimitrova@campus.tu-berlin.de>, Johann-Ludwig Herzog <herzog.2@campus.tu-berlin.de>, Phi Long Mikesch <mikesch@campus.tu-berlin.de>, Yang Felix Wang <yang.f.wang@campus.tu-berlin.de>",
    sitename="Llama2Inference.jl",
    format=Documenter.HTML(;
        canonical="https://yangfelix.github.io/Llama2Inference.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
    ],
    workdir = joinpath(@__DIR__, ".."),
)

deploydocs(;
    repo="github.com/yangfelix/Llama2Inference.jl",
    devbranch="main",
)
