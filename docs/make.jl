using DisjunctiveAlgorithms
using Documenter

DocMeta.setdocmeta!(DisjunctiveAlgorithms, :DocTestSetup, :(using DisjunctiveAlgorithms); recursive=true)

makedocs(;
    modules=[DisjunctiveAlgorithms],
    authors="Daniel Nguyen",
    sitename="DisjunctiveAlgorithms.jl",
    format=Documenter.HTML(;
        canonical="https://infiniteopt.github.io/DisjunctiveAlgorithms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/infiniteopt/DisjunctiveAlgorithms.jl",
)
