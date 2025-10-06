using Pkg
Pkg.activate(".");

include("bsmodel.jl")
include("klhr.jl")

using BridgeStan
const BS = BridgeStan

# using CairoMakie
using Comonicon
using Statistics

"""
Run relaxationtime experiment

# Arguments

- `algorithm`: algorithm name, either klhr or klhrsinh

# Options

- `--iterations, -i`: number of sampling iterations
- `--warmup, -w`: number of warmup iterations

# Flags

- `-v, --verbose`: print stuff
"""
Comonicon.@main function main(algorithm;
                              iterations::Int64=2_000,
                              warmup::Int64=div(iterations, 2),
                              verbose::Bool=false)

    BS.set_bridgestan_path!(joinpath(homedir(), "bridgestan"))
    model = "normal"
    source_dir = dirname(@__FILE__)
    bsmodel = BS.StanModel(joinpath(source_dir, "stan/$(model).stan"),
                           joinpath(source_dir, "stan/$(model).json"))


    if algorithm == "klhr"
        draws = klhr(bsmodel;
                     M = iterations,
                     warmup = warmup)
    else
        println("don't know what to do yet with $(algorithm)")
    end

    println(mean(draws[warmup+1:end, :], dims = 1));
    println(std(draws[warmup+1:end, :], dims = 1));
end
