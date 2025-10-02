using LinearAlgebra
using Test
using Statistics

"""
    Adapted from https://www.cse.msu.edu/~weng/research/CCIPCApami.pdf
"""
struct OnlinePCA{T<:AbstractFloat}
    n::Base.RefValue{Int64}
    D::Int64
    K::Int64
    l::Int64
    v::Matrix{T}
    tol::T
end

Base.eltype(::OnlinePCA{T}) where {T} = T

function OnlinePCA(D, K; l = 0, tol::Float64 = 1.0e-12)
    return OnlinePCA(Ref(0), D, K, l, zeros(Float64, D, K), tol)
end

function update!(opca::OnlinePCA, u::AbstractVector)
    opca.n[] += 1
    for i in 1:min(opca.K, opca.n[])
        if i == opca.n[]
            opca.v[:, i] .= u
        else
            w = (opca.n[] - 1.0 - opca.l) / opca.n[]
            v = opca.v[:, i]
            opca.v[:, i] .= w .* v .+ (1 - w) .* u .* (u' * v) ./ (norm(v) + opca.tol) # eq 10
            v = opca.v[:, i]
            u = u .- (u' * v) .* v ./ (norm(v) ^ 2 + opca.tol)
        end
    end
end

function values(opca::OnlinePCA)
    nv = map(norm, eachcol(opca.v))
    if any(isnan.(nv)) || any(isinf.(nv))
        nv .= 0.0
    end
    return nv .+ opca.tol
end

function vectors(opca::OnlinePCA)
    return opca.v ./ values(opca)'
end

function reset!(om::OnlinePCA)
    om.n[] = 0
    om.v .= 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "OnlinePCA test" begin
        D = 2
        N = 10_000
        X = randn(N, D)

        opca = OnlinePCA(2, 2)
        for n in 1:N
            update!(opca, X[n, :])
        end

        println(vectors(opca))

        F = svd(cov(X))
        println(F.Vt')
    end
end
