using Test
using Statistics

struct OnlineMoments{T<:AbstractFloat}
    n::Base.RefValue{Int64}
    m::Vector{T}
    v::Vector{T}
end

Base.eltype(::OnlineMoments{T}) where {T} = T

function OnlineMoments(T, d)
    return OnlineMoments(Ref(0), zeros(T, d), ones(T, d))
end

OnlineMoments(d) = OnlineMoments(Float64, d)

function update!(om::OnlineMoments, x::AbstractVector)
    om.n[] += 1
    w = 1.0 / om.n[]
    d = x .- om.m
    om.m .+= d .* w
    om.v .+= -om.v .* w .+ d .^ 2 .* w .* (1 - w)
end

function reset!(om::OnlineMoments)
    om.n[] = 0
    om.m .= 0
    om.v .= 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "OnlineMoments test" begin
        om = OnlineMoments(2)

        N = 1_000
        X = randn(N, 2)

        for n in 1:N
            update!(om, X[n, :])
        end

        @test isapprox(om.m, mean(X, dims = 1)[:])
        @test isapprox(om.v, var(X, corrected = false, dims = 1)[:])

        reset!(om)
        @test om.n[] == 0
        @test all(om.m .== 0)
        @test all(om.v .== 1)
    end
end
