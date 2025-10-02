include("bsmodel.jl")
include("onlinemoments.jl")
include("onlinepca.jl")
include("windowedadaptation.jl")

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Optim

function LVI(ldg, eta, x, w, rho, origin)
    N = length(x)
    dm = zero(eltype(x))
    ds = zero(eltype(x))
    out = zero(eltype(x))
    for n in 1:N
        z = exp(eta[2]) * x[n] + eta[1]
        xi = rho * z + origin
        ld, g = ldg(xi)
        wn = w[n]
        out += wn * ld
        dm += wn * (g' * rho)
        ds += wn * (g' * rho * exp(eta[2]) * x[n])
    end
    return -out - eta[2], -dm, -ds - 1
end

function fit(ldg, rho, origin; N = 20, tol = 1e-2)
    x, w = gausshermite(N, normalize = true);

    function fg!(F, G, eta)
        out, dm, ds = LVI(ldg, eta, x, w, rho, origin)
        if G !== nothing
            G[1] = dm
            G[2] = ds
        end
        if F !== nothing
            return out
        end
    end

    init = zeros(2)
    obj = OnceDifferentiable(Optim.only_fg!(fg!), init)
    method = BFGS()
    opts = Optim.Options(x_abstol = tol, x_reltol = tol, f_abstol = tol, f_reltol = tol, g_abstol = tol)
    state = Optim.initial_state(method, opts, obj, init)
    r = Optim.optimize(obj, init, method, opts, state)
    # mkl = r.minimizer[1]
    # skl = sqrt(state.invH[1, 1])
    return r.minimizer
end

"""
sinh-asinh transformation.

mu in R, sigma > 0, delta > 0, and epsilon in R

location dicted by mu
scale dictaed by sigma
tail weight dictated by delta
skewness dicated by epsilon
"""
function sas(z, mu, sigma, delta, epsilon)
    return mu + sigma * sinh(delta * asinh(z) + epsilon)
end

"""
fit a sinh-arcsinh distirbution

https://academic.oup.com/jrssig/article/16/2/6/7029435?login=true
"""
function LVI_sas(ldg, eta, x, w, rho, origin)
    N = length(x)
    dm = zero(eltype(x))
    ds = zero(eltype(x))
    out = zero(eltype(x))
    for n in 1:N
        z = sas(x[n], eta[1], exp(eta[2]), exp(eta[3]), eta[4])
        # TODO what else?
        xi = rho * z + origin
        ld, g = ldg(xi)
        wn = w[n]
        out += wn * ld
        dm += wn * (g' * rho)
        ds += wn * (g' * rho * exp(eta[2]) * x[n])
    end
    return -out - eta[2], -dm, -ds - 1
end

"""
https://arxiv.org/pdf/bayes-an/9506004
"""
function overrelaxed_proposal(NDist, K)
    u = cdf(NDist, 0)
    r = rand(Binomial(K, u))
    up = if r > K - r
        v = rand(Beta(K - r + 1, 2r - K))
        u * v
    elseif r < K - r
        v = rand(Beta(r + 1, K - 2r))
        1 - (1 - u) * v
    elseif r == K - r
        u
    end
    return quantile(NDist, up)
end

"""
Original KLHR.
"""
function klhr(bsmodel;
              M = 1_000,
              warmup = div(M, 2),
              N = 10,
              J = 2,
              K = 10,
              tol = 1e-4,
              init = [],
              windowsize = 50,
              windowscale = 2)

    logp = bsmodel_ld(bsmodel)
    logp_grad = bsmodel_ldg(bsmodel)
    D = bsmodel_dim(bsmodel)

    onlinemoments = OnlineMoments(D)
    rm = zeros(D)
    rv = ones(D)

    onlinepca = OnlinePCA(D, J)
    reigenvecs = zeros(D, J)
    reigenvals = ones(J)

    wa = WindowedAdaptation(warmup; windowsize, windowscale)

    draws = zeros(M, D)
    draws[1, :] = if length(init) == D
        draws[1, :] .= init
    else
        rand(Uniform(-1, 1), D)
    end

    acceptance_rate = 0.0
    mvn_direction = MvNormal(zeros(D), ones(D))

    for m in 2:M
        rho = rand(mvn_direction)
        rho ./= norm(rho)

        prev = draws[m - 1, :]
        mkl, skl = fit(logp_grad, rho, prev; N, tol)

        ND = Normal(mkl, skl)
        z = overrelaxed_proposal(ND, K) # randn(ND)
        prop = rho * z + prev

        a = logp(prop)
        a -= logp(prev)
        a += logpdf(ND, 0)
        a -= logpdf(ND, z)

        accept = log(rand()) < min(0, a)
        draws[m, :] = accept * prop + (1 - accept) * prev
        acceptance_rate += (accept - acceptance_rate) / (m - 1)

        if window_closed(wa, m)
            rm .= onlinemoments.m
            rv .= onlinemoments.v
            reset!(onlinemoments)

            reigenvecs .= vectors(onlinepca)
            reigenvals .= values(onlinepca)
            reset!(onlinepca)
        else
            update!(onlinemoments, draws[m, :])
            update!(onlinepca, draws[m, :] .- rm)
        end
    end

    println("acceptance rate = $(acceptance_rate)")
    return draws
end
