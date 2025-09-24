using BridgeStan
const BS = BridgeStan

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Optim

function bsmodel_ld(bsmodel)
    function bsm_ld(x)
        ld = try
            BS.log_density(bsmodel, x)
        catch e
            -Inf
        end
        return ld
    end
    return bsm_ld
end

function bsmodel_ldg(bsmodel)
    function bsm_ldg(x)
        ld, g = try
            BS.log_density_gradient(bsmodel, x)
        catch e
            D = BS.param_unc_num(bsmodel)
            -Inf, zeros(D)
        end
        return ld, g
    end
    return bsm_ldg
end

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
    mkl = r.minimizer[1]
    skl = sqrt(state.invH[1, 1])
    return mkl, skl
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

function onlinemoments!(m, v, n, x)
    w = 1.0 / n
    d = x .- m
    m .+= d .* w
    v .+= -v .* w .+ d .* d .* w .* (1 - w)
end

function calculatewindows(warmup, iterations, windowsize = 50, windowscale = 2)
    windows = []
    if warmup > windowsize
        push!(windows, windowsize)
        while true
            windowsize *= windowscale
            closewindow = windows[end]
            nextclose_window = closewindow + windowsize
            if closewindow + windowscale * windowsize >= warmup
                closewindow = warmup
            else
                closewindow = nextclose_window
            end

            push!(windows, closewindow)

            if closewindow >= warmup
                break
            end
        end
    end
    return windows
end

function onlinepca!(evecs, evals, n, u, l = 0, tol = 1e-10)
    K = length(evals)
    for i in 1:min(K, n)
        if i == n - 1
            evecs[:, i] .= u
        else
            w = (n - 1 - l) / n
            v = evecs[:, i]
            nv = norm(v)
            v[:, i] .= w .* v .+ (1 - w) .* u .* u' * v ./ (nv + tol)
            v = v[:, i]
            nv = norm(v)
            u .-= u' * v .* v ./ (nv ^ 2 + tol)
        end
    end
end

function eigen_values(evals, tol = 1e-10)
    nvs = map(norm, eachcol(evals))
    if any(isinf.(nvs)) || any(isnan.(nvs))
        nvs .= 0.0
    end
    return evals ./ (nvs .+ tol)
end

"""
Original KLHR.
"""
function klhr(bsmodel; M = 1_000, N = 10,
              overrelaxed = false, K = 10,
              tol = 1e-4, init = [])
    D = BS.param_unc_num(bsmodel)
    draws = zeros(M, D)
    draws[1, :] = if length(init) == D
        draws[1, :] .= init
    else
        rand(Uniform(-1, 1), D)
    end

    acceptance_rate = 0.0
    mvn_direction = MvNormal(zeros(D), ones(D))

    logp = bsmodel_ld(bsmodel)
    logp_grad = bsmodel_ldg(bsmodel)

    for m in 2:M
        rho = rand(mvn_direction)
        rho ./= norm(rho)

        prev = draws[m - 1, :]
        mkl, skl = fit(logp_grad, rho, prev; N, tol)

        ND = Normal(mkl, skl)
        z = if overrelaxed
            #
            overrelaxed_proposal(ND, K)
        else
            rand(ND)
        end

        prop = rho * z + prev

        a = logp(prop)
        a -= logp(prev)
        a += logpdf(ND, 0)
        a -= logpdf(ND, z)

        accept = log(rand()) < min(0, a)
        draws[m, :] = accept * prop + (1 - accept) * prev
        acceptance_rate += (accept - acceptance_rate) / (m - 1)
    end

    println("acceptance rate = $(acceptance_rate)")
    return draws
end
