include("bsmodel.jl")
include("onlinemoments.jl")
include("onlinepca.jl")
include("overrelaxed.jl")
include("windowedadaptation.jl")

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Optim

function unpack(eta, cutoff = 700)
    m = eta[1]
    s = exp(clamp(eta[2], -cutoff, cutoff))
    return m, s
end

function L(ldg, eta, x, w, rho, origin)
    N = length(x)
    out = zero(eltype(x))
    m, s = unpack(eta)
    for n in 1:N
        z = s * x[n] + m
        xi = rho * z + origin
        ld = ld(xi)
        out += w[n] * ld
    end
    out += eta[2]
    return -out
end

function grad!(g, ldg, eta, x, w, rho, origin)
    N = length(x)
    m, s = unpack(eta)
    for n in 1:N
        z = s * x[n] + m
        xi = rho * z + origin
        _, g = ldg(xi)


    end
    return -grad
end

function LVI(ldgh, eta, x, w, rho, origin)
    N = length(x)
    out = zero(eltype(x))
    grad = zeros(eltype(x), 2)
    #hess = zeros(eltype(x), 2, 2)
    m, s = unpack(eta)
    for n in 1:N
        xn = x[n]
        z = s * xn + m
        xi = rho * z + origin
        ld, g, h = ldgh(xi)
        wn = w[n]
        out += wn * ld
        grho = g' *  rho
        grad[1] += wn * grho
        grad[2] += wn * grho * s * xn
        # Hrho2 = dot(rho' * h, rho)
        # sq = ones(2, 2) * Hrho2
        # t = sqrt(2.0) * xn * s
        # sq[1, 2] *= t
        # sq[2, 1] *= t
        # sq[2, 2] *= t ^ 2
        # sq[2, 2] += (g' * rho) * t
        # hess .+= wn .* sq
    end
    out += eta[2]
    grad[2] += 1
    return -out, -grad #, -hess
end

function fit(ldg, rho, origin; N = 20, tol = 1e-2)
    x, w = gausshermite(N, normalize = true);

    function fgh!(F, G, eta)
        out, grad = LVI(ldg, eta, x, w, rho, origin)
        # if H !== nothing
        #     H .= hess
        # end
        if G !== nothing
            G .= grad
        end
        if F !== nothing
            return out
        end
    end

    init = zeros(2)
    obj = OnceDifferentiable(Optim.only_fg!(fgh!), init)
    method = BFGS()
    # method = NewtonTrustRegion()
    opts = Optim.Options(x_abstol = tol, x_reltol = tol, f_abstol = tol, f_reltol = tol, g_abstol = tol)
    state = Optim.initial_state(method, opts, obj, init)
    r = Optim.optimize(obj, init, method, opts, state)
    return r.minimizer
end


function random_direction(evals, evecs, v)
    p = evals ./ sum(evals)
    m = sum(evecs .* p', dims = 2)[:]
    rho = rand(MvNormal(m, Diagonal(v)))
    return rho ./ norm(rho)
end


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
    logp_grad = bsmodel_ldgh(bsmodel)
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

    for m in 2:M
        rho = random_direction(reigenvals, reigenvecs, rv)
        rho ./= norm(rho)

        prev = draws[m - 1, :]
        # eta = fit(logp_grad, rho, prev; N, tol)


        mkl, skl = unpack(eta)
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
