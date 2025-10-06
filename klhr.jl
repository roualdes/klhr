include("bsmodel.jl")
include("onlinemoments.jl")
include("onlinepca.jl")
include("overrelaxed.jl")
include("windowedadaptation.jl")

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using Optim

function unpack(eta, cutoff = 700.0)
    m = eta[1]
    s = exp(clamp(eta[2], -cutoff, cutoff))
    return m, s
end

function KL(logp, eta, x, w, rho, origin)
    N = length(x)
    out = 0.0
    m, s = unpack(eta)
    sqrt2 = sqrt(2)
    for n in 1:N
        z = sqrt2 * s * x[n] + m
        xi = rho * z + origin
        lp = logp(xi)
        out += w[n] * lp
    end
    out /= sqrt(pi)
    out += eta[2]
    return -out
end

function KLgrad!(grad, logpg, eta, x, w, rho, origin)
    N = length(x)
    m, s = unpack(eta)
    sqrt2 = sqrt(2)
    for n in 1:N
        ssx = sqrt2 * s * x[n]
        z = ssx + m
        xi = rho * z + origin
        _, g = logpg(xi)
        wgr = w[n] * (g' * rho)
        grad[1] += wgr
        grad[2] += wgr * ssx
    end
    grad /= sqrt(pi)
    grad[2] += 1
    grad *= -1
end

function random_direction(evals, evecs, v)
    p = evals ./ sum(evals)
    m = sum(evecs .* p', dims = 2)[:]
    rho = rand(MvNormal(m, Diagonal(v)))
    return rho ./ norm(rho)
end

function batch_match(grad_logp, rho, prev, B)
    m, mn = 0.0, 0.0
    s, sn = 0.1, 0.1

    oz = OnlineMoments(1)
    og = OnlineMoments(1)

    t = 0
    while true
        for _ in 1:B
            z = randn() * s + m
            update!(oz, [z])
            _, grad = grad_logp(rho * z + prev)
            g = dot(grad, rho)
            println("g = $(g)")
            update!(og, [g])
        end

        zbar = oz.m[1]
        c = oz.v[1]
        gbar = og.m[1]
        gamma = og.v[1]

        println("gbar = $(gbar)")
        println("gamma = $(gamma)")

        lt = B / (t + 1)
        Lt = lt / (1 + lt)

        u = lt * gamma + Lt * gbar ^ 2
        v = s + lt * c + Lt * (m - zbar) ^ 2

        sn = 2v / (1 + (1 + 4u * v) ^ 0.5)
        mn = (1 - Lt) * m + Lt * (sn * gbar + zbar)

        if isapprox(m, mn, atol = 1e-2) && isapprox(s, sn, atol = 1e-2)
            println("batch & match took $(t) iterations to converge")
            break
        end

        if t >= 100
            println("max iterations reach, t = $(t)")
            if isnan(sn)
                sn = 1e-2
                println("sn is nan")
                println("zbar = $(zbar), c = $(c), gbar = $(gbar), gamma = $(gamma), u = $(u), v = $(v)")
            else
                sn = clamp(sn, 1e-10, 1e10)
            end

            if isnan(mn)
                mn = 0.0
                println("mn is nan")
                println("zbar = $(zbar), c = $(c), gbar = $(gbar), gamma = $(gamma), u = $(u), v = $(v)")
            else
                mn = clamp(mn, -1e10, 1e10)
            end
            break
        end

        m = isnan(mn) ? 0.0 : mn
        s = isnan(sn) ? 1.0 : sn
        t += 1
    end
    return mn, sn
end

function klhr(bsmodel;
              M = 1_000,
              warmup = div(M, 2),
              N = 16,
              J = 2,
              K = 10,
              init = [],
              windowsize = 50,
              windowscale = 2)

    logp = bsmodel_ld(bsmodel)
    logp_grad = bsmodel_ldgh(bsmodel)
    D = bsmodel_dim(bsmodel)

    wa = WindowedAdaptation(warmup; windowsize, windowscale)
    onlinemoments = OnlineMoments(D)
    rm = zeros(D)
    rv = ones(D)

    onlinepca = OnlinePCA(D, J)
    reigenvecs = zeros(D, J)
    reigenvals = ones(J)

    draws = zeros(M, D)
    draws[1, :] .= if length(init) == D
        init
    else
        rand(Uniform(-1, 1), D)
    end

    # x, w = gausshermite(N)
    acceptance_rate = 0.0

    for m in 2:M
        rho = random_direction(reigenvals, reigenvecs, rv)
        prev = draws[m - 1, :]
        mkl, skl = batch_match(logp_grad, rho, prev, 4)

        # L(eta) = KL(logp, eta, x, w, rho, prev)
        # grad!(g, eta) = KLgrad!(g, logp_grad, eta, x, w, rho, prev)
        # ieta = randn(2) * 0.1
        # res = Optim.optimize(L, grad!, ieta, BFGS())
        # eta = Optim.minimizer(res)
        # if Optim.iterations(res) == 0
        #     println("@ iteration $(m) didn't find minimum")
        # end

        # mkl, skl = unpack(eta)
        Q = Normal(mkl, skl)
        # z = overrelaxed_proposal(Q, K) #
        xi = rand(Q)
        prop = rho * xi + prev

        a = logp(prop)
        a -= logp(prev)
        a += logpdf(Q, 0.0)
        a -= logpdf(Q, xi)

        accept = log(rand()) < min(0.0, a)
        draws[m, :] .= accept * prop + (1 - accept) * prev
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
