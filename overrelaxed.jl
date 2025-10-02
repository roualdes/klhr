using Distributions

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
