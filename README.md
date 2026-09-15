# klhr

New Metropolis-Hastings algorithm: KL Hit and Run

Each draw picks a direction from a Gaussian shaped like a Welford-adapted
diagonal covariance, fits a one-dimensional approximation (normal or
sinh-arcsinh) to the target along that line by minimizing KL, and proposes by
ordered overrelaxation at a fixed level `K` against the fitted density.

The library targets spherically symmetric posteriors; the `normal`,
`ill-normal`, and `funnel` Stan models are the reference cases.

The `slice` sampler shares that direction law and metric adaptation exactly,
so the two differ only in what they do along the line. Its univariate update
follows Neal (2003) figure 3 at Neal's own defaults: step size `w = 1` and no
limit on stepping out or on shrinkage (`--max-steps-out 0`,
`--max-shrink-steps 0`). The width is scaled by the adapted metric, which is
the one departure from a fixed `w`.

The `barker`, `mala`, and `stan` samplers are kept for comparison only and
have not been reorganized.
