# klhr

New Metropolis-Hastings algorithm: KL Hit and Run

The direct-curvature direction component uses BridgeStan's full Hessian API.
Compile model libraries with `BRIDGESTAN_AD_HESSIAN=true` to obtain autodiff
Hessians; the project Dockerfile enables this flag. Without it, BridgeStan
falls back to finite differences of gradients.
