data {
}
parameters {
  real y;
}
model {
  target += log_mix(0.5,
                   normal_lpdf(y | -5, 0.3),
                   normal_lpdf(y | 5, 0.3));
}
