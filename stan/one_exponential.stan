data {
}
parameters {
  real<lower=0> y;
}
model {
  y ~ exponential(30);
}
