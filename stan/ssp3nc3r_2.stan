data {
  int<lower=0> G;
  int<lower=0> J;
  int<lower=0> n;
  array[G, J, n] real y;
}
parameters {
  real mu;
  real<lower=0> tau_top;
  real<lower=0> tau_sub;
  vector[G] lmu;
  matrix[G, J] lsig;
}
model {
  mu ~ normal(0, 1);
  tau_top ~ normal(0, 1);
  tau_sub ~ normal(0, 1);
  lmu ~ normal(mu, tau_top);
  for (g in 1:G) {
    for (j in 1:J) {
      lsig[g, j] ~ normal(lmu[g], tau_sub);
      y[g, j] ~ normal(0, exp(lsig[g, j]));
    }
  }
}
