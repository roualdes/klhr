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
  vector[G] z_top;              // non-centered
  matrix[G, J] z_sub;           // non-centered
}
transformed parameters {
    vector[G] lmu = mu + tau_top * z_top;
    matrix[G, J] lsig;
    for (g in 1:G) {
      for (j in 1:J) {
        lsig[g, j] = lmu[g] + tau_sub * z_sub[g, j];
      }
    }
}
model {
  mu ~ normal(0, 1);
  tau_top ~ normal(0, 1);
  tau_sub ~ normal(0, 1);
  z_top ~ std_normal();
  to_vector(z_sub) ~ std_normal();
  for (g in 1:G) {
    for (j in 1:J) {
      y[g, j] ~ normal(0, exp(lsig[g, j]));
    }
  }
}
