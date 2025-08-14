data {
  int<lower=1> N;
}
transformed data {
  real alpha = 0.9;
  real beta = sqrt(1 - alpha * alpha);
}
parameters {
  vector[N] y;
}
model {
  y[1] ~ std_normal();
  y[2:N] ~ normal(alpha * y[1:(N-1)], beta);
}
