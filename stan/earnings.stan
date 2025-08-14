data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
}

parameters {
  vector[2] beta;
  real<lower=0> sigma;
  real<lower=0> s;
}
model {
  s ~ exponential(0.01);
  beta ~ student_t(5, 0, s); /* 722.718); */
  sigma ~ exponential(0.1);
  earn ~ normal(beta[1] + beta[2] * height, sigma); /* 13040.7); */
}
