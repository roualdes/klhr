data {
  int<lower=0> N;
  vector[N] earn;
  vector[N] height;
}

parameters {
  real<lower=0> s;
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  s ~ exponential(0.01);
  beta ~ student_t(5., 0., s); /* 722.718); */
  sigma ~ exponential(0.1);
  earn ~ normal(beta[1] + beta[2] * height, sigma); /* 13040.7); */
}
