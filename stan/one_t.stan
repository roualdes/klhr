data {
}
parameters {
  real y;
}
model {
  y ~ student_t(1, 0, 1);
}
