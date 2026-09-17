local funnel = {
  model_name: "funnel",
  replications: 20,
  seed: 530,
  iterations: 1e07,
  warmup: 1e05,
};

local illnormal = {
  model_name: "ill-normal",
  replications: 20,
  seed: 898,
  iterations: 1e05,
  warmup: 5e04,
};

local normal = {
  model_name: "normal",
  replications: 20,
  seed: 204,
  iterations: 1e05,
  warmup: 5e04,
};

local samplers = ["sas", "normal", "stan", "slice"];

[
  model
  + { sampler: s }
  + (if s == "stan" && model == funnel then { target_accept: 0.95 } else {})
  for model in [funnel, illnormal, normal]
  for s in samplers
]
