data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=2> T;
  array[N] int<lower=0> y;
  array[N] int<lower=0> n;
  array[N] int<lower=1, upper=S> study;
  array[N] int<lower=1, upper=T> treatment;
  int<lower=0, upper=1> prior_only;
}

parameters {
  vector[S] mu;
  vector[T - 1] d_raw;
}

transformed parameters {
  vector[T] d;
  d[1] = 0;
  for (t in 2:T) {
    d[t] = d_raw[t - 1];
  }
}

model {
  mu ~ normal(0, 5);
  d_raw ~ normal(0, 2);
  if (prior_only == 0) {
    for (i in 1:N) {
      y[i] ~ binomial_logit(n[i], mu[study[i]] + d[treatment[i]]);
    }
  }
}

generated quantities {
  array[N] real log_lik;
  array[N] int y_rep;
  for (i in 1:N) {
    log_lik[i] = binomial_logit_lpmf(y[i] | n[i], mu[study[i]] + d[treatment[i]]);
    y_rep[i] = binomial_rng(n[i], inv_logit(mu[study[i]] + d[treatment[i]]));
  }
}
