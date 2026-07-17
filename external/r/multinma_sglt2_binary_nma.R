#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("multinma", quietly = TRUE)) {
    stop("Package 'multinma' is required. Install with install.packages('multinma').")
  }
  if (!requireNamespace("rstan", quietly = TRUE)) {
    stop("Package 'rstan' is required for multinma Stan sampling.")
  }
})

parse_args <- function(args) {
  out <- list(
    events = NULL,
    output = NULL,
    chains = 2L,
    iter = 600L,
    warmup = 300L,
    seed = 20260717L,
    adapt_delta = 0.95
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--events", "--output", "--chains", "--iter", "--warmup", "--seed", "--adapt-delta")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--events") out$events <- val
      if (key == "--output") out$output <- val
      if (key == "--chains") out$chains <- as.integer(val)
      if (key == "--iter") out$iter <- as.integer(val)
      if (key == "--warmup") out$warmup <- as.integer(val)
      if (key == "--seed") out$seed <- as.integer(val)
      if (key == "--adapt-delta") out$adapt_delta <- as.numeric(val)
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$events) || is.null(out$output)) {
    stop("Usage: multinma_sglt2_binary_nma.R --events <events.csv> --output <output.json>")
  }
  out
}

required_columns <- c(
  "study_id", "trial", "nct_id", "pmid", "outcome_id", "outcome_label",
  "arm_role", "treatment", "events", "n"
)

extract_relative_effect <- function(relative_summary) {
  if (is.list(relative_summary) && "summary" %in% names(relative_summary)) {
    as_frame <- as.data.frame(relative_summary$summary)
  } else {
    as_frame <- as.data.frame(relative_summary)
  }
  names(as_frame) <- make.names(names(as_frame))
  treatment_columns <- c(".trtb", "trtb", "trt", "trt1", "treatment", ".trt", "contrast")
  mean_columns <- c("mean", "Median", "median", "X50.")
  sd_columns <- c("sd", "SD")
  low_columns <- c("X2.5.", "q2.5", "X2.5", "lower", "ci_low")
  high_columns <- c("X97.5.", "q97.5", "X97.5", "upper", "ci_high")

  treatment_column <- intersect(treatment_columns, names(as_frame))[1]
  if (!is.na(treatment_column)) {
    active_rows <- grepl("SGLT2i", as.character(as_frame[[treatment_column]]), fixed = TRUE)
    if (any(active_rows)) {
      as_frame <- as_frame[active_rows, , drop = FALSE]
    }
  }
  if (nrow(as_frame) != 1) {
    stop("Could not identify exactly one SGLT2i relative-effect summary row.")
  }
  list(
    row_names = rownames(as_frame),
    columns = as.list(as_frame[1, , drop = FALSE]),
    mean = value_from_first_column(as_frame, mean_columns),
    sd = value_from_first_column(as_frame, sd_columns),
    ci_low = value_from_first_column(as_frame, low_columns),
    ci_high = value_from_first_column(as_frame, high_columns),
    bulk_ess = value_from_first_column(as_frame, c("Bulk_ESS", "Bulk.ESS")),
    tail_ess = value_from_first_column(as_frame, c("Tail_ESS", "Tail.ESS")),
    rhat = value_from_first_column(as_frame, c("Rhat"))
  )
}

value_from_first_column <- function(frame, candidates) {
  column <- intersect(candidates, names(frame))[1]
  if (is.na(column)) {
    return(NA_real_)
  }
  as.numeric(frame[[column]][[1]])
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  rows <- utils::read.csv(parsed$events, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(rows))
  if (length(missing) > 0) {
    stop(paste("SGLT2 events missing required columns:", paste(missing, collapse = ", ")))
  }
  if (any(rows$events <= 0) || any(rows$n - rows$events <= 0)) {
    stop("SGLT2 events have zero cells; explicit correction policy required.")
  }

  options(mc.cores = 1)
  network <- multinma::set_agd_arm(
    rows,
    study = study_id,
    trt = treatment,
    r = events,
    n = n,
    trt_ref = "Placebo"
  )
  fit <- multinma::nma(
    network,
    consistency = "consistency",
    trt_effects = "fixed",
    likelihood = "binomial",
    link = "logit",
    prior_intercept = multinma::normal(scale = 10),
    prior_trt = multinma::normal(scale = 2.5),
    chains = parsed$chains,
    iter = parsed$iter,
    warmup = parsed$warmup,
    seed = parsed$seed,
    refresh = 0,
    adapt_delta = parsed$adapt_delta
  )

  relative_summary <- multinma::relative_effects(fit, trt_ref = "Placebo")
  relative_effect <- extract_relative_effect(relative_summary)
  stanfit <- multinma::as.stanfit(fit)
  diagnostic_rows <- as.data.frame(rstan::monitor(stanfit, print = FALSE))
  diagnostic_rows$parameter <- rownames(diagnostic_rows)
  max_rhat <- max(diagnostic_rows$Rhat, na.rm = TRUE)
  min_neff <- min(diagnostic_rows$n_eff, na.rm = TRUE)
  sampler_params <- rstan::get_sampler_params(stanfit, inc_warmup = FALSE)
  divergent_transitions <- sum(unlist(lapply(
    sampler_params,
    function(params) params[, "divergent__"]
  )))
  treedepths <- unlist(lapply(
    sampler_params,
    function(params) params[, "treedepth__"]
  ))

  output <- list(
    schema_version = "multinma_sglt2_binary_nma/v1",
    benchmark_id = "sglt2_hf_primary_log_or",
    source_policy = "clinicaltrials_gov + pubmed_abstract only",
    effect_scale = "log_or",
    model = list(
      engine = "multinma",
      likelihood = "binomial",
      link = "logit",
      consistency = "consistency",
      trt_effects = "fixed",
      reference_treatment = "Placebo",
      prior_intercept = "normal(scale=10)",
      prior_trt = "normal(scale=2.5)",
      chains = parsed$chains,
      iter = parsed$iter,
      warmup = parsed$warmup,
      seed = parsed$seed,
      adapt_delta = parsed$adapt_delta
    ),
    package_versions = list(
      R = as.character(getRversion()),
      multinma = as.character(utils::packageVersion("multinma")),
      rstan = as.character(utils::packageVersion("rstan")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_arms = rows[, required_columns],
    relative_effect = relative_effect,
    diagnostics = list(
      max_rhat = as.numeric(max_rhat),
      min_neff = as.numeric(min_neff),
      divergent_transitions = as.integer(divergent_transitions),
      max_treedepth_observed = as.integer(max(treedepths)),
      n_parameters = as.integer(nrow(diagnostic_rows))
    ),
    limitations = list(
      "Single source-backed two-treatment binary NMA only.",
      "This is a narrow multinma reference candidate, not broad Bayesian NMA parity.",
      "This is not ML-NMR, ranking, inconsistency, or clinical/HTA certification."
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
