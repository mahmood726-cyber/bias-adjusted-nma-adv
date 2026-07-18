#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("metafor", quietly = TRUE)) {
    stop("Package 'metafor' is required. Install with install.packages('metafor').")
  }
})

parse_args <- function(args) {
  out <- list(effects = NULL, output = NULL, benchmark_id = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--effects", "--output", "--benchmark-id")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--effects") out$effects <- val
      if (key == "--output") out$output <- val
      if (key == "--benchmark-id") out$benchmark_id <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$effects) || is.null(out$output) || is.null(out$benchmark_id)) {
    stop("Usage: metafor_prediction_interval_breast.R --benchmark-id <id> --effects <effects.csv> --output <output.json>")
  }
  out
}

required_columns <- c("study_id", "nct_id", "pmid", "estimate", "se", "variance")

fit_summary <- function(fit, method, effects, base_se = NULL) {
  tau2 <- as.numeric(fit$tau2)
  if (!is.finite(tau2)) {
    tau2 <- 0.0
  }
  weights <- 1.0 / (effects$variance + tau2)
  estimate <- as.numeric(fit$b[1, 1])
  q <- sum(weights * (effects$estimate - estimate)^2)
  df <- as.integer(fit$k - 1)
  i2 <- if (q <= 0 || df <= 0) 0 else max(0, 100 * (q - df) / q)
  h2 <- if (df <= 0) 1 else q / df
  q_factor <- 1
  if (!is.null(base_se)) {
    q_factor <- (as.numeric(fit$se) / as.numeric(base_se))^2
  }
  list(
    method = method,
    k = as.integer(fit$k),
    estimate = estimate,
    se = as.numeric(fit$se),
    ci_low = as.numeric(fit$ci.lb),
    ci_high = as.numeric(fit$ci.ub),
    tau2 = tau2,
    q = as.numeric(q),
    q_p_value = if (df <= 0) 1 else as.numeric(stats::pchisq(q, df = df, lower.tail = FALSE)),
    df = df,
    i2 = as.numeric(i2),
    h2 = as.numeric(h2),
    hksj_q_factor = as.numeric(q_factor)
  )
}

prediction_summary <- function(pred) {
  list(
    pred = as.numeric(pred$pred),
    se = as.numeric(pred$se),
    ci_low = as.numeric(pred$ci.lb),
    ci_high = as.numeric(pred$ci.ub),
    pi_low = as.numeric(pred$pi.lb),
    pi_high = as.numeric(pred$pi.ub),
    pi_se = as.numeric(pred$pi.se),
    pi_dist = as.character(pred$pi.dist)
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- utils::read.csv(parsed$effects, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(effects))
  if (length(missing) > 0) {
    stop(paste("Prediction-interval effects missing required columns:", paste(missing, collapse = ", ")))
  }

  fit_reml <- metafor::rma.uni(
    yi = effects$estimate,
    vi = effects$variance,
    method = "REML"
  )
  fit_knha <- metafor::rma.uni(
    yi = effects$estimate,
    vi = effects$variance,
    method = "REML",
    test = "knha"
  )
  pred_reml <- metafor::predict.rma(fit_reml)
  pred_knha <- metafor::predict.rma(fit_knha)

  output <- list(
    schema_version = "metafor_prediction_interval_source/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    effect_scale = "log_hr",
    effects_path = parsed$effects,
    reference_method = "metafor::predict REML/KNHA reported-HR prediction interval",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects[, required_columns],
    metafor = list(
      reml_default = fit_summary(fit_reml, "REML", effects),
      reml_knha_unfloored = fit_summary(fit_knha, "REML", effects, base_se = fit_reml$se),
      prediction_default = prediction_summary(pred_reml),
      prediction_knha_unfloored = prediction_summary(pred_knha)
    ),
    limitations = list(
      "Reported PubMed abstract HR tokens are verified, but Kaplan-Meier curves are not digitized.",
      "This validates one reported-HR pairwise prediction-interval convention only; it is not survival NMA parity.",
      "metafor KNHA prediction intervals use the unfloored q factor; the Python artifact applies a conservative HKSJ floor.",
      "This does not certify clinical reporting, regulatory reporting, HTA reporting, or tier-one superiority."
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
