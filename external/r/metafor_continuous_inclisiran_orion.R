#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("metafor", quietly = TRUE)) {
    stop("Package 'metafor' is required. Install with install.packages('metafor').")
  }
  if (!requireNamespace("meta", quietly = TRUE)) {
    stop("Package 'meta' is required. Install with install.packages('meta').")
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
    stop("Usage: metafor_continuous_inclisiran_orion.R --benchmark-id <id> --effects <effects.csv> --output <output.json>")
  }
  out
}

required_columns <- c(
  "study_id", "study_label", "trial", "nct_id", "pmid", "source_type",
  "outcome_id", "outcome_label", "estimand", "comparison", "treatment",
  "comparator", "estimate", "se", "variance", "ci_low", "ci_high",
  "ci_level", "statistical_method", "param_type"
)

fit_summary <- function(fit, method) {
  tau2 <- as.numeric(fit$tau2)
  if (!is.finite(tau2)) {
    tau2 <- 0.0
  }
  weights <- 1.0 / (fit$vi + tau2)
  estimate <- as.numeric(fit$b[1, 1])
  q <- sum(weights * (fit$yi - estimate)^2)
  qe <- as.numeric(fit$QE)
  df <- as.integer(fit$k - 1)
  i2 <- if (q <= 0 || df <= 0) 0 else max(0, 100 * (q - df) / q)
  h2 <- if (df <= 0) 1 else q / df
  list(
    method = method,
    k = as.integer(fit$k),
    estimate = estimate,
    se = as.numeric(fit$se),
    ci_low = as.numeric(fit$ci.lb),
    ci_high = as.numeric(fit$ci.ub),
    tau2 = tau2,
    q = q,
    q_p_value = if (df <= 0) 1 else as.numeric(stats::pchisq(q, df = df, lower.tail = FALSE)),
    qe = qe,
    qe_p_value = if (df <= 0) 1 else as.numeric(stats::pchisq(qe, df = df, lower.tail = FALSE)),
    df = df,
    i2 = as.numeric(i2),
    h2 = as.numeric(h2)
  )
}

meta_summary <- function(fit) {
  list(
    method = "meta::metagen REML",
    k = as.integer(fit$k),
    common_estimate = as.numeric(fit$TE.common),
    common_se = as.numeric(fit$seTE.common),
    common_ci_low = as.numeric(fit$lower.common),
    common_ci_high = as.numeric(fit$upper.common),
    random_estimate = as.numeric(fit$TE.random),
    random_se = as.numeric(fit$seTE.random),
    random_ci_low = as.numeric(fit$lower.random),
    random_ci_high = as.numeric(fit$upper.random),
    tau2 = as.numeric(fit$tau2),
    q = as.numeric(fit$Q),
    df = as.integer(fit$df.Q),
    i2 = as.numeric(fit$I2) * 100.0
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- utils::read.csv(parsed$effects, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(effects))
  if (length(missing) > 0) {
    stop(paste("Continuous effects missing required columns:", paste(missing, collapse = ", ")))
  }
  if (any(!grepl("^NCT[0-9]{8}$", effects$nct_id))) {
    stop("Malformed NCT identifier in continuous effects input.")
  }
  if (any(!grepl("^[0-9]+$", effects$pmid))) {
    stop("Malformed PMID in continuous effects input.")
  }
  if (any(effects$source_type != "clinicaltrials_gov")) {
    stop("Continuous effects must come from clinicaltrials_gov source rows.")
  }
  if (length(unique(effects$comparison)) != 1 || unique(effects$comparison) != "inclisiran_vs_placebo") {
    stop("Unexpected continuous comparison.")
  }
  if (any(effects$param_type != "Mean Difference (Final Values)")) {
    stop("Unexpected CT.gov continuous analysis parameter type.")
  }
  if (any(effects$se <= 0) || any(effects$variance <= 0)) {
    stop("SE and variance must be positive.")
  }
  if (any(abs(effects$se^2 - effects$variance) > 1e-6)) {
    stop("SE and variance columns disagree.")
  }

  fit_fe <- metafor::rma.uni(yi = effects$estimate, vi = effects$variance, method = "FE")
  fit_dl <- metafor::rma.uni(yi = effects$estimate, vi = effects$variance, method = "DL")
  fit_pm <- metafor::rma.uni(yi = effects$estimate, vi = effects$variance, method = "PM")
  fit_reml <- metafor::rma.uni(yi = effects$estimate, vi = effects$variance, method = "REML")
  meta_reml <- meta::metagen(
    TE = effects$estimate,
    seTE = effects$se,
    studlab = effects$study_label,
    sm = "MD",
    method.tau = "REML",
    common = TRUE,
    random = TRUE
  )

  output <- list(
    schema_version = "metafor_continuous_source/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract only",
    effect_scale = "mean_difference_percentage_points",
    contrast = "inclisiran_vs_placebo",
    reference_method = "metafor::rma.uni and meta::metagen continuous mean-difference",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      meta = as.character(utils::packageVersion("meta")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects,
    metafor = list(
      fixed_effect = fit_summary(fit_fe, "FE"),
      dl_random_effect = fit_summary(fit_dl, "DL"),
      pm_random_effect = fit_summary(fit_pm, "PM"),
      reml_random_effect = fit_summary(fit_reml, "REML")
    ),
    meta = meta_summary(meta_reml),
    limitations = list(
      "CT.gov adjusted least-squares mean differences are used as reported; raw continuous IPD is not available.",
      "Different ORION trial populations are pooled only for software validation, not for clinical guidance.",
      "This validates a second continuous mean-difference reference candidate only; it is not broad pairwise feature parity.",
      "This does not certify clinical reporting, regulatory reporting, HTA reporting, or tier-one superiority."
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
