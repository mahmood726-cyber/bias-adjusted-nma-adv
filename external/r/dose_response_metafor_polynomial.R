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
  out <- list(effects = NULL, output = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--effects", "--output")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--effects") out$effects <- val
      if (key == "--output") out$output <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$effects) || is.null(out$output)) {
    stop("Usage: dose_response_metafor_polynomial.R --effects <effects.csv> --output <output.json>")
  }
  out
}

required_columns <- c("study_id", "nct_id", "pmid", "dose", "estimate", "se", "variance")

fit_summary <- function(fit, effects, mods) {
  coefficients <- as.numeric(fit$b[, 1])
  fitted <- as.numeric(mods %*% coefficients)
  residual_q <- sum((effects$estimate - fitted)^2 / effects$variance)
  list(
    coefficients = coefficients,
    coefficient_ses = as.numeric(fit$se),
    q = as.numeric(residual_q),
    metafor_qe = as.numeric(fit$QE),
    df = as.integer(fit$k - fit$p)
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- utils::read.csv(parsed$effects, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(effects))
  if (length(missing) > 0) {
    stop(paste("Dose-response effects missing required columns:", paste(missing, collapse = ", ")))
  }
  effects$dose2 <- effects$dose^2
  linear_mods <- stats::model.matrix(~ dose - 1, data = effects)
  quadratic_mods <- stats::model.matrix(~ dose + dose2 - 1, data = effects)

  fit_linear <- metafor::rma.uni(
    yi = effects$estimate,
    vi = effects$variance,
    mods = linear_mods,
    method = "FE",
    intercept = FALSE
  )
  fit_quadratic <- metafor::rma.uni(
    yi = effects$estimate,
    vi = effects$variance,
    mods = quadratic_mods,
    method = "FE",
    intercept = FALSE
  )

  output <- list(
    schema_version = "dose_response_metafor_polynomial/v1",
    benchmark_id = "semaglutide_obesity_dose_response",
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    effect_scale = "percentage_point_change_vs_placebo",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects[, required_columns],
    metafor = list(
      weighted_linear = fit_summary(fit_linear, effects, linear_mods),
      weighted_quadratic = fit_summary(fit_quadratic, effects, quadratic_mods)
    ),
    limitations = list(
      "Single source-backed dose-ranging trial; not dose-response NMA parity.",
      "Shared placebo covariance follows the local benchmark limitation.",
      "This is not MBNMAdose reference matching."
    )
  )

  jsonlite::write_json(
    output,
    parsed$output,
    auto_unbox = TRUE,
    pretty = TRUE,
    digits = NA
  )
}

main()
