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
    stop("Usage: publication_bias_metafor_regtest.R --benchmark-id <id> --effects <effects.csv> --output <output.json>")
  }
  out
}

required_columns <- c("study_id", "nct_id", "estimate", "se", "variance")

as_scalar <- function(value) {
  if (is.null(value) || length(value) == 0) {
    return(NA_real_)
  }
  as.numeric(value[[1]])
}

as_string_scalar <- function(value, default) {
  if (is.null(value) || length(value) == 0) {
    return(default)
  }
  as.character(value[[1]])
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- utils::read.csv(parsed$effects, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(effects))
  if (length(missing) > 0) {
    stop(paste("Publication-bias effects missing required columns:", paste(missing, collapse = ", ")))
  }
  if (nrow(effects) < 10) {
    stop("Publication-bias regtest reference requires at least 10 study effects.")
  }

  egger <- metafor::regtest(
    x = effects$estimate,
    sei = effects$se,
    model = "lm",
    predictor = "sei"
  )

  output <- list(
    schema_version = "publication_bias_metafor_regtest/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov reported result rows only",
    effect_scale = "log_hr",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects[, required_columns],
    metafor = list(
      egger_lm_sei = list(
        model = "lm",
        predictor = "sei",
        k = as.integer(nrow(effects)),
        statistic_name = as_string_scalar(egger$zname, "t"),
        statistic = as_scalar(egger$zval),
        p_value = as_scalar(egger$pval),
        degrees_of_freedom = as.integer(egger$dfs),
        intercept = as_scalar(stats::coef(egger$fit)[[1]]),
        slope = as_scalar(stats::coef(egger$fit)[[2]]),
        intercept_se = as_scalar(sqrt(diag(stats::vcov(egger$fit)))[[1]]),
        slope_se = as_scalar(sqrt(diag(stats::vcov(egger$fit)))[[2]])
      )
    ),
    limitations = list(
      "This is a small-study-effect diagnostic, not proof of publication bias.",
      "The source-backed benchmark is a CT.gov reported-HR star network, not a full publication-bias workflow.",
      "This output is a narrow evidence_candidate and does not establish broad metafor publication-bias parity."
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
