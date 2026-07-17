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
    stop("Usage: survival_hr_metafor_pairwise.R --benchmark-id <id> --effects <effects.csv> --output <output.json>")
  }
  out
}

required_columns <- c("study_id", "nct_id", "pmid", "estimate", "se", "variance")

fit_summary <- function(fit) {
  list(
    estimate = as.numeric(fit$b[1, 1]),
    se = as.numeric(fit$se),
    ci_low = as.numeric(fit$ci.lb),
    ci_high = as.numeric(fit$ci.ub),
    tau2 = as.numeric(fit$tau2),
    q = as.numeric(fit$QE),
    df = as.integer(fit$k - 1)
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- utils::read.csv(parsed$effects, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(effects))
  if (length(missing) > 0) {
    stop(paste("Survival HR effects missing required columns:", paste(missing, collapse = ", ")))
  }

  fit_fe <- metafor::rma.uni(
    yi = effects$estimate,
    vi = effects$variance,
    method = "FE"
  )

  output <- list(
    schema_version = "survival_hr_metafor_pairwise/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    effect_scale = "log_hr",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects[, required_columns],
    metafor = list(
      fixed_effect = fit_summary(fit_fe)
    ),
    limitations = list(
      "Reported PubMed abstract HR tokens are verified, but Kaplan-Meier curves are not digitized.",
      "This is a pairwise class meta-analysis, not a multi-treatment survival NMA.",
      "This does not establish survival NMA parity or clinical certification."
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
