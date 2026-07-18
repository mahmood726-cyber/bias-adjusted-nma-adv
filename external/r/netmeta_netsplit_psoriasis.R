#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("netmeta", quietly = TRUE)) {
    stop("Package 'netmeta' is required. Install with install.packages('netmeta').")
  }
  if (!requireNamespace("meta", quietly = TRUE)) {
    stop("Package 'meta' is required. Install with install.packages('meta').")
  }
})

parse_args <- function(args) {
  out <- list(arms = NULL, output = NULL, benchmark_id = NULL, reference = "placebo")
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--arms", "--output", "--benchmark-id", "--reference")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--arms") out$arms <- val
      if (key == "--output") out$output <- val
      if (key == "--benchmark-id") out$benchmark_id <- val
      if (key == "--reference") out$reference <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$arms) || is.null(out$output) || is.null(out$benchmark_id)) {
    stop("Usage: netmeta_netsplit_psoriasis.R --benchmark-id <id> --arms <arms.csv> --output <output.json> [--reference placebo]")
  }
  out
}

as_nullable_number <- function(value) {
  if (is.null(value) || length(value) == 0 || is.na(value[[1]])) {
    return(NA_real_)
  }
  as.numeric(value[[1]])
}

split_rows <- function(ns) {
  rows <- list()
  for (index in seq_along(ns$comparison)) {
    rows[[index]] <- list(
      comparison = as.character(ns$comparison[[index]]),
      k = as.integer(ns$k[[index]]),
      direct_evidence_proportion = as_nullable_number(ns$prop.common[[index]]),
      nma_estimate = as_nullable_number(ns$common$TE[[index]]),
      nma_se = as_nullable_number(ns$common$seTE[[index]]),
      direct_estimate = as_nullable_number(ns$direct.common$TE[[index]]),
      direct_se = as_nullable_number(ns$direct.common$seTE[[index]]),
      indirect_estimate = as_nullable_number(ns$indirect.common$TE[[index]]),
      indirect_se = as_nullable_number(ns$indirect.common$seTE[[index]]),
      difference = as_nullable_number(ns$compare.common$TE[[index]]),
      difference_se = as_nullable_number(ns$compare.common$seTE[[index]]),
      z_value = as_nullable_number(ns$compare.common$z[[index]]),
      p_value = as_nullable_number(ns$compare.common$p[[index]]),
      estimable = !is.na(ns$compare.common$TE[[index]])
    )
  }
  rows
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  arms <- utils::read.csv(parsed$arms, stringsAsFactors = FALSE)
  required <- c("fixture_id", "study", "treatment", "events", "n")
  missing <- setdiff(required, names(arms))
  if (length(missing) > 0) {
    stop(paste("Netsplit arms missing required columns:", paste(missing, collapse = ", ")))
  }
  if (!all(arms$fixture_id == parsed$benchmark_id)) {
    stop("All arm rows must match --benchmark-id.")
  }

  pw <- meta::pairwise(
    treat = treatment,
    event = events,
    n = n,
    studlab = study,
    data = arms,
    sm = "OR"
  )
  nm <- netmeta::netmeta(
    pw,
    common = TRUE,
    random = FALSE,
    reference.group = parsed$reference
  )
  ns <- netmeta::netsplit(
    nm,
    common = TRUE,
    random = FALSE,
    backtransf = FALSE
  )

  splits <- split_rows(ns)
  output <- list(
    schema_version = "netmeta_netsplit_source/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    effect_scale = "log_or",
    method = "netmeta::netsplit back-calculation SIDE",
    reference_treatment = parsed$reference,
    package_versions = list(
      R = as.character(getRversion()),
      netmeta = as.character(utils::packageVersion("netmeta")),
      meta = as.character(utils::packageVersion("meta")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    arm_rows = arms[, required],
    n_splits = as.integer(length(splits)),
    n_estimable_splits = as.integer(sum(vapply(splits, function(row) row$estimable, logical(1)))),
    splits = splits,
    limitations = list(
      "This is a netmeta back-calculation SIDE diagnostic, not broad node-splitting parity.",
      "The source-backed benchmark is one CT.gov/PubMed psoriasis PASI 90 closed-loop network.",
      "This output does not certify inconsistency diagnostics, clinical reporting, regulatory reporting, or HTA reporting."
    )
  )

  jsonlite::write_json(
    output,
    parsed$output,
    auto_unbox = TRUE,
    pretty = TRUE,
    digits = NA,
    na = "null"
  )
}

main()
