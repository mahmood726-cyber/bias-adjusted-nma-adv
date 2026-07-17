#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("netmeta", quietly = TRUE)) {
    stop("Package 'netmeta' is required. Install with install.packages('netmeta').")
  }
})

parse_args <- function(args) {
  out <- list(effects = NULL, output = NULL, reference = "placebo")
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--effects", "--output", "--reference")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--effects") out$effects <- val
      if (key == "--output") out$output <- val
      if (key == "--reference") out$reference <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$effects) || is.null(out$output)) {
    stop("Usage: ctgov_hr_network_netmeta.R --effects <effects.csv> --output <output.json> [--reference placebo]")
  }
  out
}

required_columns <- c(
  "study_id",
  "nct_id",
  "analysis_treatment",
  "control_treatment",
  "estimate",
  "se",
  "variance"
)

as_effects <- function(nm, reference) {
  effects <- list()
  for (treatment in nm$trts) {
    if (treatment != reference) {
      effects[[treatment]] <- list(
        estimate = as.numeric(nm$TE.common[treatment, reference]),
        se = as.numeric(nm$seTE.common[treatment, reference])
      )
    }
  }
  effects
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- utils::read.csv(parsed$effects, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(effects))
  if (length(missing) > 0) {
    stop(paste("CT.gov HR network effects missing required columns:", paste(missing, collapse = ", ")))
  }

  nm <- netmeta::netmeta(
    TE = estimate,
    seTE = se,
    treat1 = analysis_treatment,
    treat2 = control_treatment,
    studlab = study_id,
    data = effects,
    sm = "HR",
    common = TRUE,
    random = FALSE,
    reference.group = parsed$reference
  )

  output <- list(
    schema_version = "ctgov_hr_network_netmeta/v1",
    benchmark_id = "t2d_mace_ctgov_hr_network",
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    effect_scale = "log_hr",
    reference_treatment = parsed$reference,
    package_versions = list(
      R = as.character(getRversion()),
      netmeta = as.character(utils::packageVersion("netmeta")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects[, required_columns],
    netmeta = list(
      common = as_effects(nm, parsed$reference),
      q = as.numeric(nm$Q),
      df = as.integer(nm$df.Q)
    ),
    limitations = list(
      "ClinicalTrials.gov reported HR and CI values are verified, but no closed-loop inconsistency can be assessed.",
      "The network is a placebo-centered star; this is not broad netmeta parity.",
      "Class labels are analyst-defined treatment groupings and are not clinical superiority claims."
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
