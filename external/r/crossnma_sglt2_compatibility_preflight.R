#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("crossnma", quietly = TRUE)) {
    stop("Package 'crossnma' is required. Install with install.packages('crossnma').")
  }
  if (!requireNamespace("rjags", quietly = TRUE)) {
    stop("Package 'rjags' and a JAGS runtime are required for crossnma.")
  }
})

parse_args <- function(args) {
  out <- list(input = NULL, output = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--input", "--output")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--input") out$input <- val
      if (key == "--output") out$output <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$input) || is.null(out$output)) {
    stop("Usage: crossnma_sglt2_compatibility_preflight.R --input <effects.csv> --output <output.json>")
  }
  out
}

required_columns <- c(
  "study_id", "trial", "design", "nct_id", "pmid", "outcome_id", "outcome_label",
  "target_population", "active_treatment", "control_treatment", "comparator_class",
  "effect_scale", "reported_hr", "ci_lower", "ci_upper", "estimate", "se"
)

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- utils::read.csv(parsed$input, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(effects))
  if (length(missing) > 0) {
    stop(paste("crossnma compatibility input missing required columns:", paste(missing, collapse = ", ")))
  }
  if (!all(effects$design %in% c("rct", "nrs"))) {
    stop("crossnma compatibility input design must be rct or nrs.")
  }
  if (!all(grepl("^NCT[0-9]{8}$", effects$nct_id))) {
    stop("Malformed NCT identifier in crossnma compatibility input.")
  }
  if (!all(grepl("^[0-9]{1,9}$", as.character(effects$pmid)))) {
    stop("Malformed PMID in crossnma compatibility input.")
  }
  if (any(effects$se <= 0)) {
    stop("crossnma compatibility input SEs must be positive.")
  }

  supported_summary_measures <- c("OR", "RR", "MD", "SMD")
  observed_effect_scales <- sort(unique(effects$effect_scale))
  effect_scale_supported <- all(observed_effect_scales %in% c("OR", "RR", "MD", "SMD"))
  mismatch_fields <- list()
  for (field in c("outcome_id", "target_population", "control_treatment", "comparator_class")) {
    values <- sort(unique(as.character(effects[[field]])))
    if (length(values) > 1) {
      mismatch_fields[[field]] <- values
    }
  }

  blocking_reasons <- c()
  if (!effect_scale_supported) {
    blocking_reasons <- c(
      blocking_reasons,
      "effect_scale_log_hr_not_supported_by_crossnma_model"
    )
  }
  blocking_reasons <- c(
    blocking_reasons,
    "missing_arm_level_binary_or_continuous_outcomes"
  )
  if (length(mismatch_fields) > 0) {
    blocking_reasons <- c(
      blocking_reasons,
      paste0("estimand_mismatch:", paste(names(mismatch_fields), collapse = ","))
    )
  }

  output <- list(
    schema_version = "crossnma_sglt2_compatibility_preflight/v1",
    benchmark_id = "sglt2_rct_nrs_cross_design",
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    evidence_mode = "reported_hr_pubmed_abstract_cross_design",
    effect_scale = "log_hr",
    package_versions = list(
      R = as.character(getRversion()),
      crossnma = as.character(utils::packageVersion("crossnma")),
      rjags = as.character(utils::packageVersion("rjags")),
      JAGS = as.character(rjags::jags.version()),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    crossnma_api = list(
      package_loaded = TRUE,
      supported_summary_measures = supported_summary_measures,
      requires_arm_level_binary_or_continuous_data = TRUE,
      crossnma_model_attempted = FALSE
    ),
    compatibility = list(
      status = "blocked_incompatible_source_fixture",
      combined_borrowing_allowed = FALSE,
      effect_scale_supported = effect_scale_supported,
      observed_effect_scales = observed_effect_scales,
      observed_designs = sort(unique(effects$design)),
      mismatched_fields = mismatch_fields,
      blocking_reasons = blocking_reasons,
      certification_effect = "none"
    ),
    study_effects = effects,
    limitations = list(
      "This is a crossnma compatibility preflight, not crossnma reference matching.",
      "The source-backed SGLT2 RCT/NRS rows are reported hazard ratios, not crossnma arm-level OR/RR/MD/SMD data.",
      "RCT and NRS rows have mismatched outcome, population, comparator, and control definitions.",
      "No crossnma model is run because forcing this fixture into the package would be statistically invalid.",
      "This is not clinical, regulatory, HTA, or production certification."
    )
  )
  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
