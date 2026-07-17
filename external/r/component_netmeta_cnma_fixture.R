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
  out <- list(effects = NULL, output = NULL, inactive = "Placebo")
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--effects", "--output", "--inactive")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--effects") out$effects <- val
      if (key == "--output") out$output <- val
      if (key == "--inactive") out$inactive <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$effects) || is.null(out$output)) {
    stop("Usage: component_netmeta_cnma_fixture.R --effects <effects.csv> --output <output.json> [--inactive Placebo]")
  }
  out
}

named_effects <- function(names_vec, estimates, ses) {
  out <- list()
  for (idx in seq_along(names_vec)) {
    name <- as.character(names_vec[[idx]])
    if (!is.na(estimates[[idx]])) {
      out[[name]] <- list(
        estimate = as.numeric(estimates[[idx]]),
        se = as.numeric(ses[[idx]])
      )
    }
  }
  out
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  rows <- read.csv(parsed$effects, stringsAsFactors = FALSE)
  required <- c("study_id", "treat1", "treat2", "estimate", "se")
  missing <- setdiff(required, names(rows))
  if (length(missing) > 0) {
    stop(paste("Missing columns:", paste(missing, collapse = ", ")))
  }

  fit <- netmeta::discomb(
    TE = estimate,
    seTE = se,
    treat1 = treat1,
    treat2 = treat2,
    studlab = study_id,
    data = rows,
    sm = "MD",
    inactive = parsed$inactive,
    common = TRUE,
    random = FALSE
  )

  study_effects <- lapply(seq_len(nrow(rows)), function(idx) {
    list(
      study_id = as.character(rows$study_id[[idx]]),
      treat1 = as.character(rows$treat1[[idx]]),
      treat2 = as.character(rows$treat2[[idx]]),
      estimate = as.numeric(rows$estimate[[idx]]),
      se = as.numeric(rows$se[[idx]])
    )
  })

  output <- list(
    schema_version = "component_netmeta_cnma_fixture/v1",
    fixture_id = "netmeta_component_fixture",
    source_policy = "algorithmic_fixture_not_clinical_evidence",
    effect_scale = "mean_difference",
    inactive_treatment = parsed$inactive,
    package_versions = list(
      R = as.character(getRversion()),
      netmeta = as.character(utils::packageVersion("netmeta")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    n_studies = as.integer(fit$k),
    n_contrasts = as.integer(fit$m),
    n_components = as.integer(fit$c),
    study_effects = study_effects,
    component_effects = named_effects(fit$comps, fit$Comp.common, fit$seComp.common),
    treatment_effects = named_effects(fit$trts, fit$TE.common, fit$seTE.common),
    additive = list(
      q = as.numeric(fit$Q.additive),
      df = as.integer(fit$df.Q.additive)
    ),
    standard = list(
      q = as.numeric(fit$Q.standard),
      df = as.integer(fit$df.Q.standard)
    ),
    difference = list(
      q = as.numeric(fit$Q.diff),
      df = as.integer(fit$df.Q.diff)
    ),
    limitations = list(
      "algorithmic fixture only",
      "not source-backed component NMA validation",
      "not broad netmeta CNMA feature parity"
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
