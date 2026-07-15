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
  out <- list(arms = NULL, output = NULL, reference = "A")
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--arms", "--output", "--reference")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--arms") out$arms <- val
      if (key == "--output") out$output <- val
      if (key == "--reference") out$reference <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$arms) || is.null(out$output)) {
    stop("Usage: multiarm_netmeta_fixture.R --arms <arms.csv> --output <output.json> [--reference A]")
  }
  out
}

as_effects <- function(nm, reference, model) {
  te <- if (model == "common") nm$TE.common else nm$TE.random
  se <- if (model == "common") nm$seTE.common else nm$seTE.random
  effects <- list()
  for (treatment in nm$trts) {
    if (treatment != reference) {
      effects[[treatment]] <- list(
        estimate = as.numeric(te[treatment, reference]),
        se = as.numeric(se[treatment, reference])
      )
    }
  }
  effects
}

fit_fixture <- function(rows, fixture_id, reference) {
  fixture_rows <- rows[rows$fixture_id == fixture_id, , drop = FALSE]
  pw <- meta::pairwise(
    treat = treatment,
    event = events,
    n = n,
    studlab = study,
    data = fixture_rows,
    sm = "OR"
  )
  nm <- netmeta::netmeta(pw, common = TRUE, random = TRUE, reference.group = reference)
  list(
    fixture_id = fixture_id,
    reference_treatment = reference,
    treatments = as.character(nm$trts),
    tau2 = as.numeric(nm$tau2),
    q = as.numeric(nm$Q),
    df = as.integer(nm$df.Q),
    common = as_effects(nm, reference, "common"),
    random = as_effects(nm, reference, "random")
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  rows <- read.csv(parsed$arms, stringsAsFactors = FALSE)
  required <- c("fixture_id", "study", "treatment", "events", "n")
  missing <- setdiff(required, names(rows))
  if (length(missing) > 0) {
    stop(paste("Missing columns:", paste(missing, collapse = ", ")))
  }

  fixtures <- lapply(sort(unique(rows$fixture_id)), function(fixture_id) {
    fit_fixture(rows, fixture_id, parsed$reference)
  })
  output <- list(
    schema_version = "multiarm_netmeta_fixture/v1",
    effect_scale = "log_or",
    package_versions = list(
      R = as.character(getRversion()),
      netmeta = as.character(utils::packageVersion("netmeta")),
      meta = as.character(utils::packageVersion("meta"))
    ),
    fixtures = fixtures
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
