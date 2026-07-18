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
  out <- list(events = NULL, output = NULL, benchmark_id = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--events", "--output", "--benchmark-id")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--events") out$events <- val
      if (key == "--output") out$output <- val
      if (key == "--benchmark-id") out$benchmark_id <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$events) || is.null(out$output) || is.null(out$benchmark_id)) {
    stop("Usage: metafor_gosh_sglt2.R --benchmark-id <id> --events <events.csv> --output <output.json>")
  }
  out
}

build_study_effects <- function(events_path) {
  rows <- utils::read.csv(events_path, stringsAsFactors = FALSE)
  required <- c(
    "study_id", "trial", "nct_id", "pmid", "outcome_id", "outcome_label",
    "arm_role", "treatment", "events", "n"
  )
  missing <- setdiff(required, names(rows))
  if (length(missing) > 0) {
    stop(paste("Missing columns:", paste(missing, collapse = ", ")))
  }

  out <- list()
  for (study_id in sort(unique(rows$study_id))) {
    study <- rows[rows$study_id == study_id, , drop = FALSE]
    active <- study[study$arm_role == "active", , drop = FALSE]
    control <- study[study$arm_role == "control", , drop = FALSE]
    if (nrow(active) != 1 || nrow(control) != 1) {
      stop(paste(study_id, "requires exactly one active and one control arm."))
    }
    cells <- c(
      active$events,
      active$n - active$events,
      control$events,
      control$n - control$events
    )
    if (any(cells <= 0)) {
      stop(paste(study_id, "has a zero cell; explicit correction policy required."))
    }
    yi <- log((active$events / (active$n - active$events)) /
      (control$events / (control$n - control$events)))
    vi <- 1 / active$events + 1 / (active$n - active$events) +
      1 / control$events + 1 / (control$n - control$events)
    out[[study_id]] <- data.frame(
      study_id = study_id,
      nct_id = active$nct_id,
      pmid = active$pmid,
      yi = as.numeric(yi),
      vi = as.numeric(vi),
      sei = sqrt(as.numeric(vi)),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

subset_rows <- function(gosh_result, study_ids) {
  rows <- list()
  for (index in seq_len(nrow(gosh_result$res))) {
    included <- which(as.logical(gosh_result$incl[index, ]))
    rows[[index]] <- list(
      subset_id = as.integer(index),
      subset_indices_zero_based = as.integer(included - 1),
      subset_study_ids = as.character(study_ids[included]),
      k = as.integer(gosh_result$res$k[[index]]),
      estimate = as.numeric(gosh_result$res$estimate[[index]]),
      q = as.numeric(gosh_result$res$QE[[index]]),
      i2 = as.numeric(gosh_result$res$I2[[index]]),
      h2 = as.numeric(gosh_result$res$H2[[index]]),
      tau2 = as.numeric(gosh_result$res$tau2[[index]]),
      tau = as.numeric(gosh_result$res$tau[[index]])
    )
  }
  rows
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- build_study_effects(parsed$events)

  fit <- metafor::rma.uni(
    yi = effects$yi,
    vi = effects$vi,
    method = "FE",
    slab = effects$study_id
  )
  gosh_result <- metafor::gosh(fit)
  subsets <- subset_rows(gosh_result, effects$study_id)

  output <- list(
    schema_version = "metafor_gosh_source/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract only",
    effect_scale = "log_or",
    method = "metafor::gosh rma.uni fixed-effect",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects,
    n_studies = as.integer(nrow(effects)),
    n_subsets = as.integer(length(subsets)),
    validated_min_subset_size = as.integer(1),
    subsets = subsets,
    limitations = list(
      "This is a GOSH-style subset diagnostic, not an outlier-removal rule or clinical conclusion.",
      "The benchmark is one source-backed SGLT2i heart-failure pairwise meta-analysis, not broad metafor GOSH visualization parity.",
      "This output does not certify publication-bias, fraud-containment, clinical, regulatory, or HTA reporting."
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
