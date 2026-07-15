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
  out <- list(events = NULL, output = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--events", "--output")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--events") out$events <- val
      if (key == "--output") out$output <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$events) || is.null(out$output)) {
    stop("Usage: pairwise_metafor_meta.R --events <events.csv> --output <output.json>")
  }
  out
}

build_study_effects <- function(events_path) {
  rows <- read.csv(events_path, stringsAsFactors = FALSE)
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

fit_summary <- function(fit) {
  ci <- c(as.numeric(fit$ci.lb), as.numeric(fit$ci.ub))
  list(
    estimate = as.numeric(fit$b[1, 1]),
    se = as.numeric(fit$se),
    ci_low = ci[[1]],
    ci_high = ci[[2]],
    tau2 = as.numeric(fit$tau2),
    q = as.numeric(fit$QE),
    df = as.integer(fit$k - 1)
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- build_study_effects(parsed$events)

  fit_fe <- metafor::rma.uni(yi = effects$yi, vi = effects$vi, method = "FE")
  fit_reml_hksj <- metafor::rma.uni(
    yi = effects$yi,
    vi = effects$vi,
    method = "REML",
    test = "knha"
  )

  output <- list(
    schema_version = "pairwise_metafor_meta/v1",
    effect_scale = "log_or",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      meta = as.character(utils::packageVersion("meta"))
    ),
    study_effects = effects,
    metafor = list(
      fixed_effect = fit_summary(fit_fe),
      reml_hksj = fit_summary(fit_reml_hksj)
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE)
}

main()
