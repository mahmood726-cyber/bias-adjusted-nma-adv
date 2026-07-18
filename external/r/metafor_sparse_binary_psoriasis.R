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
    stop("Usage: metafor_sparse_binary_psoriasis.R --benchmark-id <id> --events <events.csv> --output <output.json>")
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
      trial = active$trial,
      nct_id = active$nct_id,
      pmid = active$pmid,
      outcome_id = active$outcome_id,
      outcome_label = active$outcome_label,
      treatment = active$treatment,
      comparator = control$treatment,
      active_events = as.integer(active$events),
      active_n = as.integer(active$n),
      control_events = as.integer(control$events),
      control_n = as.integer(control$n),
      yi = as.numeric(yi),
      vi = as.numeric(vi),
      sei = sqrt(as.numeric(vi)),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

fit_summary <- function(fit, method) {
  q <- as.numeric(fit$QE)
  df <- as.integer(fit$k - 1)
  i2 <- if (q <= 0 || df <= 0) 0 else max(0, 100 * (q - df) / q)
  h2 <- if (df <= 0) 1 else q / df
  list(
    method = method,
    k = as.integer(fit$k),
    estimate = as.numeric(fit$b[1, 1]),
    se = as.numeric(fit$se),
    ci_low = as.numeric(fit$ci.lb),
    ci_high = as.numeric(fit$ci.ub),
    tau2 = as.numeric(fit$tau2),
    q = q,
    q_p_value = if (df <= 0) 1 else as.numeric(stats::pchisq(q, df = df, lower.tail = FALSE)),
    df = df,
    i2 = as.numeric(i2),
    h2 = as.numeric(h2)
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- build_study_effects(parsed$events)

  fit_fe <- metafor::rma.uni(yi = effects$yi, vi = effects$vi, method = "FE")
  fit_dl <- metafor::rma.uni(yi = effects$yi, vi = effects$vi, method = "DL")
  fit_reml <- metafor::rma.uni(yi = effects$yi, vi = effects$vi, method = "REML")

  output <- list(
    schema_version = "metafor_sparse_binary_source/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    effect_scale = "log_or",
    contrast = "etanercept_vs_placebo",
    zero_cell_policy = "no_continuity_correction_zero_cells_fail_closed",
    reference_method = "metafor::rma.uni sparse binary count-derived log-OR",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      meta = as.character(utils::packageVersion("meta")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects,
    metafor = list(
      fixed_effect = fit_summary(fit_fe, "FE"),
      dl_random_effect = fit_summary(fit_dl, "DL"),
      reml_random_effect = fit_summary(fit_reml, "REML")
    ),
    limitations = list(
      "This validates one CT.gov/PubMed low-control-event binary pairwise contrast only.",
      "It is not Mantel-Haenszel parity, zero-event parity, broad sparse-event parity, clinical guidance, regulatory reporting, or HTA certification.",
      "Zero cells remain fail-closed and require an explicit correction policy before fitting."
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
