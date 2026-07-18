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
    stop("Usage: publication_bias_metafor_trimfill_glp1.R --benchmark-id <id> --effects <effects.csv> --output <output.json>")
  }
  out
}

read_effects <- function(effects_path) {
  rows <- utils::read.csv(effects_path, stringsAsFactors = FALSE)
  required <- c("study_id", "nct_id", "pmid", "estimate", "se", "variance")
  missing <- setdiff(required, names(rows))
  if (length(missing) > 0) {
    stop(paste("Missing columns:", paste(missing, collapse = ", ")))
  }
  if (any(!grepl("^NCT[0-9]{8}$", rows$nct_id))) {
    stop("Malformed NCT identifier in trim-and-fill input.")
  }
  if (any(!grepl("^[0-9]+$", as.character(rows$pmid)))) {
    stop("Malformed PMID in trim-and-fill input.")
  }
  if (any(!is.finite(rows$estimate)) || any(!is.finite(rows$se)) ||
      any(!is.finite(rows$variance)) || any(rows$se <= 0) || any(rows$variance <= 0)) {
    stop("Trim-and-fill input estimates, SEs, and variances must be finite and positive.")
  }
  rows
}

model_summary <- function(model) {
  list(
    k = as.integer(model$k),
    estimate = as.numeric(model$b[1, 1]),
    se = as.numeric(model$se[1]),
    ci_low = as.numeric(model$ci.lb[1]),
    ci_high = as.numeric(model$ci.ub[1]),
    q = as.numeric(model$QE),
    df = as.integer(model$k - model$p),
    q_p_value = as.numeric(model$QEp),
    i2 = as.numeric(model$I2),
    h2 = as.numeric(model$H2),
    tau2 = as.numeric(model$tau2)
  )
}

filled_rows <- function(trimmed) {
  out <- list()
  fill <- as.logical(trimmed$fill)
  for (idx in seq_along(fill)) {
    out[[idx]] <- list(
      row_index_one_based = as.integer(idx),
      label = as.character(trimmed$slab[[idx]]),
      imputed = fill[[idx]],
      yi = as.numeric(trimmed$yi.f[[idx]]),
      vi = as.numeric(trimmed$vi.f[[idx]]),
      sei = sqrt(as.numeric(trimmed$vi.f[[idx]]))
    )
  }
  out
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  effects <- read_effects(parsed$effects)
  fit <- metafor::rma.uni(
    yi = effects$estimate,
    vi = effects$variance,
    method = "FE",
    slab = effects$study_id
  )
  trimmed <- metafor::trimfill(fit)

  output <- list(
    schema_version = "publication_bias_metafor_trimfill/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    evidence_mode = "reported_hr_pubmed_abstract",
    effect_scale = "log_hr",
    method = "metafor::trimfill rma.uni fixed-effect",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    study_effects = effects,
    observed = model_summary(fit),
    trimfill = c(
      list(
        k0 = as.integer(trimmed$k0),
        se_k0 = as.numeric(trimmed$se.k0),
        side = as.character(trimmed$side),
        estimator = as.character(trimmed$k0.est),
        p_k0 = as.numeric(trimmed$p.k0)
      ),
      model_summary(trimmed)
    ),
    filled_rows = filled_rows(trimmed),
    limitations = list(
      "Trim-and-fill is a sensitivity analysis, not proof or correction of publication bias.",
      "This output validates one source-backed GLP-1 MACE reported-HR benchmark, not broad publication-bias parity.",
      "This output does not certify clinical, regulatory, or HTA reporting."
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
