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
  out <- list(benchmarks = list(), output = NULL)
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--benchmark", "--output")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--output") {
        out$output <- val
      } else {
        parts <- strsplit(val, "=", fixed = TRUE)[[1]]
        if (length(parts) != 2 || !nzchar(parts[[1]]) || !nzchar(parts[[2]])) {
          stop("--benchmark values must use benchmark_id=effects.csv")
        }
        out$benchmarks[[length(out$benchmarks) + 1]] <- list(
          benchmark_id = parts[[1]],
          effects = parts[[2]]
        )
      }
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$output) || length(out$benchmarks) == 0) {
    stop("Usage: metafor_tau2_crosscheck_survival.R --benchmark <id=effects.csv> [--benchmark ...] --output <output.json>")
  }
  out
}

read_effects <- function(effects_path) {
  rows <- utils::read.csv(effects_path, stringsAsFactors = FALSE)
  required <- c("study_id", "nct_id", "pmid", "estimate", "se", "variance")
  missing <- setdiff(required, names(rows))
  if (length(missing) > 0) {
    stop(paste("Tau2 cross-check effects missing columns:", paste(missing, collapse = ", ")))
  }
  if (nrow(rows) < 2) {
    stop("Tau2 cross-check requires at least two study effects.")
  }
  if (any(!grepl("^NCT[0-9]{8}$", rows$nct_id))) {
    stop("Malformed NCT identifier in tau2 cross-check input.")
  }
  if (any(!grepl("^[0-9]+$", as.character(rows$pmid)))) {
    stop("Malformed PMID in tau2 cross-check input.")
  }
  if (any(!is.finite(rows$estimate)) || any(!is.finite(rows$se)) ||
      any(!is.finite(rows$variance)) || any(rows$se <= 0) || any(rows$variance <= 0)) {
    stop("Tau2 cross-check estimates, SEs, and variances must be finite and positive.")
  }
  rows
}

model_summary <- function(effects, method_key) {
  fit <- metafor::rma.uni(
    yi = effects$estimate,
    vi = effects$variance,
    method = method_key,
    test = "z",
    slab = effects$study_id
  )
  tau2 <- as.numeric(fit$tau2)
  if (!is.finite(tau2)) {
    tau2 <- 0.0
  }
  weights <- 1.0 / (effects$variance + tau2)
  estimate <- as.numeric(fit$b[1, 1])
  q_model <- sum(weights * (effects$estimate - estimate)^2)
  df <- as.integer(nrow(effects) - 1)
  q_p <- if (df <= 0) 1.0 else stats::pchisq(q_model, df = df, lower.tail = FALSE)
  i2 <- if (q_model <= 0 || df <= 0) 0.0 else max(0.0, 100.0 * (q_model - df) / q_model)
  h2 <- if (df <= 0) 1.0 else q_model / df
  list(
    method = method_key,
    k = as.integer(fit$k),
    estimate = estimate,
    se = as.numeric(fit$se[1]),
    ci_low = as.numeric(fit$ci.lb[1]),
    ci_high = as.numeric(fit$ci.ub[1]),
    tau2 = tau2,
    q = as.numeric(q_model),
    df = df,
    q_p_value = as.numeric(q_p),
    i2 = as.numeric(i2),
    h2 = as.numeric(h2)
  )
}

benchmark_result <- function(entry) {
  effects <- read_effects(entry$effects)
  methods <- c("FE", "DL", "PM", "REML")
  list(
    benchmark_id = entry$benchmark_id,
    effects_path = entry$effects,
    evidence_mode = "reported_hr_pubmed_abstract",
    effect_scale = "log_hr",
    study_effects = effects,
    methods = lapply(methods, function(method_key) model_summary(effects, method_key)),
    limitations = list(
      "This is a source-backed tau2 estimator cross-check, not broad optimizer parity.",
      "The benchmark rows are reported HR/CI tokens from allowed public sources, not reconstructed IPD.",
      "This output does not certify clinical, regulatory, or HTA reporting."
    )
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  output <- list(
    schema_version = "metafor_tau2_crosscheck/v1",
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    reference_method = "metafor::rma.uni FE/DL/PM/REML tau2 cross-check",
    package_versions = list(
      R = as.character(getRversion()),
      metafor = as.character(utils::packageVersion("metafor")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    benchmarks = lapply(parsed$benchmarks, benchmark_result)
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
