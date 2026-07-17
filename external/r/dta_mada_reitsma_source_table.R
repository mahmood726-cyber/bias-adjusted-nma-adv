#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("mada", quietly = TRUE)) {
    stop("Package 'mada' is required. Install with install.packages('mada').")
  }
})

parse_args <- function(args) {
  out <- list(input = NULL, output = NULL, benchmark_id = "midkine_elisa_cancer_dta")
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--input", "--output", "--benchmark-id")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--input") out$input <- val
      if (key == "--output") out$output <- val
      if (key == "--benchmark-id") out$benchmark_id <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$input) || is.null(out$output)) {
    stop("Usage: dta_mada_reitsma_source_table.R --input <input.csv> --output <output.json> [--benchmark-id id]")
  }
  out
}

required_columns <- c(
  "study_id", "citation", "country", "cancer", "index_test",
  "reference_standard", "threshold", "tp", "fp", "fn", "tn",
  "source_type", "source_doi", "table_doi", "table_id", "row_label"
)

extract_row <- function(coefs, pattern) {
  hits <- grep(pattern, rownames(coefs), value = TRUE)
  if (length(hits) == 0) {
    stop(paste("Could not find coefficient row matching", pattern))
  }
  coefs[hits[[1]], , drop = TRUE]
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  rows <- utils::read.csv(
    parsed$input,
    stringsAsFactors = FALSE,
    na.strings = character(0),
    check.names = FALSE
  )
  missing <- setdiff(required_columns, names(rows))
  if (length(missing) > 0) {
    stop(paste("DTA source table missing required columns:", paste(missing, collapse = ", ")))
  }
  if (nrow(rows) < 5) {
    stop("DTA source table requires at least five studies for bivariate validation.")
  }
  if (any(rows$source_type != "open_access_paper")) {
    stop("DTA source table rows must use source_type=open_access_paper.")
  }

  counts <- data.frame(
    TP = as.numeric(rows$tp),
    FN = as.numeric(rows$fn),
    FP = as.numeric(rows$fp),
    TN = as.numeric(rows$tn)
  )
  fit <- mada::reitsma(
    data = counts,
    correction = 0.5,
    correction.control = "all",
    method = "reml"
  )
  sm <- summary(fit)
  coefs <- sm$coefficients
  sens_row <- extract_row(coefs, "^sensitivity$")
  fpr_row <- extract_row(coefs, "^false pos\\. rate$")
  logit_sens_row <- extract_row(coefs, "^tsens")
  logit_fpr_row <- extract_row(coefs, "^tfpr")
  psi <- sm$Psi
  vc <- sm$vcov
  auc <- sm$AUC

  tau2_sens <- as.numeric(psi[1, 1])
  tau2_fpr <- as.numeric(psi[2, 2])
  cov_sens_fpr <- as.numeric(psi[1, 2])
  rho <- cov_sens_fpr / sqrt(tau2_sens * tau2_fpr)
  pooled_fpr <- as.numeric(fpr_row[["Estimate"]])
  fpr_ci_low <- as.numeric(fpr_row[["95%ci.lb"]])
  fpr_ci_high <- as.numeric(fpr_row[["95%ci.ub"]])
  pooled_spec <- 1 - pooled_fpr

  study_effects <- lapply(seq_len(nrow(rows)), function(idx) {
    list(
      study_id = rows$study_id[[idx]],
      citation = rows$citation[[idx]],
      country = rows$country[[idx]],
      cancer = rows$cancer[[idx]],
      index_test = rows$index_test[[idx]],
      reference_standard = rows$reference_standard[[idx]],
      threshold = rows$threshold[[idx]],
      tp = as.numeric(rows$tp[[idx]]),
      fp = as.numeric(rows$fp[[idx]]),
      fn = as.numeric(rows$fn[[idx]]),
      tn = as.numeric(rows$tn[[idx]]),
      source_type = rows$source_type[[idx]],
      source_doi = rows$source_doi[[idx]],
      table_doi = rows$table_doi[[idx]],
      table_id = rows$table_id[[idx]],
      row_label = rows$row_label[[idx]]
    )
  })

  output <- list(
    schema_version = "dta_mada_reitsma_source_table/v1",
    benchmark_id = parsed$benchmark_id,
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    evidence_mode = "open_access_jats_table_2x2",
    effect_scale = "logit_sensitivity_and_logit_false_positive_rate",
    certification_effect = "none",
    package_versions = list(
      R = as.character(getRversion()),
      mada = as.character(utils::packageVersion("mada")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    n_studies = nrow(rows),
    continuity_correction = 0.5,
    correction_control = "all",
    method = "reml",
    converged = isTRUE(sm$converged),
    logLik = as.numeric(sm$logLik),
    AIC = as.numeric(sm$AIC),
    BIC = as.numeric(sm$BIC),
    study_effects = study_effects,
    summary = list(
      logit_sensitivity = as.numeric(logit_sens_row[["Estimate"]]),
      logit_fpr = as.numeric(logit_fpr_row[["Estimate"]]),
      se_logit_sensitivity = as.numeric(logit_sens_row[["Std. Error"]]),
      se_logit_fpr = as.numeric(logit_fpr_row[["Std. Error"]]),
      pooled_sensitivity = as.numeric(sens_row[["Estimate"]]),
      sensitivity_ci_low = as.numeric(sens_row[["95%ci.lb"]]),
      sensitivity_ci_high = as.numeric(sens_row[["95%ci.ub"]]),
      pooled_fpr = pooled_fpr,
      fpr_ci_low = fpr_ci_low,
      fpr_ci_high = fpr_ci_high,
      pooled_specificity = pooled_spec,
      specificity_ci_low = 1 - fpr_ci_high,
      specificity_ci_high = 1 - fpr_ci_low,
      tau2_sensitivity = tau2_sens,
      tau2_fpr = tau2_fpr,
      cov_sensitivity_fpr = cov_sens_fpr,
      rho_sensitivity_fpr = as.numeric(rho),
      log_diagnostic_odds_ratio = as.numeric(logit_sens_row[["Estimate"]]) - as.numeric(logit_fpr_row[["Estimate"]]),
      diagnostic_odds_ratio = exp(as.numeric(logit_sens_row[["Estimate"]]) - as.numeric(logit_fpr_row[["Estimate"]])),
      auc = as.numeric(auc$AUC),
      partial_auc = as.numeric(auc$pAUC),
      vcov_logit_sensitivity = as.numeric(vc[1, 1]),
      vcov_logit_fpr = as.numeric(vc[2, 2]),
      vcov_logit_cov = as.numeric(vc[1, 2])
    ),
    limitations = list(
      "Source-backed open-access DTA table validation only; not clinical diagnostic accuracy guidance.",
      "This output is a narrow mada::reitsma reference check, not broad HSROC or DTA production certification."
    )
  )
  jsonlite::write_json(
    output,
    parsed$output,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null",
    digits = 15
  )
}

main()
