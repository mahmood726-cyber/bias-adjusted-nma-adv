#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with install.packages('jsonlite').")
  }
  if (!requireNamespace("MBNMAdose", quietly = TRUE)) {
    stop("Package 'MBNMAdose' is required. Install with install.packages('MBNMAdose').")
  }
  if (!requireNamespace("rjags", quietly = TRUE)) {
    stop("Package 'rjags' and a JAGS runtime are required for MBNMAdose.")
  }
})

parse_args <- function(args) {
  out <- list(
    arms = NULL,
    output = NULL,
    chains = 3L,
    iter = 3000L,
    burnin = 1000L,
    thin = 2L,
    seed = 20260718L
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--arms", "--output", "--chains", "--iter", "--burnin", "--thin", "--seed")) {
      if (i + 1 > length(args)) {
        stop(paste("Missing value for", key))
      }
      val <- args[[i + 1]]
      if (key == "--arms") out$arms <- val
      if (key == "--output") out$output <- val
      if (key == "--chains") out$chains <- as.integer(val)
      if (key == "--iter") out$iter <- as.integer(val)
      if (key == "--burnin") out$burnin <- as.integer(val)
      if (key == "--thin") out$thin <- as.integer(val)
      if (key == "--seed") out$seed <- as.integer(val)
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  if (is.null(out$arms) || is.null(out$output)) {
    stop("Usage: mbnmadose_semaglutide_polynomial.R --arms <arms.csv> --output <output.json>")
  }
  out
}

required_columns <- c(
  "study_id", "nct_id", "pmid", "arm_id", "group_id", "treatment", "agent",
  "dose", "dose_unit", "dose_frequency", "lsmean", "se", "outcome_id", "outcome_label"
)

weighted_linear_reference <- function(arms) {
  design <- cbind(intercept = 1, dose = arms$dose)
  weights <- diag(1 / (arms$se ^ 2), nrow = nrow(arms), ncol = nrow(arms))
  covariance <- solve(t(design) %*% weights %*% design)
  coefficients <- covariance %*% t(design) %*% weights %*% arms$lsmean
  list(
    intercept = as.numeric(coefficients[1]),
    slope = as.numeric(coefficients[2]),
    intercept_se = sqrt(as.numeric(covariance[1, 1])),
    slope_se = sqrt(as.numeric(covariance[2, 2]))
  )
}

main <- function() {
  parsed <- parse_args(commandArgs(trailingOnly = TRUE))
  arms <- utils::read.csv(parsed$arms, stringsAsFactors = FALSE)
  missing <- setdiff(required_columns, names(arms))
  if (length(missing) > 0) {
    stop(paste("MBNMAdose semaglutide arms missing required columns:", paste(missing, collapse = ", ")))
  }
  if (length(unique(arms$study_id)) != 1) {
    stop("This narrow MBNMAdose smoke adapter expects exactly one source-backed dose-ranging study.")
  }
  if (any(arms$se <= 0)) {
    stop("All standard errors must be positive.")
  }
  if (sum(arms$dose == 0) != 1) {
    stop("Exactly one placebo/reference dose arm is required.")
  }
  if (length(unique(arms$dose[arms$dose > 0])) < 2) {
    stop("At least two active dose levels are required.")
  }
  if (!all(grepl("^NCT[0-9]{8}$", arms$nct_id))) {
    stop("Malformed NCT identifier in MBNMAdose arms.")
  }
  if (!all(grepl("^[0-9]{1,9}$", as.character(arms$pmid)))) {
    stop("Malformed PMID in MBNMAdose arms.")
  }

  set.seed(parsed$seed)
  mbnma_input <- data.frame(
    studyID = arms$study_id,
    agent = arms$agent,
    dose = arms$dose,
    y = arms$lsmean,
    se = arms$se,
    stringsAsFactors = FALSE
  )
  network <- MBNMAdose::mbnma.network(
    mbnma_input,
    description = "Semaglutide obesity dose-ranging source-backed smoke"
  )
  fit <- MBNMAdose::mbnma.run(
    network,
    fun = MBNMAdose::dpoly(degree = 1),
    method = "common",
    likelihood = "normal",
    link = "identity",
    n.iter = parsed$iter,
    n.burnin = parsed$burnin,
    n.chains = parsed$chains,
    n.thin = parsed$thin,
    jags.seed = parsed$seed
  )

  summary_table <- as.data.frame(fit$BUGSoutput$summary)
  summary_table$parameter <- rownames(summary_table)
  beta_rows <- summary_table[grepl("^beta\\.1\\[", summary_table$parameter), , drop = FALSE]
  if (nrow(beta_rows) != 1) {
    stop("Could not identify exactly one beta.1 dose-response parameter.")
  }
  beta <- beta_rows[1, , drop = FALSE]
  wls <- weighted_linear_reference(arms)

  output <- list(
    schema_version = "mbnmadose_semaglutide_polynomial/v1",
    benchmark_id = "semaglutide_obesity_dose_response",
    source_policy = "clinicaltrials_gov + pubmed_abstract + open_access_paper only",
    evidence_mode = "ctgov_dose_response_lsmean",
    effect_scale = "absolute_percentage_point_change",
    package_versions = list(
      R = as.character(getRversion()),
      MBNMAdose = as.character(utils::packageVersion("MBNMAdose")),
      rjags = as.character(utils::packageVersion("rjags")),
      JAGS = as.character(rjags::jags.version()),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    ),
    model = list(
      engine = "MBNMAdose",
      likelihood = "normal",
      link = "identity",
      dose_function = "dpoly(degree=1)",
      method = "common",
      chains = parsed$chains,
      iter = parsed$iter,
      burnin = parsed$burnin,
      thin = parsed$thin,
      seed = parsed$seed
    ),
    study_arms = arms,
    mbnma = list(
      beta_1 = list(
        parameter = as.character(beta$parameter),
        mean = as.numeric(beta$mean),
        sd = as.numeric(beta$sd),
        ci_low = as.numeric(beta$`2.5%`),
        median = as.numeric(beta$`50%`),
        ci_high = as.numeric(beta$`97.5%`),
        rhat = as.numeric(beta$Rhat),
        n_eff = as.numeric(beta$n.eff)
      ),
      model_fit = list(
        dic = as.numeric(fit$BUGSoutput$DIC),
        pd = as.numeric(fit$BUGSoutput$pD),
        deviance_mean = as.numeric(summary_table[summary_table$parameter == "deviance", "mean"]),
        residual_deviance_mean = as.numeric(summary_table[summary_table$parameter == "totresdev", "mean"])
      )
    ),
    independent_wls_reference = wls,
    diagnostics = list(
      max_rhat = max(summary_table$Rhat, na.rm = TRUE),
      min_neff = min(summary_table$n.eff, na.rm = TRUE),
      n_parameters = nrow(summary_table)
    ),
    limitations = list(
      "Single source-backed semaglutide dose-ranging trial only.",
      "This is a narrow MBNMAdose linear polynomial smoke reference, not dose-response NMA feature parity.",
      "Shared-placebo covariance and multi-trial class effects are not tested here.",
      "This is not clinical, regulatory, HTA, or production certification."
    )
  )

  jsonlite::write_json(output, parsed$output, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

main()
