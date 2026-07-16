"""Advanced Bias-Adjusted NMA pooling with exact Binomial GLMM, Kenward-Roger corrections, and sensitivity analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
import scipy.stats
from scipy.optimize import minimize, minimize_scalar

from bias_nma_adv.data import EvidenceDataset, ValidationError, StudyRecord, OutcomeADRecord, ArmRecord
from bias_nma_adv.sponsor_bias import RegistrySponsorAuditor

@dataclass(frozen=True)
class AdvancedNMAFitResult:
    outcome_id: str
    measure_type: str
    reference_treatment: str
    reference_design: str
    parameter_names: tuple[str, ...]
    parameter_estimates: np.ndarray
    parameter_cov: np.ndarray
    taus: dict[str, float]  # Design stratum to heterogeneity tau
    df: int
    hksj: bool
    q_factor: float
    n_studies: int
    n_contrasts: int
    treatment_effects: dict[str, float]
    treatment_ses: dict[str, float]
    design_biases: dict[str, float]
    design_biases_ses: dict[str, float]
    covariate_effects: dict[str, float]
    covariate_effects_ses: dict[str, float]
    study_specific_biases: dict[str, float]
    study_specific_biases_ses: dict[str, float]
    weight_sensitivity_stds: dict[str, float] = field(default_factory=dict)
    exact_binomial_active: bool = False
    target_population: str = "enriched_as_randomised"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def contrast(
        self,
        treatment_a: str,
        treatment_b: str,
        *,
        design: str | None = None,
        covariates: dict[str, float] | None = None,
        study_id: str | None = None,
        alpha: float = 0.05
    ) -> tuple[float, float, float, float]:
        """Calculate the estimated difference between treatment_a and treatment_b.

        Returns (effect, se, ci_lower, ci_upper).
        """
        cov_dict = covariates or {}
        design_name = design or self.reference_design

        # Construct coefficient vector for contrast
        coeff = self._contrast_coeff(treatment_a, treatment_b, design_name, cov_dict, study_id)
        effect = float(coeff @ self.parameter_estimates)
        variance = float(coeff @ self.parameter_cov @ coeff)
        se = math.sqrt(max(variance, 0.0))

        # Confidence interval
        if self.hksj:
            t_crit = scipy.stats.t.ppf(1.0 - alpha / 2.0, self.df)
            ci_lower = effect - t_crit * se
            ci_upper = effect + t_crit * se
        else:
            z_crit = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
            ci_lower = effect - z_crit * se
            ci_upper = effect + z_crit * se

        return effect, se, ci_lower, ci_upper

    def _contrast_coeff(
        self,
        treatment_a: str,
        treatment_b: str,
        design: str,
        covariates: dict[str, float],
        study_id: str | None
    ) -> np.ndarray:
        coeff = np.zeros(len(self.parameter_names), dtype=float)

        # 1. Treatment main effects
        self._set_param_coeff(coeff, f"trt_{treatment_a}", 1.0)
        self._set_param_coeff(coeff, f"trt_{treatment_b}", -1.0)

        # 2. Study-specific bias
        if study_id is not None:
            self._set_param_coeff(coeff, f"bias_study_{study_id}", 1.0)
        elif design != self.reference_design:
            # Fallback to design bias
            self._set_param_coeff(coeff, f"bias_{design}", 1.0)

        # 3. Covariate interactions
        for cov_name, val in covariates.items():
            # Treatment-by-covariate interaction
            self._set_param_coeff(coeff, f"trt_{treatment_a}_x_{cov_name}", val)
            self._set_param_coeff(coeff, f"trt_{treatment_b}_x_{cov_name}", -val)

            # Design-by-covariate interaction (only relevant if not study-specific)
            if study_id is None and design != self.reference_design:
                self._set_param_coeff(coeff, f"bias_{design}_x_{cov_name}", val)

        return coeff

    def _set_param_coeff(self, coeff: np.ndarray, param_name: str, val: float) -> None:
        if param_name in self.parameter_names:
            idx = self.parameter_names.index(param_name)
            coeff[idx] = val


@dataclass(frozen=True)
class _StudyBlock:
    study_id: str
    design: str
    rob_weight: float
    covariates: dict[str, float]
    y: np.ndarray
    v: np.ndarray
    trt_plus: tuple[str, ...]
    trt_minus: tuple[str, ...]
    baseline_events: float
    baseline_n: int
    active_events: tuple[float, ...]
    active_n: tuple[int, ...]


class AdvancedBiasAdjustedNMAPooler:
    """Frequentist advanced bias-adjusted NMA pooling engine."""

    def __init__(
        self,
        hksj: bool = True,
        hksj_df: str = "studies",
        down_weight: bool = True,
        variance_type: str = "model",
        random_effects: str | bool = True,  # True, False, or "stratified"
        treatment_shrinkage_lambda: float = 0.0,
        study_specific_bias: bool = False,
        bias_prior_mean: float = 0.0,
        exact_binomial: str | bool = "auto",
        apply_sponsor_bias: bool = False,
        sponsor_auditor: RegistrySponsorAuditor | None = None,
        apply_indirectness: bool = False,
        target_population: str = "enriched_as_randomised",
    ):
        if hksj_df not in {"studies", "contrasts"}:
            raise ValueError("hksj_df must be 'studies' or 'contrasts'.")
        if variance_type not in {"model", "sandwich"}:
            raise ValueError("variance_type must be 'model' or 'sandwich'.")
        if random_effects not in {True, False, "stratified"}:
            raise ValueError("random_effects must be True, False, or 'stratified'.")
        if exact_binomial not in {True, False, "auto"}:
            raise ValueError("exact_binomial must be True, False, or 'auto'.")
        if apply_sponsor_bias and sponsor_auditor is None:
            raise ValueError("apply_sponsor_bias=True requires sponsor_auditor.")
        if target_population not in {"enriched_as_randomised", "unselected_target"}:
            raise ValueError(
                "target_population must be 'enriched_as_randomised' or 'unselected_target'."
            )
        self.hksj = hksj
        self.hksj_df = hksj_df
        self.down_weight = down_weight
        self.variance_type = variance_type
        self.random_effects = random_effects
        self.treatment_shrinkage_lambda = treatment_shrinkage_lambda
        self.study_specific_bias = study_specific_bias
        self.bias_prior_mean = bias_prior_mean
        self.exact_binomial = exact_binomial
        self.apply_sponsor_bias = apply_sponsor_bias
        self.sponsor_auditor = sponsor_auditor
        self.apply_indirectness = apply_indirectness
        self.target_population = target_population

    def fit(
        self,
        dataset: EvidenceDataset,
        outcome_id: str,
        reference_treatment: str,
        reference_design: str = "rct",
        bias_prior_sd: float = 1.0,
        covariates: list[str] | None = None
    ) -> AdvancedNMAFitResult:
        cov_names = covariates or []
        measure_type = dataset.measure_type_for_outcome(outcome_id)

        # 1. Assemble study blocks
        blocks = self._build_study_blocks(dataset, outcome_id, measure_type)
        if not blocks:
            raise ValidationError(f"No studies available for outcome_id '{outcome_id}'.")

        # 2. Network connectivity check
        all_treatments = self._all_treatments(blocks)
        if reference_treatment not in all_treatments:
            raise ValidationError(f"reference_treatment '{reference_treatment}' not present in network.")

        self._validate_identifiable_network(blocks, reference_treatment, outcome_id)

        parameter_treatments = tuple(
            t for t in sorted(all_treatments) if t != reference_treatment
        )

        # Compute degree centralities for topological regularization
        treatment_degrees = {t: 0 for t in all_treatments}
        for block in blocks:
            study_trts = set(block.trt_plus) | set(block.trt_minus)
            for t in study_trts:
                treatment_degrees[t] += 1
        max_deg = max(treatment_degrees.values()) if treatment_degrees else 1
        treatment_centralities = {t: deg / max_deg for t, deg in treatment_degrees.items()}

        # 3. Non-reference designs & studies for bias adjustment
        all_designs = {b.design for b in blocks}
        parameter_designs = tuple(
            d for d in sorted(all_designs) if d != reference_design
        )

        nrs_studies = tuple(
            b.study_id for b in blocks if b.design != reference_design
        )

        # 4. Construct parameter mapping and design matrix
        param_names = self._build_parameter_names(
            parameter_treatments,
            parameter_designs,
            cov_names,
            self.study_specific_bias,
            nrs_studies
        )
        y, x, v = self._assemble_design(
            blocks,
            param_names,
            reference_treatment,
            reference_design,
            cov_names,
            self.study_specific_bias
        )

        n_contrasts = int(y.shape[0])
        warnings_list: list[str] = []
        if self.apply_indirectness:
            warnings_list.append(
                "Indirectness is enabled as a declared estimand sensitivity, but no "
                "mechanism-specific numerical delta is applied by default."
            )
        
        # Checking rank of X (with a fallback regularization standard)
        if np.linalg.matrix_rank(x) < x.shape[1] and not (self.treatment_shrinkage_lambda > 0.0 or self.study_specific_bias):
            raise ValidationError(
                f"Design matrix for outcome '{outcome_id}' is rank-deficient; "
                "parameters are not jointly identifiable. Simplify covariates or enable shrinkage."
            )

        unique_designs = sorted(list(all_designs))
        design_to_idx = {d: idx for idx, d in enumerate(unique_designs)}

        # 5. Determine whether to run Exact Binomial Likelihood
        use_exact = False
        if measure_type == "binary":
            if self.exact_binomial is True:
                use_exact = True
            elif self.exact_binomial == "auto":
                # Automatically trigger if rare events (e.g. rate < 5%) or any 0 events detected
                total_events = 0
                total_n = 0
                has_zero = False
                for b in blocks:
                    total_events += b.baseline_events + sum(b.active_events)
                    total_n += b.baseline_n + sum(b.active_n)
                    if b.baseline_events == 0.0 or any(ae == 0.0 for ae in b.active_events):
                        has_zero = True
                rate = total_events / max(total_n, 1.0)
                if rate < 0.05 or has_zero:
                    use_exact = True

        if use_exact:
            # FIT EXACT BINOMIAL GLMM
            n_studies = len(blocks)
            n_params = len(param_names)
            
            # Prior overall baseline rate logit
            total_evs = sum(b.baseline_events + sum(b.active_events) for b in blocks)
            total_n_val = sum(b.baseline_n + sum(b.active_n) for b in blocks)
            raw_rate = (total_evs + 0.5) / (total_n_val + 1.0)
            mu_0 = math.log(raw_rate / (1.0 - raw_rate))
            sigma_mu = 4.0

            # Initial guess: baseline logits + zeros for beta
            theta0 = np.zeros(n_studies + n_params, dtype=float)
            for s_idx, block in enumerate(blocks):
                s_rate = (block.baseline_events + 0.1) / (block.baseline_n + 0.2)
                theta0[s_idx] = math.log(s_rate / (1.0 - s_rate))

            # Assemble prior precision matrix P for gradient and objective
            p_matrix = np.zeros((n_params, n_params), dtype=float)
            study_rob = {b.study_id: b.rob_weight for b in blocks}
            for idx, name in enumerate(param_names):
                if name.startswith("bias_study_"):
                    s_id = name[11:]
                    rob = study_rob.get(s_id, 1.0)
                    p_matrix[idx, idx] = 1.0 / (((bias_prior_sd * bias_prior_sd) * (1.0 - rob)) + 1e-6)
                elif name.startswith("bias_") or "_x_" in name:
                    p_matrix[idx, idx] = 1.0 / (bias_prior_sd * bias_prior_sd)
                elif name.startswith("trt_") and "_x_" not in name:
                    trt = name[4:]
                    p_matrix[idx, idx] = self.treatment_shrinkage_lambda * (1.0 - treatment_centralities.get(trt, 1.0))

            res = minimize(
                self._exact_binomial_nll,
                theta0,
                args=(blocks, x, param_names, p_matrix, self.bias_prior_mean, mu_0, sigma_mu),
                jac=self._exact_binomial_grad,
                method="L-BFGS-B"
            )

            if not res.success:
                warnings_list.append("Exact binomial optimization failed to converge; fallback to approximate model.")
                use_exact = False

        # Fit model
        if use_exact:
            opt_mus = res.x[:n_studies]
            beta = res.x[n_studies:]

            # Compute Hessian at optimum for covariance matrix
            h_mu_mu = np.zeros(n_studies)
            h_mu_beta = np.zeros((n_studies, n_params))
            h_beta_beta = p_matrix.copy()

            cursor = 0
            for s_idx, block in enumerate(blocks):
                # Baseline
                eta_0 = opt_mus[s_idx]
                p_0 = 1.0 / (1.0 + np.exp(-eta_0))
                w_0 = block.baseline_n * p_0 * (1.0 - p_0)
                h_mu_mu[s_idx] += w_0 + 1.0 / (sigma_mu * sigma_mu)

                # Actives
                size = len(block.active_events)
                for a_idx in range(size):
                    x_row = x[cursor + a_idx]
                    eta_k = opt_mus[s_idx] + x_row @ beta
                    p_k = 1.0 / (1.0 + np.exp(-eta_k))
                    w_k = block.active_n[a_idx] * p_k * (1.0 - p_k)

                    h_mu_mu[s_idx] += w_k
                    h_mu_beta[s_idx] += w_k * x_row
                    h_beta_beta += w_k * np.outer(x_row, x_row)

                cursor += size

            h_total = np.zeros((n_studies + n_params, n_studies + n_params))
            for s_idx in range(n_studies):
                h_total[s_idx, s_idx] = h_mu_mu[s_idx]
            h_total[:n_studies, n_studies:] = h_mu_beta
            h_total[n_studies:, :n_studies] = h_mu_beta.T
            h_total[n_studies:, n_studies:] = h_beta_beta

            h_inv = np.linalg.pinv(h_total)
            cov = h_inv[n_studies:, n_studies:]
            taus_dict = {d: 0.0 for d in unique_designs}
            df = max(1, n_studies - n_params)
            q_factor = 1.0
            weight_sensitivity_stds = {}
        else:
            # 6. Fit standard REML / GLS with Prior Shrinkage and Kenward-Roger Correction
            taus_dict = {d: 0.0 for d in unique_designs}
            if self.random_effects == "stratified" and y.shape[0] > x.shape[1]:
                taus_vec = self._optimize_tau_reml_stratified(y, x, v, blocks, design_to_idx)
                for d, idx in design_to_idx.items():
                    taus_dict[d] = float(taus_vec[idx])
            elif self.random_effects is True and y.shape[0] > x.shape[1]:
                tau_val = self._optimize_tau_reml(y, x, v)
                for d in unique_designs:
                    taus_dict[d] = tau_val
            elif self.random_effects:
                warnings_list.append("Insufficient degrees of freedom for random effects; tau fixed at 0.0.")

            # Build block-wise random-effects covariance matrix M
            m = v.copy()
            tot_size = y.shape[0]
            cursor = 0
            for block in blocks:
                size = block.y.shape[0]
                tau = taus_dict[block.design]
                m[cursor : cursor + size, cursor : cursor + size] += np.eye(size, dtype=float) * (tau * tau)
                cursor += size

            beta, cov = self._estimate_gls_with_shrinkage_coupled_raw(
                y, x, m, param_names, blocks, bias_prior_sd,
                self.treatment_shrinkage_lambda, treatment_centralities,
                self.variance_type, self.bias_prior_mean
            )

            # Kenward-Roger Style Covariance Correction
            if self.random_effects and y.shape[0] > x.shape[1]:
                try:
                    m_inv = np.linalg.inv(m)
                    n_designs = len(unique_designs)
                    d_matrices = []
                    for g in range(n_designs):
                        d_g = np.zeros(tot_size, dtype=float)
                        c = 0
                        for block in blocks:
                            size = block.y.shape[0]
                            if design_to_idx[block.design] == g:
                                d_g[c : c + size] = 1.0
                            c += size
                        d_matrices.append(d_g)

                    proj = m_inv - m_inv @ x @ cov @ x.T @ m_inv
                    fisher_info = np.zeros((n_designs, n_designs))
                    for g in range(n_designs):
                        for h in range(n_designs):
                            val = 0.5 * np.sum((proj ** 2) * np.outer(d_matrices[h], d_matrices[g]))
                            fisher_info[g, h] = val

                    cov_taus = np.linalg.pinv(fisher_info)
                    cov_correction = np.zeros_like(cov)

                    for g in range(n_designs):
                        m_inv_dg_m_inv = (m_inv * d_matrices[g]) @ m_inv
                        d_phi_g = cov @ (x.T @ m_inv_dg_m_inv @ x) @ cov
                        for h in range(n_designs):
                            m_inv_dh_m_inv = (m_inv * d_matrices[h]) @ m_inv
                            d_phi_h = cov @ (x.T @ m_inv_dh_m_inv @ x) @ cov
                            cov_correction += cov_taus[g, h] * (d_phi_g @ np.linalg.pinv(cov) @ d_phi_h)

                    cov = cov + cov_correction
                except Exception as e:
                    warnings_list.append(f"Kenward-Roger covariance correction failed: {str(e)}")

            n_contrasts = int(y.shape[0])
            n_studies = len(blocks)
            n_params = len(param_names)

            if self.hksj_df == "studies":
                df = n_studies - n_params
                if df <= 0:
                    df = max(1, n_contrasts - n_params)
            else:
                df = max(1, n_contrasts - n_params)

            m_inv = np.linalg.inv(m)
            resid = y - x @ beta
            q_stat = float(resid.T @ m_inv @ resid)
            q_factor = 1.0

            if self.hksj:
                q_factor = max(1.0, q_stat / df)
                cov = cov * q_factor

            # Quality-Weight Sensitivity Perturbation Analysis
            weight_sensitivity_stds = {}
            if self.down_weight and len(blocks) > 1:
                perturbed_effects = {t: [] for t in parameter_treatments}
                for p_seed in range(30):
                    rng = np.random.default_rng(p_seed)
                    p_blocks = []
                    for b in blocks:
                        noise = rng.uniform(-0.1, 0.1)
                        new_w = float(np.clip(b.rob_weight * (1.0 + noise), 0.01, 1.0))
                        new_v = b.v * (b.rob_weight / new_w)
                        p_blocks.append(_StudyBlock(
                            study_id=b.study_id,
                            design=b.design,
                            rob_weight=new_w,
                            covariates=b.covariates,
                            y=b.y,
                            v=new_v,
                            trt_plus=b.trt_plus,
                            trt_minus=b.trt_minus,
                            baseline_events=b.baseline_events,
                            baseline_n=b.baseline_n,
                            active_events=b.active_events,
                            active_n=b.active_n
                        ))
                    try:
                        p_y, p_x, p_v = self._assemble_design(
                            p_blocks, param_names, reference_treatment, reference_design, cov_names, self.study_specific_bias
                        )
                        p_m = p_v.copy()
                        c_idx = 0
                        for pb in p_blocks:
                            sz = pb.y.shape[0]
                            tau = taus_dict[pb.design]
                            p_m[c_idx : c_idx + sz, c_idx : c_idx + sz] += np.eye(sz, dtype=float) * (tau * tau)
                            c_idx += sz

                        p_beta, _ = self._estimate_gls_with_shrinkage_coupled_raw(
                            p_y, p_x, p_m, param_names, p_blocks, bias_prior_sd,
                            self.treatment_shrinkage_lambda, treatment_centralities,
                            self.variance_type, self.bias_prior_mean
                        )
                        for idx, name in enumerate(param_names):
                            if name.startswith("trt_") and "_x_" not in name:
                                trt = name[4:]
                                perturbed_effects[trt].append(float(p_beta[idx]))
                    except Exception:
                        continue

                for t in parameter_treatments:
                    if perturbed_effects[t]:
                        weight_sensitivity_stds[t] = float(np.std(perturbed_effects[t]))

        # 11. Extract parameter estimates and standard errors
        treatment_effects = {reference_treatment: 0.0}
        treatment_ses = {reference_treatment: 0.0}
        design_biases = {reference_design: 0.0}
        design_biases_ses = {reference_design: 0.0}
        covariate_effects = {}
        covariate_effects_ses = {}
        study_specific_biases = {}
        study_specific_biases_ses = {}

        for idx, name in enumerate(param_names):
            est = float(beta[idx])
            se = math.sqrt(max(float(cov[idx, idx]), 0.0))

            if name.startswith("trt_") and "_x_" not in name:
                trt = name[4:]
                treatment_effects[trt] = est
                treatment_ses[trt] = se
            elif name.startswith("bias_study_"):
                s_id = name[11:]
                study_specific_biases[s_id] = est
                study_specific_biases_ses[s_id] = se
            elif name.startswith("bias_") and "_x_" not in name:
                bias_d = name[5:]
                design_biases[bias_d] = est
                design_biases_ses[bias_d] = se
            else:
                covariate_effects[name] = est
                covariate_effects_ses[name] = se

        return AdvancedNMAFitResult(
            outcome_id=outcome_id,
            measure_type=measure_type,
            reference_treatment=reference_treatment,
            reference_design=reference_design,
            parameter_names=param_names,
            parameter_estimates=beta,
            parameter_cov=cov,
            taus=taus_dict,
            df=df,
            hksj=self.hksj,
            q_factor=q_factor,
            n_studies=n_studies,
            n_contrasts=n_contrasts,
            treatment_effects=treatment_effects,
            treatment_ses=treatment_ses,
            design_biases=design_biases,
            design_biases_ses=design_biases_ses,
            covariate_effects=covariate_effects,
            covariate_effects_ses=covariate_effects_ses,
            study_specific_biases=study_specific_biases,
            study_specific_biases_ses=study_specific_biases_ses,
            weight_sensitivity_stds=weight_sensitivity_stds,
            exact_binomial_active=use_exact,
            target_population=self.target_population,
            warnings=tuple(warnings_list)
        )

    def _build_study_blocks(
        self,
        dataset: EvidenceDataset,
        outcome_id: str,
        measure_type: str
    ) -> list[_StudyBlock]:
        arm_lookup = dataset.arm_lookup()
        studies_with_outcome = sorted(
            {o.study_id for o in dataset.outcomes_ad if o.outcome_id == outcome_id}
        )

        blocks: list[_StudyBlock] = []
        for study_id in studies_with_outcome:
            outcomes = dataset.outcomes_by_study_outcome(study_id, outcome_id)
            if len(outcomes) < 2:
                continue

            study = dataset.studies.get(study_id)
            design = study.design if study else "other"
            rob_weight = study.rob_weight if (study and self.down_weight) else 1.0
            if self.apply_sponsor_bias:
                if study is None:
                    raise ValidationError(
                        f"{study_id}: sponsor-bias adjustment requires a StudyRecord."
                    )
                assert self.sponsor_auditor is not None
                if study_id not in self.sponsor_auditor.registry_db:
                    raise ValidationError(
                        f"{study_id}: sponsor-bias adjustment requires registered "
                        "sponsor and attrition metadata."
                    )
                rob_weight = self.sponsor_auditor.adjust_doi_welton_quality(
                    study_id,
                    base_quality=rob_weight,
                )
            covariates = study.covariates if study else {}

            # Sweeting's Adaptive Continuity Correction
            has_zero = False
            for outcome in outcomes:
                arm = arm_lookup.get((outcome.study_id, outcome.arm_id))
                if arm and outcome.measure_type == "binary":
                    events = outcome.value
                    non_events = float(arm.n) - outcome.value
                    if events <= 0.0 or non_events <= 0.0:
                        has_zero = True
                        break

            cc = {}
            if has_zero and measure_type == "binary":
                total_n = sum(arm_lookup[(o.study_id, o.arm_id)].n for o in outcomes if (o.study_id, o.arm_id) in arm_lookup)
                for outcome in outcomes:
                    arm = arm_lookup.get((outcome.study_id, outcome.arm_id))
                    if arm:
                        cc[arm.arm_id] = (float(arm.n) / max(total_n, 1.0)) * 1.0
            else:
                for outcome in outcomes:
                    cc[outcome.arm_id] = 0.0

            arm_effects = []
            for outcome in outcomes:
                if outcome.measure_type != measure_type:
                    continue
                arm = arm_lookup.get((outcome.study_id, outcome.arm_id))
                if not arm:
                    continue

                mean, variance = self._arm_measure_and_variance(
                    n=arm.n,
                    value=outcome.value,
                    se=outcome.se,
                    measure_type=measure_type,
                    cc=cc.get(arm.arm_id, 0.0)
                )
                variance = variance / rob_weight

                arm_effects.append({
                    "arm_id": arm.arm_id,
                    "treatment_id": arm.treatment_id,
                    "mean": mean,
                    "variance": variance,
                    "events": outcome.value,
                    "n": arm.n
                })

            if len(arm_effects) < 2:
                continue

            arm_effects.sort(key=lambda x: x["arm_id"])
            baseline = arm_effects[0]
            nonbaseline = arm_effects[1:]

            y = np.array([arm["mean"] - baseline["mean"] for arm in nonbaseline], dtype=float)
            baseline_var = baseline["variance"]
            v = np.full((len(nonbaseline), len(nonbaseline)), baseline_var, dtype=float)
            for idx, arm in enumerate(nonbaseline):
                v[idx, idx] = baseline_var + arm["variance"]

            blocks.append(_StudyBlock(
                study_id=study_id,
                design=design,
                rob_weight=rob_weight,
                covariates=covariates,
                y=y,
                v=v,
                trt_plus=tuple(arm["treatment_id"] for arm in nonbaseline),
                trt_minus=tuple(baseline["treatment_id"] for _ in nonbaseline),
                baseline_events=baseline["events"],
                baseline_n=baseline["n"],
                active_events=tuple(arm["events"] for arm in nonbaseline),
                active_n=tuple(arm["n"] for arm in nonbaseline)
            ))
        return blocks

    @staticmethod
    def _arm_measure_and_variance(
        n: int,
        value: float,
        se: float | None,
        measure_type: str,
        cc: float = 0.0
    ) -> tuple[float, float]:
        if measure_type == "continuous":
            if se is None or se <= 0:
                raise ValidationError("Continuous outcomes require SE > 0.")
            return value, se * se

        if measure_type != "binary":
            raise ValidationError(f"Unsupported measure_type: {measure_type}")

        events = value
        non_events = float(n) - value
        if events < 0 or non_events < 0:
            raise ValidationError("Binary outcome value must be within [0, n].")

        odds = (events + cc) / (non_events + cc)
        mean = math.log(odds)
        variance = (1.0 / (events + cc)) + (1.0 / (non_events + cc))
        return mean, variance

    @staticmethod
    def _all_treatments(blocks: list[_StudyBlock]) -> set[str]:
        treatments: set[str] = set()
        for block in blocks:
            treatments.update(block.trt_plus)
            treatments.update(block.trt_minus)
        return treatments

    def _connected_components(self, blocks: list[_StudyBlock]) -> list[set[str]]:
        treatments = self._all_treatments(blocks)
        edges: set[tuple[str, str]] = set()
        for block in blocks:
            study_trts = sorted(set(block.trt_plus) | set(block.trt_minus))
            for i, left in enumerate(study_trts):
                for right in study_trts[i + 1 :]:
                    edges.add((left, right))

        adj = {t: set() for t in treatments}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        visited = set()
        components = []
        for node in treatments:
            if node not in visited:
                comp = set()
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)
        return components

    def _validate_identifiable_network(self, blocks: list[_StudyBlock], reference_treatment: str, outcome_id: str) -> None:
        components = self._connected_components(blocks)
        if len(components) <= 1:
            return

        ref_component = next((c for c in components if reference_treatment in c), set())
        disconnected = [c for c in components if reference_treatment not in c]
        disc_text = "; ".join(", ".join(c) for c in disconnected)
        raise ValidationError(
            f"Outcome '{outcome_id}' has a disconnected treatment network. Reference component: "
            f"{', '.join(ref_component)}. Disconnected components: {disc_text}."
        )

    @staticmethod
    def _build_parameter_names(
        parameter_treatments: tuple[str, ...],
        parameter_designs: tuple[str, ...],
        cov_names: list[str],
        study_specific_bias: bool,
        nrs_studies: tuple[str, ...]
    ) -> tuple[str, ...]:
        names: list[str] = []

        # 1. Treatment main effects
        for trt in parameter_treatments:
            names.append(f"trt_{trt}")

        # 2. Bias terms (Study-specific or design-level)
        if study_specific_bias:
            for s_id in nrs_studies:
                names.append(f"bias_study_{s_id}")
        else:
            for design in parameter_designs:
                names.append(f"bias_{design}")

        # 3. Covariate interactions (only if not study-specific bias, which absorbs all study covariates)
        for cov in cov_names:
            # Treatment interactions
            for trt in parameter_treatments:
                names.append(f"trt_{trt}_x_{cov}")
            
            # Design bias interactions
            if not study_specific_bias:
                for design in parameter_designs:
                    names.append(f"bias_{design}_x_{cov}")

        return tuple(names)

    @staticmethod
    def _assemble_design(
        blocks: list[_StudyBlock],
        param_names: tuple[str, ...],
        reference_treatment: str,
        reference_design: str,
        cov_names: list[str],
        study_specific_bias: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_parts = []
        x_parts = []
        v_parts = []

        param_to_idx = {name: idx for idx, name in enumerate(param_names)}

        for block in blocks:
            y_parts.append(block.y)
            v_parts.append(block.v)

            n_contrasts = len(block.y)
            x_block = np.zeros((n_contrasts, len(param_names)), dtype=float)

            for row_idx, (trt_plus, trt_minus) in enumerate(zip(block.trt_plus, block.trt_minus)):
                # A. Treatment Main Effects
                if trt_plus != reference_treatment:
                    x_block[row_idx, param_to_idx[f"trt_{trt_plus}"]] += 1.0
                if trt_minus != reference_treatment:
                    x_block[row_idx, param_to_idx[f"trt_{trt_minus}"]] -= 1.0

                # B. Bias Terms
                if block.design != reference_design:
                    if study_specific_bias:
                        bias_name = f"bias_study_{block.study_id}"
                        if bias_name in param_to_idx:
                            x_block[row_idx, param_to_idx[bias_name]] += 1.0
                    else:
                        bias_name = f"bias_{block.design}"
                        if bias_name in param_to_idx:
                            x_block[row_idx, param_to_idx[bias_name]] += 1.0

                # C. Covariate Interactions
                for cov in cov_names:
                    cov_val = block.covariates.get(cov, 0.0)

                    # Treatment interactions
                    if trt_plus != reference_treatment:
                        trt_cov_plus = f"trt_{trt_plus}_x_{cov}"
                        if trt_cov_plus in param_to_idx:
                            x_block[row_idx, param_to_idx[trt_cov_plus]] += cov_val
                    if trt_minus != reference_treatment:
                        trt_cov_minus = f"trt_{trt_minus}_x_{cov}"
                        if trt_cov_minus in param_to_idx:
                            x_block[row_idx, param_to_idx[trt_cov_minus]] -= cov_val

                    # Design interactions (only if not study-specific)
                    if not study_specific_bias and block.design != reference_design:
                        bias_cov_name = f"bias_{block.design}_x_{cov}"
                        if bias_cov_name in param_to_idx:
                            x_block[row_idx, param_to_idx[bias_cov_name]] += cov_val

            x_parts.append(x_block)

        y = np.concatenate(y_parts, axis=0)
        x = np.vstack(x_parts)

        tot_size = sum(block.shape[0] for block in v_parts)
        v = np.zeros((tot_size, tot_size), dtype=float)
        cursor = 0
        for block_v in v_parts:
            size = block_v.shape[0]
            v[cursor : cursor + size, cursor : cursor + size] = block_v
            cursor += size

        return y, x, v

    def _optimize_tau_reml(self, y: np.ndarray, x: np.ndarray, v: np.ndarray) -> float:
        def obj(t):
            m = v + np.eye(v.shape[0], dtype=float) * (t * t)
            try:
                m_inv = np.linalg.inv(m)
            except np.linalg.LinAlgError:
                return float("inf")
            xt_m_inv = x.T @ m_inv
            info = xt_m_inv @ x
            sign_m, logdet_m = np.linalg.slogdet(m)
            sign_info, logdet_info = np.linalg.slogdet(info)
            if sign_m <= 0 or sign_info <= 0:
                return float("inf")
            beta = np.linalg.pinv(info) @ (xt_m_inv @ y)
            resid = y - x @ beta
            q_stat = float(resid.T @ m_inv @ resid)
            return 0.5 * (logdet_m + logdet_info + q_stat)

        res = minimize_scalar(obj, bounds=(0.0, 10.0), method="bounded")
        if res.success:
            return float(max(res.x, 0.0))
        return 0.0

    def _optimize_tau_reml_stratified(
        self,
        y: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        blocks: list[_StudyBlock],
        design_to_idx: dict[str, int]
    ) -> np.ndarray:
        n_designs = len(design_to_idx)
        
        def obj(taus):
            m = v.copy()
            cursor = 0
            for block in blocks:
                size = block.y.shape[0]
                d_idx = design_to_idx.get(block.design, 0)
                tau = taus[d_idx]
                m[cursor : cursor + size, cursor : cursor + size] += np.eye(size, dtype=float) * (tau * tau)
                cursor += size

            try:
                m_inv = np.linalg.inv(m)
            except np.linalg.LinAlgError:
                return float("inf")

            xt_m_inv = x.T @ m_inv
            info = xt_m_inv @ x
            sign_m, logdet_m = np.linalg.slogdet(m)
            sign_info, logdet_info = np.linalg.slogdet(info)
            if sign_m <= 0 or sign_info <= 0:
                return float("inf")

            beta = np.linalg.pinv(info) @ (xt_m_inv @ y)
            resid = y - x @ beta
            q_stat = float(resid.T @ m_inv @ resid)
            return 0.5 * (logdet_m + logdet_info + q_stat)

        x0 = np.full(n_designs, 0.1, dtype=float)
        bounds = [(0.0, 10.0) for _ in range(n_designs)]
        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
        if res.success:
            return np.maximum(res.x, 0.0)
        return x0

    @staticmethod
    def _exact_binomial_nll(
        theta: np.ndarray,
        blocks: list[_StudyBlock],
        x: np.ndarray,
        param_names: tuple[str, ...],
        p_matrix: np.ndarray,
        bias_prior_mean: float,
        mu_0: float,
        sigma_mu: float
    ) -> float:
        n_studies = len(blocks)
        mus = theta[:n_studies]
        beta = theta[n_studies:]

        nll = 0.0
        cursor = 0
        for s_idx, block in enumerate(blocks):
            # Baseline
            eta_0 = mus[s_idx]
            nll += block.baseline_n * np.logaddexp(0.0, eta_0) - block.baseline_events * eta_0

            # Actives
            size = len(block.active_events)
            for a_idx in range(size):
                x_row = x[cursor + a_idx]
                eta_k = mus[s_idx] + x_row @ beta
                nll += block.active_n[a_idx] * np.logaddexp(0.0, eta_k) - block.active_events[a_idx] * eta_k

            cursor += size
            # Intercept regularization
            nll += 0.5 * ((mus[s_idx] - mu_0) ** 2) / (sigma_mu * sigma_mu)

        # Regularized beta penalty
        mu_vector = np.zeros(len(beta), dtype=float)
        for idx, name in enumerate(param_names):
            if name.startswith("bias_study_") or name.startswith("bias_"):
                mu_vector[idx] = bias_prior_mean

        diff = beta - mu_vector
        nll += 0.5 * diff.T @ p_matrix @ diff
        return nll

    @staticmethod
    def _exact_binomial_grad(
        theta: np.ndarray,
        blocks: list[_StudyBlock],
        x: np.ndarray,
        param_names: tuple[str, ...],
        p_matrix: np.ndarray,
        bias_prior_mean: float,
        mu_0: float,
        sigma_mu: float
    ) -> np.ndarray:
        n_studies = len(blocks)
        mus = theta[:n_studies]
        beta = theta[n_studies:]

        grad = np.zeros_like(theta)
        grad_mus = grad[:n_studies]
        grad_beta = grad[n_studies:]

        cursor = 0
        for s_idx, block in enumerate(blocks):
            # Baseline
            eta_0 = mus[s_idx]
            p_0 = 1.0 / (1.0 + np.exp(-eta_0))
            grad_mus[s_idx] += block.baseline_n * p_0 - block.baseline_events

            # Actives
            size = len(block.active_events)
            for a_idx in range(size):
                x_row = x[cursor + a_idx]
                eta_k = mus[s_idx] + x_row @ beta
                p_k = 1.0 / (1.0 + np.exp(-eta_k))
                diff = block.active_n[a_idx] * p_k - block.active_events[a_idx]

                grad_mus[s_idx] += diff
                grad_beta += diff * x_row

            cursor += size
            grad_mus[s_idx] += (mus[s_idx] - mu_0) / (sigma_mu * sigma_mu)

        mu_vector = np.zeros(len(beta), dtype=float)
        for idx, name in enumerate(param_names):
            if name.startswith("bias_study_") or name.startswith("bias_"):
                mu_vector[idx] = bias_prior_mean

        grad_beta += p_matrix @ (beta - mu_vector)
        return grad

    @staticmethod
    def _estimate_gls_with_shrinkage_coupled_raw(
        y: np.ndarray,
        x: np.ndarray,
        m: np.ndarray,
        param_names: tuple[str, ...],
        blocks: list[_StudyBlock],
        bias_prior_sd: float,
        treatment_shrinkage_lambda: float,
        treatment_centralities: dict[str, float],
        variance_type: str,
        bias_prior_mean: float
    ) -> tuple[np.ndarray, np.ndarray]:
        m_inv = np.linalg.inv(m)
        xt_m_inv = x.T @ m_inv
        info = xt_m_inv @ x

        p_matrix = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        mu_vector = np.zeros(x.shape[1], dtype=float)
        study_rob = {b.study_id: b.rob_weight for b in blocks}

        for idx, name in enumerate(param_names):
            if name.startswith("bias_study_"):
                s_id = name[11:]
                rob = study_rob.get(s_id, 1.0)
                prior_var = (bias_prior_sd * bias_prior_sd) * (1.0 - rob)
                p_matrix[idx, idx] = 1.0 / (prior_var + 1e-6)
                mu_vector[idx] = bias_prior_mean
            elif name.startswith("bias_") or "_x_" in name:
                p_matrix[idx, idx] = 1.0 / (bias_prior_sd * bias_prior_sd)
                if name.startswith("bias_"):
                    mu_vector[idx] = bias_prior_mean
            elif name.startswith("trt_") and "_x_" not in name:
                trt = name[4:]
                p_matrix[idx, idx] = treatment_shrinkage_lambda * (1.0 - treatment_centralities.get(trt, 1.0))

        post_info = info + p_matrix
        cov_model = np.linalg.pinv(post_info)
        beta = cov_model @ (xt_m_inv @ y + p_matrix @ mu_vector)

        if variance_type == "sandwich":
            meat = np.zeros_like(info)
            cursor = 0
            for block in blocks:
                size = len(block.y)
                m_s_inv = m_inv[cursor : cursor + size, cursor : cursor + size]

                x_s = x[cursor : cursor + size]
                y_s = y[cursor : cursor + size]
                e_s = y_s - x_s @ beta

                outer = np.outer(e_s, e_s)
                meat_s = x_s.T @ m_s_inv @ outer @ m_s_inv @ x_s
                meat += meat_s

                cursor += size

            cov = cov_model @ meat @ cov_model
        else:
            cov = cov_model

        return beta, cov
