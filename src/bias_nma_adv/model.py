"""Advanced Bias-Adjusted NMA pooling with Kenward-Roger corrections, directional priors, Sweeting's corrections, and sensitivity analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
import scipy.stats
from scipy.optimize import minimize, minimize_scalar

from bias_nma_adv.data import EvidenceDataset, ValidationError, StudyRecord, OutcomeADRecord, ArmRecord

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
        bias_prior_mean: float = 0.0
    ):
        if hksj_df not in {"studies", "contrasts"}:
            raise ValueError("hksj_df must be 'studies' or 'contrasts'.")
        if variance_type not in {"model", "sandwich"}:
            raise ValueError("variance_type must be 'model' or 'sandwich'.")
        if random_effects not in {True, False, "stratified"}:
            raise ValueError("random_effects must be True, False, or 'stratified'.")
        self.hksj = hksj
        self.hksj_df = hksj_df
        self.down_weight = down_weight
        self.variance_type = variance_type
        self.random_effects = random_effects
        self.treatment_shrinkage_lambda = treatment_shrinkage_lambda
        self.study_specific_bias = study_specific_bias
        self.bias_prior_mean = bias_prior_mean

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

        warnings_list: list[str] = []
        
        # Checking rank of X (with a fallback regularization standard)
        if np.linalg.matrix_rank(x) < x.shape[1] and not (self.treatment_shrinkage_lambda > 0.0 or self.study_specific_bias):
            raise ValidationError(
                f"Design matrix for outcome '{outcome_id}' is rank-deficient; "
                "parameters are not jointly identifiable. Simplify covariates or enable shrinkage."
            )

        # 5. Stratified or Single REML heterogeneity optimization
        unique_designs = sorted(list(all_designs))
        design_to_idx = {d: idx for idx, d in enumerate(unique_designs)}
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

        # 6. Fit GLS with Prior Shrinkage (coupled by RoB quality and topology)
        beta, cov = self._estimate_gls_with_shrinkage_coupled(
            y, x, m, param_names, blocks, bias_prior_sd,
            self.treatment_shrinkage_lambda, treatment_centralities,
            self.variance_type, self.bias_prior_mean
        )

        # 7. Kenward-Roger Style Covariance Correction for plug-in heterogeneity uncertainty
        if self.random_effects and y.shape[0] > x.shape[1]:
            try:
                m_inv = np.linalg.inv(m)
                # Indicator diagonal matrices for design strata
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

                # Projection matrix: P = M^-1 - M^-1 X (X^T M^-1 X + P_prior)^-1 X^T M^-1
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

        # 8. Hartung-Knapp-Sidik-Jonkman adjustment
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

        # 9. Quality-Weight Sensitivity Perturbation Analysis
        weight_sensitivity_stds = {}
        if self.down_weight and len(blocks) > 1:
            perturbed_effects = {t: [] for t in parameter_treatments}
            # Run 30 fast perturbations
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
                        trt_minus=b.trt_minus
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

                    p_beta, _ = self._estimate_gls_with_shrinkage_coupled(
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

        # 10. Extract parameter estimates and standard errors
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
            covariates = study.covariates if study else {}

            # Sweeting's Adaptive Continuity Correction for Rare/Zero events
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
                        # Sweeting's correction proportional to arm size
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
                    "variance": variance
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
                trt_minus=tuple(baseline["treatment_id"] for _ in nonbaseline)
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
    def _estimate_gls_with_shrinkage_coupled(
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

        # Prior precision matrix (P) and prior mean vector (mu)
        p_matrix = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        mu_vector = np.zeros(x.shape[1], dtype=float)
        
        # Map study IDs to their quality weights
        study_rob = {b.study_id: b.rob_weight for b in blocks}

        for idx, name in enumerate(param_names):
            if name.startswith("bias_study_"):
                # Doi-Welton Hybrid coupling: precision inversely proportional to quality
                s_id = name[11:]
                rob = study_rob.get(s_id, 1.0)
                prior_var = (bias_prior_sd * bias_prior_sd) * (1.0 - rob)
                p_matrix[idx, idx] = 1.0 / (prior_var + 1e-6)
                mu_vector[idx] = bias_prior_mean
            elif name.startswith("bias_") or "_x_" in name:
                # Standard design bias / interaction term
                p_matrix[idx, idx] = 1.0 / (bias_prior_sd * bias_prior_sd)
                if name.startswith("bias_"):
                    mu_vector[idx] = bias_prior_mean
            elif name.startswith("trt_") and "_x_" not in name:
                # Topological Regularization: penalize sparse treatments
                trt = name[4:]
                centrality = treatment_centralities.get(trt, 1.0)
                p_matrix[idx, idx] = treatment_shrinkage_lambda * (1.0 - centrality)

        post_info = info + p_matrix
        cov_model = np.linalg.pinv(post_info)
        
        # Regularized GLS solver with non-zero prior means: beta = (X^T M^-1 X + P)^-1 (X^T M^-1 y + P mu)
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
