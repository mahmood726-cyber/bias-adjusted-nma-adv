"""Advanced Bias-Adjusted NMA pooling with Meta-Regression, HKSJ, and Scoped Down-Weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar

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
    tau: float
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
    warnings: tuple[str, ...]

    def contrast(
        self,
        treatment_a: str,
        treatment_b: str,
        *,
        design: str | None = None,
        covariates: dict[str, float] | None = None,
        alpha: float = 0.05
    ) -> tuple[float, float, float, float]:
        """Calculate the estimated difference between treatment_a and treatment_b.

        Returns (effect, se, ci_lower, ci_upper).
        """
        cov_dict = covariates or {}
        design_name = design or self.reference_design

        # Construct coefficient vector for contrast
        coeff = self._contrast_coeff(treatment_a, treatment_b, design_name, cov_dict)
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
        covariates: dict[str, float]
    ) -> np.ndarray:
        coeff = np.zeros(len(self.parameter_names), dtype=float)

        # 1. Treatment main effects
        self._set_param_coeff(coeff, f"trt_{treatment_a}", 1.0)
        self._set_param_coeff(coeff, f"trt_{treatment_b}", -1.0)

        # 2. Design bias main effects
        if design != self.reference_design:
            self._set_param_coeff(coeff, f"bias_{design}", 1.0)

        # 3. Covariate interactions
        for cov_name, val in covariates.items():
            # Treatment-by-covariate interaction
            self._set_param_coeff(coeff, f"trt_{treatment_a}_x_{cov_name}", val)
            self._set_param_coeff(coeff, f"trt_{treatment_b}_x_{cov_name}", -val)

            # Design-by-covariate interaction
            if design != self.reference_design:
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
    """Frequentist advanced bias-adjusted NMA with meta-regression, HKSJ, and scoped down-weighting."""

    def __init__(self, hksj: bool = True, hksj_df: str = "studies", down_weight: bool = True, variance_type: str = "model"):
        if hksj_df not in {"studies", "contrasts"}:
            raise ValueError("hksj_df must be 'studies' or 'contrasts'.")
        if variance_type not in {"model", "sandwich"}:
            raise ValueError("variance_type must be 'model' or 'sandwich'.")
        self.hksj = hksj
        self.hksj_df = hksj_df
        self.down_weight = down_weight
        self.variance_type = variance_type

    def fit(
        self,
        dataset: EvidenceDataset,
        outcome_id: str,
        reference_treatment: str,
        reference_design: str = "rct",
        random_effects: bool = True,
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

        # 3. Non-reference designs
        all_designs = {b.design for b in blocks}
        parameter_designs = tuple(
            d for d in sorted(all_designs) if d != reference_design
        )

        # 4. Construct parameter mapping and design matrix
        param_names = self._build_parameter_names(
            parameter_treatments, parameter_designs, cov_names
        )
        y, x, v = self._assemble_design(
            blocks, param_names, reference_treatment, reference_design, cov_names
        )

        warnings_list: list[str] = []
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValidationError(
                f"Design matrix for outcome '{outcome_id}' is rank-deficient; "
                "parameters are not jointly identifiable. Simplify covariates or designs."
            )

        # 5. REML heterogeneity optimization
        tau = 0.0
        if random_effects and y.shape[0] > x.shape[1]:
            tau = self._optimize_tau_reml(y, x, v)
        elif random_effects:
            warnings_list.append("Insufficient degrees of freedom for random effects; tau fixed at 0.0.")

        # 6. Fit GLS with Shrinkage Prior
        beta, cov = self._estimate_gls_with_shrinkage(
            y, x, v, tau, param_names, bias_prior_sd, blocks, self.variance_type
        )

        # 7. Hartung-Knapp-Sidik-Jonkman adjustment
        n_contrasts = int(y.shape[0])
        n_studies = len(blocks)
        n_params = len(param_names)

        if self.hksj_df == "studies":
            df = n_studies - n_params
            if df <= 0:
                df = max(1, n_contrasts - n_params)
        else:
            df = max(1, n_contrasts - n_params)

        # Residual weighted sum of squares
        m = v + np.eye(v.shape[0]) * (tau * tau)
        m_inv = np.linalg.inv(m)
        resid = y - x @ beta
        q_stat = float(resid.T @ m_inv @ resid)
        q_factor = 1.0

        if self.hksj:
            q_factor = max(1.0, q_stat / df)
            cov = cov * q_factor

        # 8. Extract parameter estimates and standard errors
        treatment_effects = {reference_treatment: 0.0}
        treatment_ses = {reference_treatment: 0.0}
        design_biases = {reference_design: 0.0}
        design_biases_ses = {reference_design: 0.0}
        covariate_effects = {}
        covariate_effects_ses = {}

        for idx, name in enumerate(param_names):
            est = float(beta[idx])
            se = math.sqrt(max(float(cov[idx, idx]), 0.0))

            if name.startswith("trt_") and "_x_" not in name:
                trt = name[4:]
                treatment_effects[trt] = est
                treatment_ses[trt] = se
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
            tau=tau,
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

            cc = self._study_continuity_correction(outcomes, arm_lookup, measure_type)
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
                    cc=cc
                )
                # Apply scoped down-weighting: inflate arm variance
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
    def _study_continuity_correction(outcomes: list[OutcomeADRecord], arm_lookup: dict, measure_type: str) -> float:
        if measure_type != "binary":
            return 0.0
        for outcome in outcomes:
            arm = arm_lookup.get((outcome.study_id, outcome.arm_id))
            if not arm:
                continue
            events = outcome.value
            non_events = float(arm.n) - outcome.value
            if events <= 0.0 or non_events <= 0.0:
                return 0.5
        return 0.0

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

        # BFS components finder
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
        cov_names: list[str]
    ) -> tuple[str, ...]:
        names: list[str] = []

        # 1. Treatment main effects
        for trt in parameter_treatments:
            names.append(f"trt_{trt}")

        # 2. Design bias main effects
        for design in parameter_designs:
            names.append(f"bias_{design}")

        # 3. Covariate interactions
        for cov in cov_names:
            # Treatment interactions
            for trt in parameter_treatments:
                names.append(f"trt_{trt}_x_{cov}")
            # Design bias interactions
            for design in parameter_designs:
                names.append(f"bias_{design}_x_{cov}")

        return tuple(names)

    @staticmethod
    def _assemble_design(
        blocks: list[_StudyBlock],
        param_names: tuple[str, ...],
        reference_treatment: str,
        reference_design: str,
        cov_names: list[str]
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

                # B. Design Bias Main Effects
                if block.design != reference_design:
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

                    # Design interactions
                    if block.design != reference_design:
                        bias_cov_name = f"bias_{block.design}_x_{cov}"
                        if bias_cov_name in param_to_idx:
                            x_block[row_idx, param_to_idx[bias_cov_name]] += cov_val

            x_parts.append(x_block)

        y = np.concatenate(y_parts, axis=0)
        x = np.vstack(x_parts)

        # Block-diagonal covariance matrix
        tot_size = sum(block.shape[0] for block in v_parts)
        v = np.zeros((tot_size, tot_size), dtype=float)
        cursor = 0
        for block_v in v_parts:
            size = block_v.shape[0]
            v[cursor : cursor + size, cursor : cursor + size] = block_v
            cursor += size

        return y, x, v

    def _optimize_tau_reml(self, y: np.ndarray, x: np.ndarray, v: np.ndarray) -> float:
        # Objective: minimize negative REML log-likelihood
        # Brent's method is ideal for univariate optimization
        def obj(t):
            return self._reml_nll(t, y, x, v)

        res = minimize_scalar(obj, bounds=(0.0, 10.0), method="bounded")
        if res.success:
            return float(max(res.x, 0.0))
        return 0.0

    @staticmethod
    def _reml_nll(tau: float, y: np.ndarray, x: np.ndarray, v: np.ndarray) -> float:
        m = v + np.eye(v.shape[0], dtype=float) * (tau * tau)
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

    @staticmethod
    def _estimate_gls_with_shrinkage(
        y: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        tau: float,
        param_names: tuple[str, ...],
        bias_prior_sd: float,
        blocks: list[_StudyBlock],
        variance_type: str = "model"
    ) -> tuple[np.ndarray, np.ndarray]:
        m = v + np.eye(v.shape[0], dtype=float) * (tau * tau)
        m_inv = np.linalg.inv(m)

        xt_m_inv = x.T @ m_inv
        info = xt_m_inv @ x

        # Prior precision matrix (P) for ridge penalty
        # Applied to bias and interaction terms, but NOT to main treatment effects
        p_matrix = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        bias_precision = 1.0 / (bias_prior_sd * bias_prior_sd)

        for idx, name in enumerate(param_names):
            if name.startswith("bias_") or "_x_" in name:
                p_matrix[idx, idx] = bias_precision

        post_info = info + p_matrix
        cov_model = np.linalg.pinv(post_info)
        beta = cov_model @ (xt_m_inv @ y)

        if variance_type == "sandwich":
            # Meat: Sum_s X_s^T M_s^-1 e_s e_s^T M_s^-1 X_s
            meat = np.zeros_like(info)
            cursor = 0
            for block in blocks:
                size = len(block.y)
                m_s = block.v + np.eye(size, dtype=float) * (tau * tau)
                m_s_inv = np.linalg.inv(m_s)

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
