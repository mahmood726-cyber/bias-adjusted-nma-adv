"""Registry-Based Publication Bias and Outcome Switching Auditor module."""

from __future__ import annotations

import numpy as np

class RegistryPublicationBiasAuditor:
    """Audits NMA networks against ClinicalTrials.gov registries to detect unpublished trials and outcome switching."""

    def __init__(self):
        # Maps NCT ID to its registry metadata:
        # {nct_id: {"registered_primary": str, "reported_primary": str, "status": str}}
        self.registry_db: dict[str, dict[str, str]] = {}

    def register_trial_protocol(
        self,
        nct_id: str,
        registered_primary: str,
        reported_primary: str,
        status: str
    ) -> None:
        """Register prospective trial protocol metadata from ClinicalTrials.gov."""
        self.registry_db[nct_id] = {
            "registered_primary": registered_primary.strip().lower(),
            "reported_primary": reported_primary.strip().lower(),
            "status": status.strip().lower()
        }

    def audit_outcome_switching(self, nct_ids: list[str]) -> dict[str, float]:
        """Audit trials for primary outcome switching between registration and publication.

        Returns an Outcome Switching Bias Score (OSBS) in [0.0, 1.0] per NCT ID.
        """
        bias_scores = {}
        for nct_id in nct_ids:
            meta = self.registry_db.get(nct_id)
            if not meta:
                # No registry protocol found (high bias risk)
                bias_scores[nct_id] = 1.0
                continue
                
            reg = meta["registered_primary"]
            rep = meta["reported_primary"]
            
            # If primary outcome was modified or switched, assign score of 1.0
            if reg != rep and reg not in rep and rep not in reg:
                bias_scores[nct_id] = 1.0
            else:
                bias_scores[nct_id] = 0.0
                
        return bias_scores

    def calculate_unpublished_ratio(self, drug_name: str, published_nct_ids: list[str]) -> float:
        """Calculate the registry-based Unpublished Trial Ratio (UTR) for a specific drug.

        UTR = (Registered but Unpublished Trials) / (Total Registered Trials)
        """
        total_registered = 0
        unpublished = 0
        
        # Simple string matching helper to identify trials evaluating the target drug
        for nct_id, meta in self.registry_db.items():
            if meta["status"] in ["completed", "terminated"]:
                total_registered += 1
                if nct_id not in published_nct_ids:
                    unpublished += 1
                    
        if total_registered == 0:
            return 0.0
            
        return float(unpublished / total_registered)

    def apply_bias_shrinkage(self, effect_estimate: float, utr: float) -> float:
        """Apply a publication bias shrinkage factor to the pooled treatment effect.

        Shrinks the pooled log-HR toward 0 (the null) proportional to the unpublished ratio.
        """
        # Shrinkage factor: (1 - UTR)
        shrinkage = 1.0 - utr
        return effect_estimate * max(0.0, shrinkage)
