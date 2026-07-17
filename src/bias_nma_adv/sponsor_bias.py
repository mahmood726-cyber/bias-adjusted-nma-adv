"""Registry-Based Sponsorship Bias and Attrition Weighting module."""

from __future__ import annotations

import numpy as np

class RegistrySponsorAuditor:
    """Audits trial sponsorship classes and participant flow attrition to adjust quality weights."""

    def __init__(self):
        # Maps NCT ID to its registry metadata:
        # {nct_id: {"sponsor_class": str, "randomized": int, "lost_to_follow_up": int}}
        self.registry_db: dict[str, dict[str, str | int]] = {}

    def register_trial_flow(
        self,
        nct_id: str,
        sponsor_class: str,
        randomized: int,
        lost_to_follow_up: int
    ) -> None:
        """Register trial collaborator class and participant flow denominators."""
        if randomized <= 0:
            raise ValueError("randomized must be positive.")
        if lost_to_follow_up < 0 or lost_to_follow_up > randomized:
            raise ValueError("lost_to_follow_up must satisfy 0 <= lost_to_follow_up <= randomized.")
        self.registry_db[nct_id] = {
            "sponsor_class": sponsor_class.strip().lower(),
            "randomized": randomized,
            "lost_to_follow_up": lost_to_follow_up
        }

    def calculate_attrition_ratio(self, nct_id: str) -> float:
        """Calculate the Loss-to-Follow-Up Attrition Ratio (LAR).

        LAR = (Lost to Follow-up) / (Randomized)
        """
        meta = self.registry_db.get(nct_id)
        if not meta or meta["randomized"] == 0:
            return 1.0
            
        return float(meta["lost_to_follow_up"] / meta["randomized"])

    def calculate_sponsorship_bias_score(self, nct_id: str) -> float:
        """Assign a bias score based on funding collaborator class.

        Industry-sponsored trials are assigned 1.0 (higher bias risk),
        independent/NIH sponsored trials are assigned 0.0.
        """
        meta = self.registry_db.get(nct_id)
        if not meta:
            return 1.0 # High risk if unlogged
            
        if meta["sponsor_class"] == "industry":
            return 1.0
        return 0.0

    def adjust_doi_welton_quality(
        self,
        nct_id: str,
        base_quality: float
    ) -> float:
        """Down-weight the study quality score based on industry funding and high attrition.

        Reduces base quality by 20% for industry funding and up to 30% for high attrition (>5%).
        """
        lar = self.calculate_attrition_ratio(nct_id)
        sbs = self.calculate_sponsorship_bias_score(nct_id)
        
        quality = base_quality
        
        # 1. Sponsor adjustment
        if sbs == 1.0:
            quality *= 0.80
            
        # 2. Attrition adjustment (penalize if LAR > 0.05)
        if lar > 0.05:
            penalty = min(0.30, (lar - 0.05) * 2.0)
            quality *= (1.0 - penalty)
            
        return max(0.1, float(quality))
