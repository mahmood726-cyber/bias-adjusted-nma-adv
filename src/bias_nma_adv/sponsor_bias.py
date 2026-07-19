"""Registry-Based Sponsorship Bias and Attrition Weighting module."""

from __future__ import annotations

import numpy as np


class SponsorClassError(ValueError):
    """Raised when a sponsor class cannot be classified and strict mode is on."""


#: Classes that count as industry sponsorship (bias score 1.0).
INDUSTRY_SPONSOR_CLASSES: frozenset[str] = frozenset({"industry"})

#: Classes that are affirmatively NON-industry (bias score 0.0). Only an
#: explicit member of this set earns the clean score. AACT's ambiguous buckets
#: -- "other", "unknown", "ambig", "network", "" -- are deliberately absent:
#: they do not establish independence, so they must not be scored as if they do.
NON_INDUSTRY_SPONSOR_CLASSES: frozenset[str] = frozenset(
    {"nih", "fed", "other_gov", "indiv", "academic", "non_industry"}
)


class RegistrySponsorAuditor:
    """Audits trial sponsorship classes and participant flow attrition to adjust quality weights.

    Sponsor classification FAILS CLOSED. `sponsor_class` was previously matched
    by exact equality against "industry", so every other string -- including
    AACT's `OTHER`, `NETWORK`, `UNKNOWN`, `AMBIG`, a typo, or "" -- fell through
    to the BEST possible score of 0.0. That made a trial registered with untidy
    metadata score better than one not registered at all (unregistered already
    returned 1.0), which is a perverse incentive: it rewarded bad metadata.

    Now only an affirmative member of NON_INDUSTRY_SPONSOR_CLASSES scores 0.0.
    Anything unrecognised scores 1.0 (same as unregistered) and is recorded in
    `unrecognised_sponsor_classes` so the substitution is visible rather than
    silent. Pass `strict=True` to raise instead.
    """

    def __init__(self, strict: bool = False):
        # Maps NCT ID to its registry metadata:
        # {nct_id: {"sponsor_class": str, "randomized": int, "lost_to_follow_up": int}}
        self.registry_db: dict[str, dict[str, str | int]] = {}
        self.strict = bool(strict)
        #: nct_id -> the raw class string that could not be classified.
        self.unrecognised_sponsor_classes: dict[str, str] = {}

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
        normalised = sponsor_class.strip().lower()
        if (
            normalised not in INDUSTRY_SPONSOR_CLASSES
            and normalised not in NON_INDUSTRY_SPONSOR_CLASSES
        ):
            if self.strict:
                raise SponsorClassError(
                    f"{nct_id}: sponsor_class {sponsor_class!r} is not a recognised class. "
                    f"Industry: {sorted(INDUSTRY_SPONSOR_CLASSES)}; "
                    f"non-industry: {sorted(NON_INDUSTRY_SPONSOR_CLASSES)}. "
                    "Ambiguous registry values (e.g. 'other', 'unknown', 'network') do not "
                    "establish independence and are scored as high risk."
                )
            self.unrecognised_sponsor_classes[nct_id] = normalised
        self.registry_db[nct_id] = {
            "sponsor_class": normalised,
            "randomized": randomized,
            "lost_to_follow_up": lost_to_follow_up
        }

    def sponsor_class_status(self, nct_id: str) -> str:
        """Return industry | non_industry | unrecognised | unregistered."""
        meta = self.registry_db.get(nct_id)
        if not meta:
            return "unregistered"
        value = str(meta["sponsor_class"])
        if value in INDUSTRY_SPONSOR_CLASSES:
            return "industry"
        if value in NON_INDUSTRY_SPONSOR_CLASSES:
            return "non_industry"
        return "unrecognised"

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
        status = self.sponsor_class_status(nct_id)
        if status == "non_industry":
            return 0.0
        # industry / unrecognised / unregistered all score high risk. Absent or
        # ambiguous provenance must never read as clean provenance.
        return 1.0

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
