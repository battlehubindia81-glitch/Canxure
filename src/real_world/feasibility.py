from __future__ import annotations


class RealWorldFeasibilityEngine:
    """Offline-safe feasibility layer separated from biological prediction."""

    def assess(self, drug_name: str, country: str = "India", offline: bool = True) -> dict[str, object]:
        """Return provenance-preserving feasibility placeholders unless verified data exist.

        Purpose: avoid fabricating approvals, prices, or trial eligibility.
        Inputs: drug name and country. Input shape: scalar strings.
        Outputs: status mapping. Output shape: one record.
        Data types: dict of strings/bools/counts.
        Exceptions: none.
        Assumptions: offline mode cannot verify authoritative regulatory data.
        """
        return {
            "regulatory_status": "DATA_UNAVAILABLE" if offline else "UNKNOWN",
            "availability_status": "UNKNOWN",
            "affordability_category": "PRICE_DATA_UNAVAILABLE",
            "generic_available": None,
            "biosimilar_available": None,
            "clinical_trial_count": 0,
            "provenance": {"source": "offline fallback", "country": country, "drug_name": drug_name},
        }
