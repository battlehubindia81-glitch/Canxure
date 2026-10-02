
from __future__ import annotations

from typing import Any

from src.simulation.real_data_adapter import RealDataSimulationAdapter


class CanxureRealDataAPI:
    """
    Thin backend-facing service around the production real-data adapter.

    This layer does not modify:
    - ResearchDigitalTwin architecture
    - model weights
    - source datasets
    - legacy VirtualSimulationChamber

    It only exposes validated adapter operations in API-safe dictionaries.
    """

    def __init__(
        self,
        base_dir: str,
        project_dir: str,
        checkpoint_name: str = "epoch_015.pt",
        device: str | None = None,
    ) -> None:
        self.adapter = RealDataSimulationAdapter(
            base_dir=base_dir,
            project_dir=project_dir,
            checkpoint_name=checkpoint_name,
            device=device,
        )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """
        Convert numpy/PyTorch scalar values into ordinary Python values.
        """
        if hasattr(value, "item"):
            return value.item()

        if isinstance(value, dict):
            return {
                str(k): CanxureRealDataAPI._json_safe(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                CanxureRealDataAPI._json_safe(v)
                for v in value
            ]

        return value

    def health(self) -> dict[str, Any]:
        """
        Basic service health/status information.
        """
        return {
            "service": "Canxure Real-Data API",
            "status": "ok",
            "model": type(self.adapter.model).__name__,
            "checkpoint": "epoch_015.pt",
            "sample_count": len(self.adapter.available_sample_ids()),
            "drug_count": len(self.adapter.available_drug_ids()),
        }

    def list_samples(self) -> dict[str, Any]:
        return {
            "sample_count": len(self.adapter.available_sample_ids()),
            "sample_ids": [
                str(x) for x in self.adapter.available_sample_ids()
            ],
        }

    def list_drugs(self) -> dict[str, Any]:
        return {
            "drug_count": len(self.adapter.available_drug_ids()),
            "drug_ids": [
                str(x) for x in self.adapter.available_drug_ids()
            ],
        }

    def predict_pair(
        self,
        sample_id: str,
        drug_id: str,
    ) -> dict[str, Any]:
        result = self.adapter.predict_pair(
            sample_id=sample_id,
            drug_id=drug_id,
        )

        return self._json_safe(result)

    def predict_drugs(
        self,
        sample_id: str,
        drug_ids: list[str],
    ) -> list[dict[str, Any]]:
        results = self.adapter.predict_drugs(
            sample_id=sample_id,
            drug_ids=drug_ids,
        )

        return [
            self._json_safe(result)
            for result in results
        ]


def create_api(
    base_dir: str = "/content/drive/MyDrive/Canxure_backup",
    project_dir: str = "/content/Canxure",
    checkpoint_name: str = "epoch_015.pt",
    device: str | None = None,
) -> CanxureRealDataAPI:
    """
    Factory used by the eventual HTTP server/frontend boundary.
    """
    return CanxureRealDataAPI(
        base_dir=base_dir,
        project_dir=project_dir,
        checkpoint_name=checkpoint_name,
        device=device,
    )
