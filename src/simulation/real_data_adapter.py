
from __future__ import annotations
import sys

from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
import torch


class RealDataSimulationAdapter:
    """
    Production bridge between real Canxure feature artifacts and
    ResearchDigitalTwin.

    Inputs:
        sample_id : real DepMap/Canxure cell-line ID
        drug_id   : real validated GDSC drug ID

    Internal inputs:
        expression : (1, 500)
        mutation   : (1, 64)
        biology    : (1, 64)
        drug       : (1, 1024)

    Outputs:
        efficacy
        toxicity
        resistance
        clinical_response logits

    This adapter does NOT modify the model architecture and performs
    no training.
    """

    def __init__(
        self,
        base_dir: str | Path,
        project_dir: str | Path,
        checkpoint_name: str = "epoch_015.pt",
        device: str | torch.device | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.project_dir = Path(project_dir)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ----------------------------------------------------
        # ARTIFACT PATHS
        # ----------------------------------------------------
        self.expression_file = (
            self.base_dir
            / "data"
            / "processed"
            / "real_expression_500d.npz"
        )

        self.mutation_file = (
            self.base_dir
            / "data"
            / "processed"
            / "real_mutation_64d.npz"
        )

        self.biology_file = (
            self.base_dir
            / "data"
            / "processed"
            / "real_biology_64d.npz"
        )

        self.drug_file = (
            self.base_dir
            / "data"
            / "features"
            / "real_gdsc_drug_fingerprints_1024d.npz"
        )

        self.checkpoint_file = (
            self.base_dir
            / "models"
            / "research"
            / "real_data_checkpoints"
            / checkpoint_name
        )

        # ----------------------------------------------------
        # VERIFY ARTIFACTS
        # ----------------------------------------------------
        required = {
            "expression": self.expression_file,
            "mutation": self.mutation_file,
            "biology": self.biology_file,
            "drug": self.drug_file,
            "checkpoint": self.checkpoint_file,
        }

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required Canxure production artifacts:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------------
        # LOAD CELL-LINE FEATURES
        # ----------------------------------------------------
        expression_npz = np.load(
            self.expression_file,
            allow_pickle=True
        )
        mutation_npz = np.load(
            self.mutation_file,
            allow_pickle=True
        )
        biology_npz = np.load(
            self.biology_file,
            allow_pickle=True
        )
        drug_npz = np.load(
            self.drug_file,
            allow_pickle=True
        )

        self.expression = expression_npz["expression"].astype(
            np.float32,
            copy=False
        )
        self.expression_ids = np.asarray(
            expression_npz["sample_ids"]
        ).astype(str)

        self.mutation = mutation_npz["mutation"].astype(
            np.float32,
            copy=False
        )
        self.mutation_ids = np.asarray(
            mutation_npz["sample_ids"]
        ).astype(str)

        self.biology = biology_npz["biology"].astype(
            np.float32,
            copy=False
        )
        self.biology_ids = np.asarray(
            biology_npz["sample_ids"]
        ).astype(str)

        self.drug = drug_npz["fingerprints"].astype(
            np.float32,
            copy=False
        )
        self.drug_ids = np.asarray(
            drug_npz["drug_ids"]
        ).astype(str)

        # ----------------------------------------------------
        # SHAPE VALIDATION
        # ----------------------------------------------------
        if self.expression.shape != (696, 500):
            raise ValueError(
                f"Unexpected expression shape: {self.expression.shape}"
            )

        if self.mutation.shape != (696, 64):
            raise ValueError(
                f"Unexpected mutation shape: {self.mutation.shape}"
            )

        if self.biology.shape != (696, 64):
            raise ValueError(
                f"Unexpected biology shape: {self.biology.shape}"
            )

        if self.drug.shape != (421, 1024):
            raise ValueError(
                f"Unexpected drug fingerprint shape: {self.drug.shape}"
            )

        # ----------------------------------------------------
        # BUILD ID LOOKUPS
        # ----------------------------------------------------
        self.expression_index = {
            sample_id: idx
            for idx, sample_id in enumerate(self.expression_ids)
        }

        self.mutation_index = {
            sample_id: idx
            for idx, sample_id in enumerate(self.mutation_ids)
        }

        self.biology_index = {
            sample_id: idx
            for idx, sample_id in enumerate(self.biology_ids)
        }

        self.drug_index = {
            drug_id: idx
            for idx, drug_id in enumerate(self.drug_ids)
        }

        # ----------------------------------------------------
        # CROSS-MODALITY VALIDATION
        # ----------------------------------------------------
        expression_set = set(self.expression_ids)
        mutation_set = set(self.mutation_ids)
        biology_set = set(self.biology_ids)

        common = (
            expression_set
            & mutation_set
            & biology_set
        )

        if len(common) != 696:
            raise ValueError(
                f"Expected 696 common cell lines, found {len(common)}"
            )

        if len(self.drug_index) != 421:
            raise ValueError(
                "Drug IDs are not unique."
            )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------
        if str(self.project_dir) not in sys.path:
            sys.path.insert(0, str(self.project_dir))

        from src.models.research_model import ResearchDigitalTwin

        self.model = ResearchDigitalTwin(
            expression_dim=500,
            mutation_dim=64,
            biology_dim=64,
            drug_dim=1024,
            clinical_classes=2,
        ).to(self.device)

        checkpoint = torch.load(
            self.checkpoint_file,
            map_location=self.device,
            weights_only=False,
        )

        load_result = self.model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        if load_result.missing_keys:
            raise RuntimeError(
                f"Checkpoint missing keys: {load_result.missing_keys}"
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint unexpected keys: "
                f"{load_result.unexpected_keys}"
            )

        self.model.eval()

        self.checkpoint_epoch = int(
            checkpoint.get("epoch", 15)
        )

    # --------------------------------------------------------
    # PUBLIC RESOLUTION METHODS
    # --------------------------------------------------------
    def available_sample_ids(self) -> list[str]:
        return sorted(
            set(self.expression_ids)
            & set(self.mutation_ids)
            & set(self.biology_ids)
        )

    def available_drug_ids(self) -> list[str]:
        return sorted(self.drug_index.keys())

    # --------------------------------------------------------
    # INTERNAL RESOLUTION
    # --------------------------------------------------------
    def _resolve_sample(self, sample_id: str):
        sample_id = str(sample_id)

        if sample_id not in self.expression_index:
            raise ValueError(
                f"Unknown sample_id: {sample_id}"
            )

        if sample_id not in self.mutation_index:
            raise ValueError(
                f"Unknown sample_id in mutation data: {sample_id}"
            )

        if sample_id not in self.biology_index:
            raise ValueError(
                f"Unknown sample_id in biology data: {sample_id}"
            )

        expr = self.expression[
            self.expression_index[sample_id]
        ]

        mut = self.mutation[
            self.mutation_index[sample_id]
        ]

        bio = self.biology[
            self.biology_index[sample_id]
        ]

        return expr, mut, bio

    def _resolve_drug(self, drug_id: str):
        drug_id = str(drug_id)

        if drug_id not in self.drug_index:
            raise ValueError(
                f"Unknown drug_id: {drug_id}"
            )

        return self.drug[
            self.drug_index[drug_id]
        ]

    # --------------------------------------------------------
    # SINGLE-PAIR PREDICTION
    # --------------------------------------------------------
    def predict_pair(
        self,
        sample_id: str,
        drug_id: str,
    ) -> Dict[str, Any]:

        expr, mut, bio = self._resolve_sample(sample_id)
        drug = self._resolve_drug(drug_id)

        expression_tensor = torch.from_numpy(
            expr
        ).unsqueeze(0).to(self.device)

        mutation_tensor = torch.from_numpy(
            mut
        ).unsqueeze(0).to(self.device)

        biology_tensor = torch.from_numpy(
            bio
        ).unsqueeze(0).to(self.device)

        drug_tensor = torch.from_numpy(
            drug
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                expression_tensor,
                mutation_tensor,
                biology_tensor,
                drug_tensor,
            )

        efficacy = float(
            outputs["efficacy"]
            .view(-1)[0]
            .detach()
            .cpu()
            .item()
        )

        toxicity = float(
            outputs["toxicity"]
            .view(-1)[0]
            .detach()
            .cpu()
            .item()
        )

        resistance = float(
            outputs["resistance"]
            .view(-1)[0]
            .detach()
            .cpu()
            .item()
        )

        clinical = (
            outputs["clinical_response"]
            .view(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        return {
            "sample_id": sample_id,
            "drug_id": drug_id,
            "predicted_log2_auc": efficacy,
            "toxicity": toxicity,
            "resistance": resistance,
            "clinical_response_logit_0": clinical[0],
            "clinical_response_logit_1": clinical[1],
            "checkpoint_epoch": self.checkpoint_epoch,
        }

    # --------------------------------------------------------
    # MULTI-DRUG PREDICTION
    # --------------------------------------------------------
    def predict_drugs(
        self,
        sample_id: str,
        drug_ids: Iterable[str] | None = None,
    ) -> list[Dict[str, Any]]:

        if drug_ids is None:
            drug_ids = self.available_drug_ids()

        drug_ids = [str(x) for x in drug_ids]

        results = [
            self.predict_pair(sample_id, drug_id)
            for drug_id in drug_ids
        ]

        return results
