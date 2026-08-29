from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for neural digital twin models") from exc


class MLP(nn.Sequential):
    def __init__(self, dims: list[int], dropout: float = 0.1):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)])
        super().__init__(*layers)


class ResearchDigitalTwin(nn.Module):
    """Multimodal tumor-drug network with expression, mutation, biology, and drug encoders."""

    def __init__(self, expression_dim: int = 500, mutation_dim: int = 64, biology_dim: int = 64, drug_dim: int = 1024, clinical_classes: int = 2):
        super().__init__()
        self.expression_encoder = MLP([expression_dim, 1024, 512])
        self.mutation_encoder = MLP([mutation_dim, 512, 256])
        self.biology_encoder = MLP([biology_dim, 512, 256])
        self.drug_encoder = MLP([drug_dim, 512, 256])
        self.fusion = MLP([1280, 768, 512])
        self.efficacy = MLP([512, 128, 1])
        self.toxicity = MLP([512, 128, 1])
        self.resistance = MLP([512, 128, 1])
        self.clinical_response = MLP([512, 128, clinical_classes])

    def forward(self, expression: torch.Tensor, mutation: torch.Tensor, biology: torch.Tensor, drug: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run batched multimodal inference.

        Purpose: predict research-only tumor-drug task outputs.
        Inputs: expression (B,500), mutation (B,64), biology (B,64), drug (B,1024).
        Outputs: dict tensors for efficacy, toxicity, resistance, clinical logits.
        Output shape: (B,1) regression/probability logits and (B,C) clinical logits.
        Data types: torch.float32 tensors.
        Exceptions: PyTorch dimension errors for incompatible tensors.
        Assumptions: outputs are computational hypotheses, not clinical recommendations.
        """
        fused = self.fusion(torch.cat([self.expression_encoder(expression), self.mutation_encoder(mutation), self.biology_encoder(biology), self.drug_encoder(drug)], dim=1))
        return {
            "efficacy": self.efficacy(fused),
            "toxicity": torch.sigmoid(self.toxicity(fused)),
            "resistance": torch.sigmoid(self.resistance(fused)),
            "clinical_response": torch.softmax(self.clinical_response(fused), dim=1),
        }
