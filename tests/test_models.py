import torch

from src.models.metrics import count_parameters, count_trainable_parameters
from src.models.research_model import ResearchDigitalTwin
from src.training.checkpointing import load_checkpoint, save_checkpoint


def test_model_forward_and_parameters(tmp_path):
    model = ResearchDigitalTwin()
    out = model(torch.randn(2, 500), torch.randn(2, 64), torch.randn(2, 64), torch.randn(2, 1024))
    assert out["efficacy"].shape == (2, 1)
    assert count_parameters(model) == count_trainable_parameters(model)
    assert 2_000_000 < count_parameters(model) < 10_000_000
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, 1, 0.5, {})
    loaded = load_checkpoint(path, model)
    assert loaded["epoch"] == 1
