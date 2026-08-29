from src.data.demo import synthetic_expression
from src.data.preprocessing import preprocess_expression


def test_preprocess_expression_500_genes():
    out, meta = preprocess_expression(synthetic_expression(6, 650))
    assert out.shape == (6, 501)
    assert "EGFR" in meta["selected_genes"]
