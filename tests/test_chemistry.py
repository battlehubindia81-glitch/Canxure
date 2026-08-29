import pytest

from src.chemistry.fingerprints import smiles_to_fingerprint, validate_smiles


def test_fingerprint_shape():
    fp = smiles_to_fingerprint("CCO")
    assert fp.shape == (1024,)
    assert fp.sum() > 0


def test_invalid_smiles():
    with pytest.raises(ValueError):
        smiles_to_fingerprint("")
